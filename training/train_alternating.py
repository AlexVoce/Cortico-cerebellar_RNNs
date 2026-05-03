# training/train_alternating.py
import os
import numpy as np
import torch
import torch.nn as nn
import sys

# --- Path Setup ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
alex_utils_path = os.path.dirname(__file__)
if alex_utils_path not in sys.path:
    sys.path.insert(0, alex_utils_path)

# --- Imports ---
from src.utils.save import save_model, find_next_free_network_number
from alex_utils import get_grad_norms, set_active_module,step_optimizer
from alex_training.variants import (
    variant_cb_then_rnn,
    variant_rnn_then_cb_finetune,
    variant_interleaved_finetune,
    variant_interleaved_curr,
    variant_cb_only,
    variant_cb_only_reservoir,
    variant_train_simultaneous,
)
from alex_training.rflo import init_rflo_state, rflo_step
from alex_training.base import head_idx_factory, get_eat_lambda
from src.tasks.task_registry import compute_loss


VARIANTS = {
    "cb_then_rnn": variant_cb_then_rnn,
    "rnn_then_cb_finetune": variant_rnn_then_cb_finetune,
    "interleaved_finetune": variant_interleaved_finetune,
    "interleaved_curr": variant_interleaved_curr,
    "cb_only": variant_cb_only,
    "cb_only_reservoir": variant_cb_only_reservoir,
    "train_simultaneous": variant_train_simultaneous,
}


def _scheduled_cb_lr(base_cb_lr, current_n, n_start, decay=0.05, min_frac=0.1):
    """
    Continual inverse-time decay with N (no max-N normalization):
      lr(N) = base_cb_lr / (1 + decay * (N - n_start))
    Clipped to min_frac * base_cb_lr.
    """
    step = max(0, int(current_n) - int(n_start))
    lr = float(base_cb_lr) / (1.0 + float(decay) * step)
    return max(float(base_cb_lr) * float(min_frac), lr)


def _set_cb_lr(optimizer, new_lr):
    if isinstance(optimizer, dict):
        cb_opt = optimizer.get("CB")
        if cb_opt is None:
            return
        for group in cb_opt.param_groups:
            group["lr"] = float(new_lr)
        return

    for group in optimizer.param_groups:
        if group.get("name") == "CB":
            group["lr"] = float(new_lr)
            return

def safe_log_and_save(row, subdir, stats, save_every=10):
    """
    1. Maps variant keys (short) to stats keys (long).
    2. Updates stats in-memory.
    3. Safely saves to disk.
    """
    # MAP KEYS: variant output -> stats storage
    # If the key exists in row, map it. Otherwise ignore.
    mapping = {
        "N": "n_task",
        "acc": "accuracy",
        "gRNN": "grad_rnn",
        "gCB": "grad_cb",
        "gRNN_pre": "grad_rnn_pre",
        "gCB_pre": "grad_cb_pre",
        "task_loss": "task_loss",
        "eat_loss": "eat_loss",
        "loss": "loss",
        "phase": "phase",
        "epoch": "epoch",
        "task": "task",
        "stage": "stage",
        # "ht_norm_mean": "ht_norm_mean",
        # "ht_norm_max": "ht_norm_max",
        # "cb_norm_mean": "cb_norm_mean",
        # "cb_norm_max": "cb_norm_max",
        # "pre_norm_mean": "pre_norm_mean",
        # "pre_norm_max": "pre_norm_max",
        # "post_norm_mean": "post_norm_mean",
        # "post_norm_max": "post_norm_max",
        # "nonfinite_ht": "nonfinite_ht",
        # "nonfinite_cb": "nonfinite_cb",
        # "nonfinite_pre": "nonfinite_pre",
        # "nonfinite_post": "nonfinite_post",
        # "max_abs_logit": "max_abs_logit",
    }

    # Update global stats object
    for row_k, stats_k in mapping.items():
        if row_k in row and stats_k in stats:
            stats[stats_k].append(row[row_k])
            
    # Save to disk periodically
    if row.get('epoch', 0) % save_every == 0:
        final_path = os.path.join(subdir, 'stats.npy')
        tmp_path = os.path.join(subdir, 'stats_tmp.npy')
        try:
            np.save(tmp_path, stats)
            os.replace(tmp_path, final_path)
        except Exception as e:
            print(f"Warning: Failed to save stats: {e}", flush=True)

def train_alternating(
    model, curriculum_type, task_function, num_epochs, Ns_init, run_number,
    batch_size, training_steps, test_steps,
    device, base_path, affixes,
    n_heads=1, n_forget=1, task_name="dms",
    scramble=False, rnn_lr=0.05, cb_lr=0.05,
    readout_head_dyn="sliding", cb_store=False,
    alt_variant="cb_then_rnn", args=None, shared_optimiser=True,
    threshold_cb_lead=60.0, threshold_rnn_lead=80.0, threshold_final=98.0, learning_alg='bptt',
    spec=None, cb_l2=0.0, cb_l1=0.0, rnn_eat=False, rnn_eat_lambda=0.1, rnn_eat_loss_type='hidden', target_end_n=150, 
    cb_schedule=False, subdir_override=None, stage_tag=None,
    skip_init_phase=False,
    reservoir_interval_n=10,
    **kwargs
):
    if spec is None:
        raise ValueError("train_alternating requires task spec (spec=...)")
    DEBUG = False
    THRESHOLD_BASE = threshold_final
    MAX_STAGE_EPOCHS = 100 if DEBUG else num_epochs
    MAX_GLOBAL_EPOCHS = num_epochs # using subdir override to mark task switching means we want to allow more epochs since we are not starting from scratch
    TARGET_END_N = int(target_end_n) if target_end_n is not None else 150
    criterion = spec["criterion_ctor"]()

    if alt_variant not in VARIANTS:
        raise ValueError(f"Unknown alt_variant='{alt_variant}'. Options: {list(VARIANTS.keys())}")
    variant_fn = VARIANTS[alt_variant]
    if readout_head_dyn == "single":
        head_idx = lambda n: 0  # always use head 0
    else:
        head_idx = head_idx_factory(Ns_init, num_heads=len(model.heads))

    # ---- init tracking ----
    current_N = Ns_init[0]
    active_Ns = [current_N]
    active_Ns = sorted(active_Ns)  # ensure active_Ns is always sorted for consistent head indexing
    global_epoch = 0
    stats = {
        "stage": [], "task": [], "n_task": [], "phase": [], "loss": [], "accuracy": [],
        "grad_rnn": [], "grad_cb": [], "epoch": [], "grad_rnn_pre": [], "grad_cb_pre": [], "task_loss": [], "eat_loss": [],
        # "ht_norm_mean": [], "ht_norm_max": [],
        # "cb_norm_mean": [], "cb_norm_max": [], 
        # "pre_norm_mean": [], "pre_norm_max": [],
        # "post_norm_mean": [], "post_norm_max": [], 
        # "nonfinite_ht": [], "nonfinite_cb": [], "nonfinite_pre": [], "nonfinite_post": [],
        # "max_abs_logit": [],
    }
    # ---- saving ----
    if subdir_override is None:
        run_number = find_next_free_network_number(
            base_path=base_path, curriculum_type=curriculum_type, task=task_name,
            affixes=affixes, n_heads=n_heads, n_forget=n_forget
        )
        subdir = save_model(
            model, curriculum_type, n_heads, n_forget, task_name,
            run_number, current_N, current_N, args=args, base_path=base_path, affixes=affixes
        )
    else:
        subdir = subdir_override
        os.makedirs(subdir, exist_ok=True)
    if getattr(args, 'resume_ckpt', None) and subdir_override is not None:
        stats_path = os.path.join(subdir, 'stats.npy')
        if os.path.exists(stats_path):
            try:
                old_stats = np.load(stats_path, allow_pickle=True).item()
                for k in stats.keys():
                    if k in old_stats and isinstance(old_stats[k], list):
                        stats[k] = list(old_stats[k])
                print(f"[resume] Loaded existing alternating stats from {stats_path} (rows={len(stats['n_task'])})", flush=True)
            except Exception as e:
                print(f"[resume] Warning: failed to load existing alternating stats ({e}); starting fresh stats in same folder.", flush=True)
    print(f"[train_alternating] Saving results to: {subdir}", flush=True)
    # ---- optimizer groups (PERSIST ON MODEL) ----
    # Reuse optimizer across task-switch stages (we use subdir_override as the marker)
    reuse_ok = (subdir_override is not None) and hasattr(model, "_task_switch_optimizer")

    if reuse_ok:
        optimizer = model._task_switch_optimizer
        print("[train_alternating] Reusing existing optimizer (task-switch mode)", flush=True)
    else:
        cb_params = list(model.cb.parameters()) if (hasattr(model, "cb") and model.cb) else []
        cb_param_ids = {id(p) for p in cb_params}
        rnn_params = [p for p in model.parameters() if id(p) not in cb_param_ids]

        if shared_optimiser:
            optimizer = torch.optim.SGD(
                [
                    {"params": rnn_params, "lr": rnn_lr, "name": "RNN"},
                    {"params": cb_params, "lr": cb_lr, "name": "CB"},
                ],
                momentum=0.1, nesterov=True
            )
        else:
            optimizer_rnn = torch.optim.SGD(rnn_params, lr=rnn_lr, momentum=0.1, nesterov=True)
            optimizer_cb = torch.optim.SGD(cb_params, lr=cb_lr, momentum=0.1, nesterov=True) if cb_params else None
            optimizer = {"RNN": optimizer_rnn, "CB": optimizer_cb}

    if cb_schedule:
        cb_decay = float(kwargs.get("cb_lr_decay", 0.05))
        cb_min_frac = float(kwargs.get("cb_lr_min_frac", 0.1))
        stage_cb_lr = _scheduled_cb_lr(cb_lr, current_N, Ns_init[0], decay=cb_decay, min_frac=cb_min_frac)
        _set_cb_lr(optimizer, stage_cb_lr)
        print(
            f"[cb_schedule] N={current_N} -> CB lr={stage_cb_lr:.6g} "
            f"(base={cb_lr:.6g}, decay={cb_decay}, min_frac={cb_min_frac})",
            flush=True,
        )

    # only persist it for task-switch stages
    if subdir_override is not None and not reuse_ok:
        model._task_switch_optimizer = optimizer
        print("[train_alternating] Created + stored optimizer (task-switch mode)", flush=True)

    # ---- Phase 0: train base with RNN_ONLY until solved ----
    phase = "RNN_ONLY"
    set_active_module(model, phase)
    trainable = [(n, p.requires_grad) for n, p in model.named_parameters()]
    print("Trainable params after set_active_module:")
    for n, rg in trainable:
        if rg:
            print("  ", n)
    # Resume/task-switch mode and optional user flag can skip Phase 0.
    if subdir_override is not None:
        print(f"[resume] Phase 0 skipped; resuming directly with variants from N={current_N}", flush=True)
        goto_alternating = True
    elif skip_init_phase:
        print(f"[skip_init_phase] Phase 0 skipped; starting alternating variant from N={current_N}", flush=True)
        goto_alternating = True
    else:
        goto_alternating = False
    if not goto_alternating:   
        print(f"=== Starting Phase 0 (Base) on N={current_N} ===", flush=True)
        for _ep in range(num_epochs):
            global_epoch += 1
            if global_epoch > MAX_GLOBAL_EPOCHS:
                np.save(os.path.join(subdir, "stats.npy"), stats)
                return stats

            # Training
            model.train()
            losses, grs, gcs = [], [], []
            # debug storage
            grs_preclip, gcs_preclip, task_loss_step, eat_loss_step = [],[],[],[]
            for _ in range(training_steps):
                if learning_alg != 'rflo':
                    if shared_optimiser:
                        optimizer.zero_grad()
                    else:
                        for opt in optimizer.values():
                            if opt:
                                opt.zero_grad()
                seq, labels = task_function(active_Ns, batch_size)
                labels = labels if isinstance(labels, list) else [labels]
                seq = seq.to(device)
                labels = [l.to(device) for l in labels]

                _, out_heads = model(seq, return_timewise=spec["timewise_output"])
                out_heads = out_heads if isinstance(out_heads, list) else [out_heads]

                selected_outputs = [out_heads[head_idx(n)] for n in active_Ns]
                loss = compute_loss(selected_outputs, labels, spec["target_type"], criterion)
                task_loss_step.append(float(loss.item()))

                cb_reg = 0.0
                if cb_l2 > 0.0 and hasattr(model, "cb") and model.cb is not None:
                    cb_reg = sum(p.norm(2) ** 2 for p in model.cb.parameters())
                    loss = loss + cb_l2 * cb_reg
                elif rnn_eat and getattr(model, "_last_eat_loss", None) is not None:
                    eat_loss_val = float(model._last_eat_loss.item())
                    rnn_eat_lambda = get_eat_lambda(rnn_eat_lambda, current_N)
                    eat_loss_step.append(eat_loss_val)
                    if eat_loss_val > 5.0:
                        loss = loss
                    else:
                        loss = loss + rnn_eat_lambda * model._last_eat_loss
                if cb_l1 > 0.0 and hasattr(model, "cb") and model.cb is not None:
                    if hasattr(model.cb, '_last_gc') and model.cb._last_gc is not None:
                        loss = loss + cb_l1 * model.cb._last_gc.abs().mean()

                loss.backward()
                gr_pre, gc_pre, _ = get_grad_norms(model)

                nn.utils.clip_grad_norm_(model.parameters(),max_norm=7.5)
                gr, gc, _ = get_grad_norms(model)
                step_optimizer(optimizer, phase, shared_optimiser=shared_optimiser)
                losses.append(loss.item()); grs.append(gr); gcs.append(gc)
                grs_preclip.append(gr_pre); gcs_preclip.append(gc_pre)

            # Eval (generic task metric)
            model.eval()
            metrics = []
            with torch.no_grad():
                for _ in range(test_steps):
                    seq, labels = task_function(active_Ns, batch_size)
                    labels = labels if isinstance(labels, list) else [labels]
                    seq = seq.to(device)
                    labels = [l.to(device) for l in labels]

                    _, out_heads = model(seq, return_timewise=spec["timewise_output"])
                    out_heads = out_heads if isinstance(out_heads, list) else [out_heads]

                    selected_outputs = [out_heads[head_idx(n)] for n in active_Ns]
                    metric = spec["metric_fn"](selected_outputs, labels)
                    metrics.append(metric)

            acc = float(np.mean([m["score"] for m in metrics]))  # keep var name 'acc' for compatibility

            # Create row (using short keys is fine, mapping handles it)
            row = {
                "N": current_N, "phase": "init", "loss": float(np.mean(losses)),
                "acc": acc, "gRNN": float(np.mean(grs)), "gCB": float(np.mean(gcs)),
                "gRNN_pre": float(np.mean(grs_preclip)), "gCB_pre": float(np.mean(gcs_preclip)),
                "task_loss": float(np.mean(task_loss_step)),
                "eat_loss": float(np.mean(eat_loss_step)) if eat_loss_step else 0.0,
                "epoch": global_epoch
            }
            row['stage'] = stage_tag if stage_tag is not None else 'base'
            row['task'] = task_name
            
            safe_log_and_save(row, subdir, stats, save_every=10)
            
            print(f"init | Global Ep {global_epoch} | Acc: {acc:.2f}% | gRNN: {np.mean(grs):.4f} | gCB: {np.mean(gcs):.4f}", flush=True) 

            metric_for_curriculum = {"phase": "init", "score": acc,
                                  "loss": float(np.mean(losses))}
            if len(metrics) > 0 and "endpoint_error" in metrics[0]:
                metric_for_curriculum["endpoint_error"] = float(np.mean([m["endpoint_error"] for m in metrics]))

            if spec["advance_fn"](metric_for_curriculum):
                print(f"Base N={current_N} Solved! Moving to Alternating Phase.", flush=True)
                break

    # If init is skipped (non-resume), first alternating stage should target start N,
    # not start N + 1.
    first_stage_use_current_n = bool(skip_init_phase and subdir_override is None)

    # ---- Alternating loop using chosen variant ----
    while current_N < TARGET_END_N:
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Max global epochs reached. Stopping.", flush=True)
            np.save(os.path.join(subdir, "stats.npy"), stats)
            return stats

        if cb_schedule:
            next_n = current_N if first_stage_use_current_n else (current_N + 1)
            cb_decay = float(kwargs.get("cb_lr_decay", 0.05))
            cb_min_frac = float(kwargs.get("cb_lr_min_frac", 0.1))
            stage_cb_lr = _scheduled_cb_lr(cb_lr, next_n, Ns_init[0], decay=cb_decay, min_frac=cb_min_frac)
            _set_cb_lr(optimizer, stage_cb_lr)
            print(
                f"[cb_schedule] N={next_n} -> CB lr={stage_cb_lr:.6g}",
                flush=True,
            )
        
        def on_log(r):
            r = dict(r)  # ensure it's a regular dict for safe_log_and_save
            r['stage'] = stage_tag if stage_tag is not None else 'alternating'
            r['task'] = task_name
            safe_log_and_save(r, subdir, stats, save_every=10)

        # Run Variant
        variant_kwargs = dict(
            model=model,
            task_fn=task_function,
            active_Ns=active_Ns,
            current_N=current_N,
            Ns_init=Ns_init,
            batch_size=batch_size,
            training_steps_n=training_steps,
            test_steps=test_steps,
            device=device,
            global_epoch=global_epoch,
            criterion=criterion,
            head_idx=head_idx,
            optimizer=optimizer,
            get_grad_norms=get_grad_norms,
            set_active_module=set_active_module,
            readout_head_dyn=readout_head_dyn,
            n_heads=n_heads,
            n_forget=n_forget,
            THRESHOLD_CB_LEAD=threshold_cb_lead,
            THRESHOLD_RNN_LEAD=threshold_rnn_lead,
            THRESHOLD_FINAL=threshold_final,
            shared_optimiser=shared_optimiser,
            on_log=on_log,
            MAX_STAGE_EPOCHS=MAX_STAGE_EPOCHS,
            MAX_GLOBAL_EPOCHS=MAX_GLOBAL_EPOCHS,
            learning_alg=learning_alg,
            rnn_eat=rnn_eat,
            rnn_eat_lambda=rnn_eat_lambda,
            rnn_eat_loss_type=rnn_eat_loss_type,
            spec=spec,
            cb_l2=cb_l2,
            cb_l1=cb_l1,
            reservoir_interval=reservoir_interval_n,
        )
        if first_stage_use_current_n:
            variant_kwargs["next_N_override"] = int(current_N)

        ok, new_active_Ns, new_current_N, _, updated_epoch = variant_fn(**variant_kwargs)
        first_stage_use_current_n = False

        # Sync Global Epoch
        global_epoch = updated_epoch
        
        # Save final stats snapshot for safety
        np.save(os.path.join(subdir, "stats.npy"), stats)

        if not ok:
            print(f"Variant failed to solve N={new_current_N if 'new_current_N' in locals() else '?'}. Stopping.", flush=True)
            return stats

        # Success!
        active_Ns = new_active_Ns
        current_N = new_current_N

        if subdir_override is None:
            save_model(
                model, curriculum_type, n_heads, n_forget, task_name,
                run_number, current_N, current_N, args=args, base_path=base_path, affixes=affixes
            )
        else:
            ckpt_dir = os.path.join(subdir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir,
                f"{task_name}_{stage_tag if stage_tag else 'stage'}_ep{global_epoch}_N{current_N}.pt"
            )
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
    return stats