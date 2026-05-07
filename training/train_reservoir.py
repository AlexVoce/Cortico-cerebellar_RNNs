# training/train_reservoir.py

import os
import numpy as np
import torch
import torch.nn as nn

from save import save_model, find_next_free_network_number
from training.train_utils import get_grad_norms, set_active_module, step_optimizer
from training.variants import variant_cb_only_reservoir
from training.base import head_idx_factory
from tasks.task_registry import compute_loss


def safe_log_and_save(row, subdir, stats, save_every=10):
    """
    Map row keys to stats keys, update the in-memory stats object,
    and periodically save stats to disk.
    """
    mapping = {
        "N": "n_task",
        "acc": "accuracy",
        "gRNN": "grad_rnn",
        "gCB": "grad_cb",
        "gRNN_pre": "grad_rnn_pre",
        "gCB_pre": "grad_cb_pre",
        "task_loss": "task_loss",
        "loss": "loss",
        "phase": "phase",
        "epoch": "epoch",
        "task": "task",
        "stage": "stage",
    }

    for row_k, stats_k in mapping.items():
        if row_k in row and stats_k in stats:
            stats[stats_k].append(row[row_k])

    if row.get("epoch", 0) % save_every == 0:
        final_path = os.path.join(subdir, "stats.npy")
        tmp_path = os.path.join(subdir, "stats_tmp.npy")

        try:
            np.save(tmp_path, stats)
            os.replace(tmp_path, final_path)
        except Exception as e:
            print(f"Warning: Failed to save stats: {e}", flush=True)


def _zero_optimizer(optimizer, shared_optimiser=True):
    """
    Zero either a shared optimizer or a dict of optimizers.
    """
    if shared_optimiser:
        optimizer.zero_grad()
    else:
        for opt in optimizer.values():
            if opt is not None:
                opt.zero_grad()


def _build_optimizer(model, rnn_lr, cb_lr, shared_optimiser=True):
    """
    Split parameters into base RNN/readout parameters and CB parameters.
    """
    cb_params = list(model.cb.parameters()) if getattr(model, "cb", None) is not None else []
    cb_param_ids = {id(p) for p in cb_params}
    rnn_params = [p for p in model.parameters() if id(p) not in cb_param_ids]

    if shared_optimiser:
        param_groups = [
            {"params": rnn_params, "lr": rnn_lr, "name": "RNN"},
        ]

        if len(cb_params) > 0:
            param_groups.append(
                {"params": cb_params, "lr": cb_lr, "name": "CB"}
            )

        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.1,
            nesterov=True,
        )

    else:
        optimizer_rnn = torch.optim.SGD(
            rnn_params,
            lr=rnn_lr,
            momentum=0.1,
            nesterov=True,
        )

        optimizer_cb = (
            torch.optim.SGD(cb_params, lr=cb_lr, momentum=0.1, nesterov=True)
            if len(cb_params) > 0
            else None
        )

        optimizer = {"RNN": optimizer_rnn, "CB": optimizer_cb}

    return optimizer


def train_reservoir(
    model,
    curriculum_type,
    task_function,
    num_epochs,
    Ns_init,
    run_number,
    batch_size,
    training_steps,
    test_steps,
    device,
    base_path,
    affixes,
    n_heads=1,
    n_forget=1,
    task_name="dms",
    rnn_lr=0.05,
    cb_lr=0.05,
    readout_head_dyn="single",
    args=None,
    shared_optimiser=True,
    threshold_final=98.0,
    spec=None,
    target_end_n=150,
    subdir_override=None,
    stage_tag=None,
    skip_init_phase=False,
    reservoir_interval_n=10,
    **kwargs,
):
    """
    Reservoir-style training.

    Phase 0:
        Train the base RNN/readout at the initial N unless skipped.

    Reservoir phase:
        Use variant_cb_only_reservoir, which freezes/unfreezes modules according
        to the reservoir/interleaved-reservoir schedule.

    Notes
    -----
    This is a cleaned reservoir-only replacement for the older alternating
    variant framework.
    """
    if spec is None:
        raise ValueError("train_reservoir requires task spec (spec=...).")

    DEBUG = False
    MAX_STAGE_EPOCHS = 100 if DEBUG else num_epochs
    MAX_GLOBAL_EPOCHS = num_epochs
    TARGET_END_N = int(target_end_n) if target_end_n is not None else 150

    criterion = spec["criterion_ctor"]()

    if readout_head_dyn == "single":
        head_idx = lambda n: 0
    else:
        head_idx = head_idx_factory(Ns_init, num_heads=len(model.heads))

    # ---- init tracking ----
    current_N = int(Ns_init[0])
    active_Ns = sorted([current_N])
    global_epoch = 0

    stats = {
        "stage": [],
        "task": [],
        "n_task": [],
        "phase": [],
        "loss": [],
        "accuracy": [],
        "grad_rnn": [],
        "grad_cb": [],
        "epoch": [],
        "grad_rnn_pre": [],
        "grad_cb_pre": [],
        "task_loss": [],
    }

    # ---- saving ----
    if subdir_override is None:
        run_number = find_next_free_network_number(
            base_path=base_path,
            curriculum_type=curriculum_type,
            task=task_name,
            affixes=affixes,
            n_heads=n_heads,
            n_forget=n_forget,
        )

        subdir = save_model(
            model,
            curriculum_type,
            n_heads,
            n_forget,
            task_name,
            run_number,
            current_N,
            current_N,
            args=args,
            base_path=base_path,
            affixes=affixes,
        )

    else:
        subdir = subdir_override
        os.makedirs(subdir, exist_ok=True)

    # ---- resume stats if needed ----
    if getattr(args, "resume_ckpt", None) and subdir_override is not None:
        stats_path = os.path.join(subdir, "stats.npy")

        if os.path.exists(stats_path):
            try:
                old_stats = np.load(stats_path, allow_pickle=True).item()

                for k in stats.keys():
                    if k in old_stats and isinstance(old_stats[k], list):
                        stats[k] = list(old_stats[k])

                print(
                    f"[resume] Loaded existing reservoir stats from {stats_path} "
                    f"(rows={len(stats['n_task'])})",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"[resume] Warning: failed to load existing reservoir stats "
                    f"({e}); starting fresh stats in same folder.",
                    flush=True,
                )

    print(f"[train_reservoir] Saving results to: {subdir}", flush=True)

    # ---- optimizer groups ----
    reuse_ok = (subdir_override is not None) and hasattr(model, "_task_switch_optimizer")

    if reuse_ok:
        optimizer = model._task_switch_optimizer
        print("[train_reservoir] Reusing existing optimizer (task-switch mode)", flush=True)

    else:
        optimizer = _build_optimizer(
            model=model,
            rnn_lr=rnn_lr,
            cb_lr=cb_lr,
            shared_optimiser=shared_optimiser,
        )

    if subdir_override is not None and not reuse_ok:
        model._task_switch_optimizer = optimizer
        print("[train_reservoir] Created + stored optimizer (task-switch mode)", flush=True)

    # ---- Phase 0: train base RNN/head at initial N ----
    phase = "RNN_ONLY"
    set_active_module(model, phase)

    print("Trainable params after set_active_module:", flush=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("  ", name, flush=True)

    if subdir_override is not None:
        print(
            f"[resume] Phase 0 skipped; resuming reservoir stage from N={current_N}",
            flush=True,
        )
        goto_reservoir = True

    elif skip_init_phase:
        print(
            f"[skip_init_phase] Phase 0 skipped; starting reservoir stage from N={current_N}",
            flush=True,
        )
        goto_reservoir = True

    else:
        goto_reservoir = False

    if not goto_reservoir:
        print(f"=== Starting Phase 0 (base RNN) on N={current_N} ===", flush=True)

        for _ep in range(num_epochs):
            global_epoch += 1

            if global_epoch > MAX_GLOBAL_EPOCHS:
                np.save(os.path.join(subdir, "stats.npy"), stats)
                return stats

            model.train()

            losses = []
            grs = []
            gcs = []
            grs_preclip = []
            gcs_preclip = []
            task_loss_step = []

            for _ in range(training_steps):
                _zero_optimizer(optimizer, shared_optimiser=shared_optimiser)

                seq, labels = task_function(active_Ns, batch_size)
                labels = labels if isinstance(labels, list) else [labels]

                seq = seq.to(device)
                labels = [label.to(device) for label in labels]

                _, out_heads = model(
                    seq,
                    return_timewise=spec["timewise_output"],
                )
                out_heads = out_heads if isinstance(out_heads, list) else [out_heads]

                selected_outputs = [out_heads[head_idx(n)] for n in active_Ns]

                loss = compute_loss(
                    selected_outputs,
                    labels,
                    spec["target_type"],
                    criterion,
                )

                task_loss_step.append(float(loss.item()))
                loss.backward()

                gr_pre, gc_pre, _ = get_grad_norms(model)

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=7.5)

                gr, gc, _ = get_grad_norms(model)

                step_optimizer(
                    optimizer,
                    phase,
                    shared_optimiser=shared_optimiser,
                )

                losses.append(float(loss.item()))
                grs.append(gr)
                gcs.append(gc)
                grs_preclip.append(gr_pre)
                gcs_preclip.append(gc_pre)

            # ---- evaluation ----
            model.eval()
            metrics = []

            with torch.no_grad():
                for _ in range(test_steps):
                    seq, labels = task_function(active_Ns, batch_size)
                    labels = labels if isinstance(labels, list) else [labels]

                    seq = seq.to(device)
                    labels = [label.to(device) for label in labels]

                    _, out_heads = model(
                        seq,
                        return_timewise=spec["timewise_output"],
                    )
                    out_heads = out_heads if isinstance(out_heads, list) else [out_heads]

                    selected_outputs = [out_heads[head_idx(n)] for n in active_Ns]
                    metric = spec["metric_fn"](selected_outputs, labels)
                    metrics.append(metric)

            acc = float(np.mean([m["score"] for m in metrics]))

            row = {
                "N": current_N,
                "phase": "init",
                "loss": float(np.mean(losses)),
                "acc": acc,
                "gRNN": float(np.mean(grs)),
                "gCB": float(np.mean(gcs)),
                "gRNN_pre": float(np.mean(grs_preclip)),
                "gCB_pre": float(np.mean(gcs_preclip)),
                "task_loss": float(np.mean(task_loss_step)),
                "epoch": global_epoch,
                "stage": stage_tag if stage_tag is not None else "base",
                "task": task_name,
            }

            safe_log_and_save(row, subdir, stats, save_every=10)

            print(
                f"init | Global Ep {global_epoch} | "
                f"Acc: {acc:.2f}% | "
                f"gRNN: {np.mean(grs):.4f} | "
                f"gCB: {np.mean(gcs):.4f}",
                flush=True,
            )

            metric_for_curriculum = {
                "phase": "init",
                "score": acc,
                "loss": float(np.mean(losses)),
            }

            if len(metrics) > 0 and "endpoint_error" in metrics[0]:
                metric_for_curriculum["endpoint_error"] = float(
                    np.mean([m["endpoint_error"] for m in metrics])
                )

            if spec["advance_fn"](metric_for_curriculum):
                print(
                    f"Base N={current_N} solved. Moving to reservoir phase.",
                    flush=True,
                )
                break

    # If init is skipped outside resume mode, first reservoir stage should target
    # the current N rather than current_N + 1.
    first_stage_use_current_n = bool(skip_init_phase and subdir_override is None)

    # ---- Reservoir curriculum loop ----
    while current_N < TARGET_END_N:
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Max global epochs reached. Stopping.", flush=True)
            np.save(os.path.join(subdir, "stats.npy"), stats)
            return stats

        def on_log(row):
            row = dict(row)
            row["stage"] = stage_tag if stage_tag is not None else "reservoir"
            row["task"] = task_name
            safe_log_and_save(row, subdir, stats, save_every=10)

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
            shared_optimiser=shared_optimiser,
            on_log=on_log,
            MAX_STAGE_EPOCHS=MAX_STAGE_EPOCHS,
            MAX_GLOBAL_EPOCHS=MAX_GLOBAL_EPOCHS,
            spec=spec,
            reservoir_interval=reservoir_interval_n,
            learning_alg=getattr(args, "learning_alg", "bptt"),
        )

        if first_stage_use_current_n:
            variant_kwargs["next_N_override"] = int(current_N)

        ok, new_active_Ns, new_current_N, _, updated_epoch = variant_cb_only_reservoir(
            **variant_kwargs
        )

        first_stage_use_current_n = False
        global_epoch = updated_epoch

        np.save(os.path.join(subdir, "stats.npy"), stats)

        if not ok:
            print(
                f"Reservoir variant failed to solve "
                f"N={new_current_N if 'new_current_N' in locals() else '?'}. "
                f"Stopping.",
                flush=True,
            )
            return stats

        active_Ns = new_active_Ns
        current_N = new_current_N

        if subdir_override is None:
            save_model(
                model,
                curriculum_type,
                n_heads,
                n_forget,
                task_name,
                run_number,
                current_N,
                current_N,
                args=args,
                base_path=base_path,
                affixes=affixes,
            )
        else:
            ckpt_dir = os.path.join(subdir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            ckpt_path = os.path.join(
                ckpt_dir,
                f"{task_name}_{stage_tag if stage_tag else 'stage'}_"
                f"ep{global_epoch}_N{current_N}.pt",
            )

            torch.save({"state_dict": model.state_dict()}, ckpt_path)

    return stats


# Temporary backward-compatible alias.
# Remove this once all imports call train_reservoir directly.
def train_alternating(*args, **kwargs):
    return train_reservoir(*args, **kwargs)