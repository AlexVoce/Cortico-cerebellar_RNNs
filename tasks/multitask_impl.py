# tasks/multitask_impl.py

"""
Multi-task training: DMS + parity + oddball on shared binary sequences.

Key design:
  - One binary sequence per batch item.
  - ALL three heads produce a prediction and get a loss on EVERY sequence.
  - No task ID signal — the heads learn to extract different things from the
    same hidden state trajectory.
  - Evaluation uses each task's own batch_fn and metric_fn from TASK_SPECS,
    so advance_fn sees the exact same metric format as single-task training.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from tasks.tasks_using import get_match, get_parity, generate_binary_sequence


MULTITASK_TASKS = ["dms", "parity"]   # index == head index

# ---------------------------------------------------------------------------
# Per-sequence label computation
# ---------------------------------------------------------------------------

def get_labels_for_sequence(
    seq: torch.Tensor,
    Ns: List[int],
    task_names: List[str],
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Compute labels for all tasks and all Ns from a single sequence.
    
    Returns:
        {task_name: {N: label_for_N, ...}}
    """
    seq_1d = seq.squeeze(-1)   # [T, 1] -> [T]
    out = {t: {} for t in task_names}
    
    for tname in task_names:
        for N in Ns:
            if tname == "dms":
                lbl = get_match(seq_1d, N)
            elif tname == "parity":
                lbl = get_parity(seq_1d, N)
            else:
                raise ValueError(f"Unknown task in MULTITASK_TASKS: {tname}")
            out[tname][N] = lbl
    return out

# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def make_multitask_batch(
    Ns_dict: Dict[str, List[int]],     # per-task active Ns
    bs: int,
    task_names: List[str] = MULTITASK_TASKS,
) -> Tuple[torch.Tensor, Dict[str, List[List[torch.Tensor]]]]:
    """
    Generate a batch of shared binary sequences.
    Compute labels for every task and every active N from those sequences.

    Sequence length is determined by the largest active N across all tasks.

    Returns
    -------
    sequences : [T, B, 1]
    labels    : {task_name: [labels_for_N0, labels_for_N1, ...]}
                where labels_for_Ni is [B] (long tensor).
    """
    global_max_N = max(ns[-1] for ns in Ns_dict.values())

    M_min = global_max_N + 2
    M_max = M_min + 3 * global_max_N
    M = np.random.randint(M_min, M_max)

    with torch.no_grad():
        raw_seqs = [
            generate_binary_sequence(M, balanced=True).unsqueeze(-1)
            for _ in range(bs)
        ]  # list of [M, 1]

        # Build a deterministic N index once to avoid any label/task misalignment.
        all_ns = sorted({n for ns in Ns_dict.values() for n in ns})

        # Compute all labels efficiently in one pass per sequence.
        all_task_all_n_labels = [
            get_labels_for_sequence(seq, all_ns, task_names)
            for seq in raw_seqs
        ]  # list of dicts: {task: {N: label_N, ...}}

        # Reorganize: {task: {N_idx: [b0_label, b1_label, ...]}}
        labels: Dict[str, List[torch.Tensor]] = {t: [] for t in task_names}
        
        for tname in task_names:
            for N in Ns_dict[tname]:
                batch_labels = torch.stack([
                    all_task_all_n_labels[b][tname][N]
                    for b in range(bs)
                ])  # [B]
                labels[tname].append(batch_labels)

        sequences = torch.stack(raw_seqs).permute(1, 0, 2)  # [T, B, 1]

    return sequences, labels


# ---------------------------------------------------------------------------
# Loss — every head trains on every sequence
# ---------------------------------------------------------------------------

def multitask_loss(
    out_heads: List[torch.Tensor],          # [task] -> [B, C]
    labels: Dict[str, List[torch.Tensor]],  # task -> [label per N]
    Ns_dict: Dict[str, List[int]],
    task_names: List[str],
    device: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Cross-entropy for each head on every sequence in the batch.
    When a task has multiple active Ns the loss is averaged across them.
    """
    total = torch.tensor(0.0, device=device)
    per_task = {}

    for tid, tname in enumerate(task_names):
        logits   = out_heads[tid]               # [B, C]
        task_Ns  = Ns_dict[tname]
        task_loss = torch.tensor(0.0, device=device)

        for n_idx in range(len(task_Ns)):
            lbl = labels[tname][n_idx].to(device)   # [B]
            task_loss = task_loss + F.cross_entropy(logits, lbl)

        task_loss = task_loss / len(task_Ns)

        if torch.isfinite(task_loss):
            total = total + task_loss
        per_task[tname] = task_loss.detach().item()

    return total, per_task

def _set_multitask_trainable(
    model: nn.Module,
    reservoir_mode: str = "off",   # "off", "global_once", "all_tasks_once"
    train_rnn_core: bool = True,
    train_cb: bool = True,
):
    """
    Multitask parameter routing.

    Heads are always trainable in multitask.
    Non-head params are routed according to reservoir mode.
    """
    # Heads always train in multitask
    for p in model.heads.parameters():
        p.requires_grad = True

    for name, p in model.named_parameters():
        if "heads" in name:
            continue

        is_cb_param = name.startswith("cb.") or ".cb." in name
        is_rnn_param = name.startswith("hh.") or ".hh." in name

        if reservoir_mode == "off":
            p.requires_grad = True
        else:
            if is_cb_param:
                p.requires_grad = train_cb
            else:
                if is_rnn_param:
                    p.requires_grad = train_rnn_core
                else:
                    p.requires_grad = True  # non-CB, non-recurrent params always train


def _build_persistent_optimizer_multitask(optimizer_ctor, model: nn.Module):
    """Build one optimizer over all parameters so momentum/state persists across routing transitions."""
    all_params = list(model.parameters())
    if len(all_params) == 0:
        raise ValueError("Model has no parameters; cannot build multitask optimizer.")
    return optimizer_ctor(all_params)


def _save_advance_checkpoint_multitask(
    model: nn.Module,
    subdir: str,
    task_label: str,
    epoch_idx: int,
    solved_n: int,
):
    ckpt_dir = os.path.join(subdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{task_label}_multitask_ep{epoch_idx}_N{solved_n}.pt")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)


def _is_periodic_refresh_n(n_value: int, start_n: int, interval_n: int) -> bool:
    if interval_n <= 0:
        return False
    if n_value <= start_n:
        return False
    return (n_value - start_n) % interval_n == 0
# ---------------------------------------------------------------------------
# Evaluation — clean per-task batches using TASK_SPECS
# ---------------------------------------------------------------------------

def eval_all_tasks(
    model: nn.Module,
    task_specs: dict,
    task_names: List[str],
    Ns_dict: Dict[str, List[int]],
    batch_size: int,
    test_steps: int,
    device: str,
    use_shared_batches: bool = True, 
) -> Dict[str, dict]:
    """
    Evaluate each task using its own batch_fn and metric_fn from TASK_SPECS.
    
    Args:
        use_shared_batches: If True, evaluate on shared multitask sequences
                           (consistent with multitask training regime).
                           If False, use clean per-task batches.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        if use_shared_batches:
            # Evaluate on shared multitask sequences (consistent with training)
            for _ in range(test_steps):
                seqs, labels = make_multitask_batch(Ns_dict, batch_size, task_names)
                seqs = seqs.to(device)
                
                _, out_heads = model(seqs, return_timewise=False)
                
                # Compute per-task metrics on shared sequences
                for tid, tname in enumerate(task_names):
                    if tname not in results:
                        results[tname] = {"scores": [], "per_heads": []}
                    
                    spec = task_specs[tname]
                    # Use labels for hardest N (last index)
                    lbl = labels[tname][-1].to(device)  # [B]
                    
                    # Compute accuracy
                    preds = out_heads[tid].argmax(dim=1)
                    acc = (preds == lbl).float().mean().item() * 100
                    results[tname]["scores"].append(acc)
            
            # Aggregate results
            final_results = {}
            for tname in task_names:
                mean_score = float(np.mean(results[tname]["scores"]))
                final_results[tname] = {
                    "score": mean_score,
                    "per_head": [mean_score],  # Single head per task
                    "name": "accuracy",
                }
            return final_results
        else:
            # Original behavior: clean per-task batches (for debugging/comparison)
            for tid, tname in enumerate(task_names):
                spec = task_specs[tname]
                Ns   = Ns_dict[tname]
                metric_accum = []

                for _ in range(test_steps):
                    seqs, labs = spec["batch_fn"](Ns, batch_size)
                    seqs = seqs.to(device)
                    labs = [l.to(device) for l in labs]

                    timewise = spec.get("timewise_output", False)
                    _, out_heads = model(seqs, return_timewise=timewise)

                    # Only this task's head
                    metric = spec["metric_fn"]([out_heads[tid]] * len(labs), labs)
                    metric_accum.append(metric)

                mean_score = float(np.nanmean([m["score"] for m in metric_accum]))
                per_head   = np.nanmean(
                    np.array([m["per_head"] for m in metric_accum]), axis=0
                ).tolist()
                results[tname] = {
                    "score":    mean_score,
                    "per_head": per_head,
                    "name":     metric_accum[0].get("name", "score"),
                }

    return results
def _all_tasks_same_n(Ns: Dict[str, List[int]]) -> bool:
    vals = [Ns[t][-1] for t in Ns]
    return len(set(vals)) == 1


def _get_global_n(Ns: Dict[str, List[int]]) -> int:
    vals = [Ns[t][-1] for t in Ns]
    if len(set(vals)) != 1:
        raise ValueError(f"Expected synchronised Ns, got: {vals}")
    return vals[0]


def _set_all_tasks_to_n(Ns: Dict[str, List[int]], new_n: int) -> Dict[str, List[int]]:
    for t in Ns:
        Ns[t] = [new_n]
    return Ns


def _task_ready(
    tname: str,
    mean_metrics: Dict[str, dict],
    task_specs: dict,
    mt_advance_threshold: Optional[float],
) -> bool:
    if mt_advance_threshold is None:
        return task_specs[tname]["advance_fn"](mean_metrics[tname])
    return mean_metrics[tname]["score"] >= mt_advance_threshold


def _all_tasks_ready(
    task_names: List[str],
    mean_metrics: Dict[str, dict],
    task_specs: dict,
    mt_advance_threshold: Optional[float],
) -> bool:
    return all(
        _task_ready(t, mean_metrics, task_specs, mt_advance_threshold)
        for t in task_names
    )


def _update_reservoir_state_on_advance(
    reservoir_mode: str,
    old_n: int,
    tname: str,
    task_specs: dict,
    global_rnn_frozen_forever: bool,
    task_warmup_done: Dict[str, bool],
) -> tuple[bool, Dict[str, bool], List[str]]:
    """Update reservoir flags when a task successfully advances N."""
    messages: List[str] = []
    start_n_t = task_specs[tname].get("start_n", 2)

    if reservoir_mode == "global_once":
        if (not global_rnn_frozen_forever) and (old_n <= start_n_t):
            global_rnn_frozen_forever = True
            messages.append(
                f"[reservoir/global_once] Warmup complete at task '{tname}' N={old_n}. RNN core now frozen forever."
            )

    elif reservoir_mode == "all_tasks_once":
        if (not task_warmup_done[tname]) and (old_n <= start_n_t):
            task_warmup_done[tname] = True
            messages.append(
                f"[reservoir/all_tasks_once] Warmup complete for task '{tname}' at N={old_n}."
            )

    elif reservoir_mode == "periodic_refresh":
        if (not task_warmup_done[tname]) and (old_n <= start_n_t):
            task_warmup_done[tname] = True
            messages.append(
                f"[reservoir/periodic_refresh] Warmup complete for task '{tname}' at N={old_n}. CB-only until next refresh N."
            )

    return global_rnn_frozen_forever, task_warmup_done, messages
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def multitask_train(
    model: nn.Module,
    task_specs: dict,
    optimizer_ctor,
    num_epochs: int,
    batch_size: int,
    device: str,
    subdir: str,
    curriculum_type: str = "single",
    training_steps: int = 50,
    test_steps: int = 25,
    target_end_n: Optional[int] = 150,
    num_add: int = 1,
    rnn_eat: bool = False,
    rnn_eat_lambda: float = 0.1,
    advance_patience: int = 1,
    task_names: List[str] = None,
    mt_advance_threshold: float = None,
    curriculum_mode: str = "synchronised",   # NEW
    max_n_gap: int = 2,                     # NEW, for bounded_async
    reservoir_mode: str = "off",             # NEW, for parameter routing
    reservoir_interval_n: int = 10,
):
    """
    curriculum_mode options:
      - "independent"   : each task advances independently (current behavior)
      - "synchronised"  : one shared N for all tasks, advance only when all tasks ready
      - "bounded_async" : tasks advance independently but cannot exceed slowest task by > max_n_gap
      - "all_tasks_gate": tasks keep their own N, but no task advances unless all tasks ready
    """
    if task_names is None:
        task_names = MULTITASK_TASKS

    if num_epochs <= 0:
        raise ValueError("num_epochs must be > 0.")
    if training_steps <= 0:
        raise ValueError("training_steps must be > 0.")
    if test_steps <= 0:
        raise ValueError("test_steps must be > 0.")
    if advance_patience <= 0:
        raise ValueError("advance_patience must be > 0.")

    valid_modes = {"independent", "synchronised", "bounded_async", "all_tasks_gate"}
    if curriculum_mode not in valid_modes:
        raise ValueError(f"Unknown curriculum_mode={curriculum_mode}. Must be one of {valid_modes}")
    valid_reservoir_modes = {"off", "global_once", "all_tasks_once", "periodic_refresh"}
    if reservoir_mode not in valid_reservoir_modes:
        raise ValueError(
            f"Unknown reservoir_mode={reservoir_mode}. "
            f"Must be one of {valid_reservoir_modes}"
        )
    if reservoir_mode == "periodic_refresh" and reservoir_interval_n <= 0:
        raise ValueError("reservoir_interval_n must be > 0 when using periodic_refresh.")

    os.makedirs(subdir, exist_ok=True)

    # ---- INITIAL Ns ----
    if curriculum_mode == "synchronised":
        shared_start_n = max(task_specs[t].get("start_n", 2) for t in task_names)
        Ns: Dict[str, List[int]] = {t: [shared_start_n] for t in task_names}
    else:
        Ns: Dict[str, List[int]] = {
            t: [task_specs[t].get("start_n", 2)]
            for t in task_names
        }

    done = {t: False for t in task_names}

    global_rnn_frozen_forever = False
    task_warmup_done = {t: False for t in task_names}

    stats = {
        "epoch": [],
        "loss": [],
        "curriculum_mode": curriculum_mode,
        "reservoir_mode": reservoir_mode,
        "reservoir_interval_n": reservoir_interval_n,
        "max_n_gap": max_n_gap,
        "advance_count": {t: 0 for t in task_names},
        "task_warmup_done": {t: [] for t in task_names},
        "rnn_frozen": [],
        **{f"acc_{t}": [] for t in task_names},
        **{f"N_{t}": [] for t in task_names},
        **{f"loss_{t}": [] for t in task_names},
    }

    print(f"\n[multitask] Starting — tasks: {task_names}", flush=True)
    print(f"[multitask] curriculum_mode={curriculum_mode}", flush=True)
    print(f"[multitask] Initial Ns: { {t: Ns[t] for t in task_names} }", flush=True)

    optimizer = _build_persistent_optimizer_multitask(optimizer_ctor, model)
    prev_train_rnn_core = None
    prev_train_cb = None

    for epoch in range(num_epochs):
        if all(done.values()):
            print("[multitask] All tasks reached target N. Done.", flush=True)
            break
        # ---- reservoir routing ----
        if reservoir_mode == "off":
            train_rnn_core = True
            train_cb = True

        elif reservoir_mode == "global_once":
            if global_rnn_frozen_forever:
                train_rnn_core = False
                train_cb = True
            else:
                train_rnn_core = True
                train_cb = False

        elif reservoir_mode == "all_tasks_once":
            if all(task_warmup_done[t] for t in task_names):
                train_rnn_core = False
                train_cb = True
                global_rnn_frozen_forever = True
            else:
                train_rnn_core = True
                train_cb = False

        elif reservoir_mode == "periodic_refresh":
            if not all(task_warmup_done[t] for t in task_names):
                train_rnn_core = True
                train_cb = False
            else:
                refresh_now = any(
                    _is_periodic_refresh_n(
                        n_value=Ns[t][-1],
                        start_n=task_specs[t].get("start_n", 2),
                        interval_n=reservoir_interval_n,
                    )
                    for t in task_names
                    if not done[t]
                )
                train_rnn_core = refresh_now
                train_cb = True

        # Update routing only if it changed; keep optimizer state persistent.
        if (train_rnn_core != prev_train_rnn_core or
            train_cb != prev_train_cb):

            _set_multitask_trainable(
                model=model,
                reservoir_mode=reservoir_mode,
                train_rnn_core=train_rnn_core,
                train_cb=train_cb,
            )

            prev_train_rnn_core = train_rnn_core
            prev_train_cb = train_cb

            print(
                f"[multitask] routing update at epoch {epoch+1}: "
                f"train_rnn_core={train_rnn_core}, train_cb={train_cb}, "
                f"global_rnn_frozen_forever={global_rnn_frozen_forever}",
                flush=True,
            )

        # ---- TRAINING ----
        model.train()
        epoch_losses = []
        epoch_task_losses = {t: [] for t in task_names}

        for _ in range(training_steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            per_task = {}

            # Shared batch based on current Ns regime
            seqs, labels = make_multitask_batch(Ns, batch_size, task_names)
            seqs = seqs.to(device)

            _, out_heads = model(seqs, return_timewise=False)

            for tid, tname in enumerate(task_names):
                task_loss_accum = torch.tensor(0.0, device=device)
                n_active = len(Ns[tname])

                for n_idx in range(n_active):
                    lbl = labels[tname][n_idx].to(device)
                    task_loss = F.cross_entropy(out_heads[tid], lbl)
                    task_loss_accum = task_loss_accum + task_loss

                task_loss_accum = task_loss_accum / n_active

                if torch.isfinite(task_loss_accum):
                    total_loss = total_loss + task_loss_accum
                per_task[tname] = task_loss_accum.detach().item()

            if not torch.isfinite(total_loss):
                continue

            if rnn_eat and getattr(model, "_last_eat_loss", None) is not None:
                total_loss = total_loss + rnn_eat_lambda * model._last_eat_loss

            total_loss.backward()

            trainable_params = [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if len(trainable_params) > 0:
                nn.utils.clip_grad_norm_(trainable_params, max_norm=7.5)

            optimizer.step()

            epoch_losses.append(total_loss.item())
            for t, v in per_task.items():
                epoch_task_losses[t].append(v)

        # ---- EVALUATION ----
        mean_metrics = eval_all_tasks(
            model, task_specs, task_names, Ns,
            batch_size, test_steps, device,
            use_shared_batches=True,
        )

        # ---- LOGGING ----
        mean_loss = float(np.nanmean(epoch_losses)) if epoch_losses else float("nan")
        stats["epoch"].append(epoch)
        stats["loss"].append(mean_loss)
        stats["rnn_frozen"].append(bool(not train_rnn_core))
        for t in task_names:
            stats["task_warmup_done"][t].append(bool(task_warmup_done[t]))

        for t in task_names:
            stats[f"acc_{t}"].append(mean_metrics[t]["score"])
            stats[f"N_{t}"].append(Ns[t][-1])
            stats[f"loss_{t}"].append(
                float(np.nanmean(epoch_task_losses[t]))
                if epoch_task_losses[t] else float("nan")
            )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] loss={mean_loss:.4f} | "
            + " | ".join(
                f"{t}: acc={mean_metrics[t]['score']:.1f}% N={Ns[t][-1]}"
                for t in task_names
            ),
            flush=True,
        )

        np.save(os.path.join(subdir, "stats_multitask.npy"), stats)

        # ---- CURRICULUM LOGIC ----
        ready_map = {
            t: _task_ready(t, mean_metrics, task_specs, mt_advance_threshold)
            for t in task_names
        }
        all_ready = all(ready_map.values())

        if curriculum_mode == "synchronised":
            # One shared N for all tasks
            shared_n = _get_global_n(Ns)

            if target_end_n is not None and shared_n >= target_end_n:
                print(f"[multitask] synchronised curriculum reached target N={target_end_n}. Done.", flush=True)
                for t in task_names:
                    done[t] = True
                continue

            if all_ready:
                for t in task_names:
                    stats["advance_count"][t] += 1
                min_pat = min(stats["advance_count"][t] for t in task_names)

                if min_pat >= advance_patience:
                    _save_advance_checkpoint_multitask(
                        model=model,
                        subdir=subdir,
                        task_label="shared",
                        epoch_idx=epoch + 1,
                        solved_n=shared_n,
                    )

                    # Reservoir updates on successful escape from start-N
                    if reservoir_mode == "global_once":
                        if not global_rnn_frozen_forever:
                            global_rnn_frozen_forever = True
                            print(
                                f"[reservoir/global_once] Warmup complete at shared N={shared_n}. "
                                f"RNN core now frozen forever.",
                                flush=True,
                            )

                    elif reservoir_mode == "all_tasks_once":
                        for t in task_names:
                            if not task_warmup_done[t]:
                                # In synchronised mode, one successful shared advance
                                # means every task received its initial warmup.
                                if shared_n >= task_specs[t].get("start_n", 2):
                                    task_warmup_done[t] = True
                                    print(
                                        f"[reservoir/all_tasks_once] Warmup complete for task '{t}' at shared N={shared_n}.",
                                        flush=True,
                                    )

                    elif reservoir_mode == "periodic_refresh":
                        for t in task_names:
                            if not task_warmup_done[t] and shared_n >= task_specs[t].get("start_n", 2):
                                task_warmup_done[t] = True
                                print(
                                    f"[reservoir/periodic_refresh] Warmup complete for task '{t}' at shared N={shared_n}. "
                                    f"CB-only until next refresh N.",
                                    flush=True,
                                )

                    new_n = shared_n + max(1, int(num_add))
                    Ns = _set_all_tasks_to_n(Ns, new_n)
                    for t in task_names:
                        stats["advance_count"][t] = 0
                    print(
                        f"[multitask] synchronised ADVANCE N {shared_n} -> {new_n} "
                        f"(all tasks ready)",
                        flush=True,
                    )
            else:
                for t in task_names:
                    stats["advance_count"][t] = 0

        else:
            slowest_n = min(Ns[t][-1] for t in task_names)

            for tname in task_names:
                if done[tname]:
                    continue

                current_n = Ns[tname][-1]

                if target_end_n is not None and current_n >= target_end_n:
                    print(f"  [{tname}] reached target N={target_end_n}. Done.", flush=True)
                    done[tname] = True
                    continue

                # Gate condition depends on mode
                if curriculum_mode == "independent":
                    allowed_to_attempt_advance = True

                elif curriculum_mode == "all_tasks_gate":
                    allowed_to_attempt_advance = all_ready

                elif curriculum_mode == "bounded_async":
                    allowed_to_attempt_advance = (current_n <= slowest_n + max_n_gap)

                else:
                    raise ValueError(f"Unhandled curriculum_mode={curriculum_mode}")

                if not allowed_to_attempt_advance:
                    stats["advance_count"][tname] = 0
                    if curriculum_mode == "bounded_async" and current_n > slowest_n + max_n_gap:
                        print(
                            f"  [{tname}] blocked by max_n_gap "
                            f"(current={current_n}, slowest={slowest_n}, max_gap={max_n_gap})",
                            flush=True,
                        )
                    continue

                if ready_map[tname]:
                    stats["advance_count"][tname] += 1
                    if stats["advance_count"][tname] >= advance_patience:
                        old_n = current_n

                        _save_advance_checkpoint_multitask(
                            model=model,
                            subdir=subdir,
                            task_label=tname,
                            epoch_idx=epoch + 1,
                            solved_n=old_n,
                        )

                        # Reservoir updates must run in non-synchronised modes too.
                        global_rnn_frozen_forever, task_warmup_done, reservoir_msgs = _update_reservoir_state_on_advance(
                            reservoir_mode=reservoir_mode,
                            old_n=old_n,
                            tname=tname,
                            task_specs=task_specs,
                            global_rnn_frozen_forever=global_rnn_frozen_forever,
                            task_warmup_done=task_warmup_done,
                        )
                        for msg in reservoir_msgs:
                            print(msg, flush=True)

                        new_n = old_n + max(1, int(num_add))
                        Ns[tname] = [new_n]
                        stats["advance_count"][tname] = 0
                        print(
                            f"  [{tname}] ADVANCE N {old_n} -> {new_n} "
                            f"(score={mean_metrics[tname]['score']:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [{tname}] ready but waiting for patience "
                            f"({stats['advance_count'][tname]}/{advance_patience})",
                            flush=True,
                        )
                else:
                    stats["advance_count"][tname] = 0

    np.save(os.path.join(subdir, "stats_multitask.npy"), stats)
    return stats