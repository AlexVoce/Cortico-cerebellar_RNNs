"""
Continual learning experiments for CB-PFC multitask model.

Two experiments in one:

1. Forgetting: train tasks sequentially, evaluate all heads every epoch.
   Gives full forgetting curves for inactive tasks.

2. Savings: when a task appears more than once in the plan, measure
   epochs-to-criterion and fixed-budget accuracy on first vs subsequent
   exposures. CB vs no-CB comparison tests whether CB preserves a latent
   trace that enables faster reacquisition.

Architecture assumptions (same as multitask setup):
  - Shared RNN, one persistent head per unique task.
  - During each phase: only the active task's head + shared RNN are updated.
    All other heads are frozen.
  - Evaluation: all heads evaluated every epoch regardless of active task.

Usage
-----
from continual_impl import continual_train

plan = ["dms", "parity", "dms"]        # task repeated = savings experiment
continual_train(
    model      = rnn,
    task_specs = TASK_SPECS,
    optimizer_ctor = lambda params: torch.optim.SGD(params, lr=0.01,
                                                    momentum=0.1, nesterov=True),
    plan       = plan,
    epochs_per_phase = 100,
    ...
)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, cast


# ---------------------------------------------------------------------------
# Parameter routing — freeze/unfreeze per phase
# ---------------------------------------------------------------------------
def _set_active_task(
    model: nn.Module,
    active_task: str,
    task_names: List[str],
    reservoir_mode: str = "off",   # "off", "global_once", "per_task_once"
    train_rnn_core: bool = True,
    train_cb: bool = True,
):
    """
    Parameter routing.

    Heads:
        - only active task head is trainable

    Non-head params:
        reservoir_mode == "off":
            - all non-head params trainable
        otherwise:
            - RNN core trainable iff train_rnn_core
            - CB trainable iff train_cb
    """
    active_tid = task_names.index(active_task)
    heads = cast(nn.ModuleList, getattr(model, "heads"))

    # Only active head trains
    for tid, _ in enumerate(task_names):
        requires_grad = (tid == active_tid)
        for p in heads[tid].parameters():
            p.requires_grad = requires_grad

    # Route non-head params
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
                    p.requires_grad = True  # non-CB, non-RNN params always train

def _rebuild_optimizer(optimizer_ctor, model: nn.Module):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found when rebuilding optimizer.")
    return optimizer_ctor(trainable_params)


def _build_persistent_optimizer(optimizer_ctor, model: nn.Module):
    """Build one optimizer over all parameters so momentum/state persists across phases."""
    all_params = list(model.parameters())
    if len(all_params) == 0:
        raise ValueError("Model has no parameters; cannot build optimizer.")
    return optimizer_ctor(all_params)


def _save_advance_checkpoint(model: nn.Module, subdir: str, task_name: str, epoch_idx: int, solved_n: int):
    ckpt_dir = os.path.join(subdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{task_name}_continual_ep{epoch_idx}_N{solved_n}.pt")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)


def _is_periodic_refresh_n(n_value: int, start_n: int, interval_n: int) -> bool:
    if interval_n <= 0:
        return False
    if n_value <= start_n:
        return False
    return (n_value - start_n) % interval_n == 0

def add_task_cue(seqs, active_task, task_names, device=None):
    """
    seqs: [T, B, input_dim_base] usually [T, B, 1]
    active_task: str
    task_names: list[str] defining cue order

    Returns:
        seqs_with_cue: [T, B, base_dim + n_tasks]
    """
    if device is None:
        device = seqs.device

    T, B, D = seqs.shape
    n_tasks = len(task_names)

    cue = torch.zeros(T, B, n_tasks, device=device, dtype=seqs.dtype)
    tid = task_names.index(active_task)
    cue[:, :, tid] = 1.0

    return torch.cat([seqs, cue], dim=-1)

def add_switch_pulse(seqs, pulse_len=3, amplitude=1.0):
    """
    seqs: [T, B, D]
    returns: [T, B, D+1]
    """
    T, B, D = seqs.shape
    pulse = torch.zeros(T, B, 1, device=seqs.device, dtype=seqs.dtype)
    pulse[:min(pulse_len, T), :, 0] = amplitude
    return torch.cat([seqs, pulse], dim=-1)

def _mean_over_tb(x: torch.Tensor) -> torch.Tensor:
    """
    x: [T,B,D] or [B,D]
    returns mean over time/batch -> [D]
    """
    if x is None:
        return None
    if x.dim() == 3:
        return x.mean(dim=(0, 1))
    elif x.dim() == 2:
        return x.mean(dim=0)
    else:
        raise ValueError(f"Unexpected tensor shape for mean-over-tb: {x.shape}")


def _normalized_dot(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm() + eps)
    b = b / (b.norm() + eps)
    return torch.dot(a, b)
# ---------------------------------------------------------------------------
# Single-phase training
# ---------------------------------------------------------------------------
def _train_phase(
    model: nn.Module,
    task_name: str,
    task_names: List[str],
    task_specs: dict,
    optimizer: torch.optim.Optimizer,
    Ns: List[int],
    batch_size: int,
    training_steps: int,
    device: str,
    rnn_eat: bool = False,
    rnn_eat_lambda: float = 0.1,
    task_id_cue: bool = False,
) -> float:
    """Run one epoch of training on the active task. Returns mean loss."""
    model.train()
    tid = task_names.index(task_name)
    spec = task_specs[task_name]
    losses = []

    for _ in range(training_steps):
        seqs, labs = spec["batch_fn"](Ns, batch_size)
        seqs = seqs.to(device)
        seqs = add_task_cue(seqs, task_name, task_names, device=device) if task_id_cue else seqs
        seqs = add_switch_pulse(seqs)
        lbl = labs[-1].to(device)

        optimizer.zero_grad()
        _, out_heads = model(seqs, return_timewise=False)
        loss = F.cross_entropy(out_heads[tid], lbl)

        if not torch.isfinite(loss):
            continue

        eat_loss = getattr(model, "_last_eat_loss", None)
        if rnn_eat and isinstance(eat_loss, torch.Tensor):
            loss = loss + rnn_eat_lambda * eat_loss

        loss.backward()

        trainable_params = [
            p for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if len(trainable_params) > 0:
            nn.utils.clip_grad_norm_(trainable_params, max_norm=7.5)

        optimizer.step()
        losses.append(loss.item())

    return float(np.nanmean(losses)) if losses else float("nan")

# ---------------------------------------------------------------------------
# Evaluation — all heads, every epoch
# ---------------------------------------------------------------------------

def _eval_all(
    model: nn.Module,
    task_names: List[str],
    task_specs: dict,
    Ns_dict: Dict[str, List[int]],
    batch_size: int,
    test_steps: int,
    device: str,
    task_id_cue: bool = False,
) -> Dict[str, dict]:
    model.eval()
    results = {}
    with torch.no_grad():
        for tid, tname in enumerate(task_names):
            spec = task_specs[tname]
            Ns   = Ns_dict[tname]
            metric_accum = []
            for _ in range(test_steps):
                seqs, labs = spec["batch_fn"](Ns, batch_size)
                seqs = seqs.to(device)
                seqs = add_task_cue(seqs, tname, task_names, device=device) if task_id_cue else seqs
                seqs = add_switch_pulse(seqs) 
                labs = [l.to(device) for l in labs]
                timewise = spec.get("timewise_output", False)
                _, out_heads = model(seqs, return_timewise=timewise)
                metric = spec["metric_fn"](
                    [out_heads[tid]] * len(labs), labs
                )
                metric_accum.append(metric)
            mean_score = float(np.nanmean([m["score"] for m in metric_accum]))
            per_head   = np.nanmean(
                np.array([m["per_head"] for m in metric_accum]), axis=0
            ).tolist()
            results[tname] = {"score": mean_score, "per_head": per_head}
    return results

# ---------------------------------------------------------------------------
# Savings analysis
# ---------------------------------------------------------------------------

def compute_savings(stats: dict, plan: List[str], threshold: float = 85.0) -> dict:
    """
    For each task that appears more than once in the plan, compute:
      - epochs_to_criterion: epochs needed in phase 1 vs phase 2+
      - savings_ratio: (phase1_epochs - phase2_epochs) / phase1_epochs
      - fixed_budget_acc: accuracy after N epochs in phase 1 vs phase 2

    Parameters
    ----------
    stats    : dict returned by continual_train
    plan     : task sequence used during training
    threshold: accuracy threshold defining "criterion"
    """
    from collections import defaultdict

    # Find which tasks appear more than once
    task_counts = defaultdict(int)
    for t in plan:
        task_counts[t] += 1
    repeated = [t for t, c in task_counts.items() if c > 1]

    savings = {}
    for task in repeated:
        # Find phase indices for this task
        phase_indices = [i for i, t in enumerate(plan) if t == task]

        epochs_to_criterion = []
        for phase_idx in phase_indices:
            key = f"acc_{task}_phase{phase_idx}"
            accs = np.array(stats[key])
            # Find first epoch where acc >= threshold
            criterion_epoch = None
            for i, acc in enumerate(accs):
                if acc >= threshold:
                    criterion_epoch = i + 1
                    break
            epochs_to_criterion.append(criterion_epoch)  # None = never solved

        # Fixed budget: compare accuracy at epoch min(phase lengths)
        phase_lengths = [len(stats[f"acc_{task}_phase{i}"]) for i in phase_indices]
        budget = min(phase_lengths)
        fixed_budget_accs = [
            float(np.mean(stats[f"acc_{task}_phase{i}"][:budget]))
            for i in phase_indices
        ]

        savings[task] = {
            "phase_indices":        phase_indices,
            "epochs_to_criterion":  epochs_to_criterion,
            "fixed_budget_accs":    fixed_budget_accs,
            "budget_epochs":        budget,
            "savings_ratio": (
                (epochs_to_criterion[0] - epochs_to_criterion[1])
                / epochs_to_criterion[0]
                if (epochs_to_criterion[0] is not None
                    and epochs_to_criterion[1] is not None)
                else float("nan")
            ),
        }

    return savings


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
maximum_N = 150  # hard cap on curriculum N; training can continue at this N

def continual_train(
    model: nn.Module,
    task_specs: dict,
    optimizer_ctor,                     # callable(params) -> optimizer
    plan: List[str],                    # e.g. ["dms", "parity", "dms"]
    epochs_per_phase: int,
    max_global_epochs: int,
    batch_size: int,
    training_steps: int,
    test_steps: int,
    device: str,
    subdir: str,
    # curriculum
    target_end_n: Optional[int] = 150,
    num_add: int = 1,
    advance_patience: int = 3,
    task_id_cue: bool = False,
    # CB eat loss
    rnn_eat: bool = False,
    rnn_eat_lambda: float = 0.1,
    # savings threshold
    savings_threshold: float = 85.0,
    ct_advance_threshold: Optional[float] = 85.0,
    restart_n_offset: int = -1,
    # reservoir mode
    reservoir_mode: str = "off",  # "off", "global_once", "per_task_once", "periodic_refresh"
    reservoir_interval_n: int = 10,
) -> dict:
    """
    Sequential continual learning with per-phase parameter freezing.

    reservoir_mode:
        "off":
            Standard training. RNN + CB always train (plus active head).

        "global_once":
            One global initial warmup: train RNN core + active head only.
            After the first successful advance out of initial N on any task,
            freeze RNN forever; thereafter train CB + active head only.

        "per_task_once":
            Each task gets one initial warmup at its start_n:
            train RNN core + active head only for that task.
            After that task first advances out of start_n, that task is marked
            warmup-complete. Thereafter whenever that task appears, train
            CB + active head only for that task.

        "periodic_refresh":
            Each task gets one initial warmup at its start_n (RNN core only,
            CB off). After warmup, train CB + active head only except every
            `reservoir_interval_n` Ns from that task's start_n, where BOTH
            RNN core and CB train for that N.
    """
    if reservoir_mode not in {"off", "global_once", "per_task_once", "periodic_refresh"}:
        raise ValueError(
            "reservoir_mode must be one of 'off', 'global_once', 'per_task_once', "
            f"'periodic_refresh', got {reservoir_mode}"
        )
    if reservoir_mode == "periodic_refresh" and reservoir_interval_n <= 0:
        raise ValueError("reservoir_interval_n must be > 0 when using periodic_refresh.")

    if len(plan) == 0:
        raise ValueError("plan must contain at least one task.")
    if epochs_per_phase <= 0:
        raise ValueError("epochs_per_phase must be > 0 to avoid an infinite phase loop.")
    if max_global_epochs <= 0:
        raise ValueError("max_global_epochs must be > 0.")
    if training_steps <= 0:
        raise ValueError("training_steps must be > 0.")
    if test_steps <= 0:
        raise ValueError("test_steps must be > 0.")
    if advance_patience <= 0:
        raise ValueError("advance_patience must be > 0.")

    os.makedirs(subdir, exist_ok=True)

    # Unique tasks (preserving first-occurrence order)
    seen = []
    for t in plan:
        if t not in seen:
            seen.append(t)
    task_names = seen

    # Validate
    for t in plan:
        if t not in task_specs:
            raise ValueError(f"Task '{t}' not in TASK_SPECS")

    if not hasattr(model, "heads"):
        raise ValueError("Model must expose a 'heads' attribute for per-task routing.")
    heads = cast(nn.ModuleList, getattr(model, "heads"))
    if len(heads) != len(task_names):
        raise ValueError(
            f"Model has {len(heads)} heads, but plan has {len(task_names)} unique tasks: {task_names}. "
            "Head count must match unique task count and ordering."
        )

    # Per-task Ns
    Ns: Dict[str, List[int]] = {
        t: [task_specs[t].get("start_n", 2)]
        for t in task_names
    }

    # Restart point for each task
    solved_N: Dict[str, int] = {
        t: task_specs[t].get("start_n", 2)
        for t in task_names
    }

    # Per-task advance streak
    streaks = {t: 0 for t in task_names}

    # Completion relative to target_end_n
    done = {t: False for t in task_names}

    # Reservoir state
    global_rnn_frozen_forever = False
    task_warmup_done = {t: False for t in task_names}

    stats: dict = {
        "plan": plan,
        "plan_executed": [],
        "task_names": task_names,
        "phase_boundaries": [],
        "reservoir_mode": reservoir_mode,
    }

    print(f"\n[continual] plan: {plan}", flush=True)
    print(f"[continual] unique tasks / head order: {task_names}", flush=True)
    print(f"[continual] epochs per phase: {epochs_per_phase}", flush=True)
    print(f"[continual] max global epochs: {max_global_epochs}", flush=True)
    print(f"[continual] reservoir_mode: {reservoir_mode}", flush=True)
    if reservoir_mode == "periodic_refresh":
        print(f"[continual] reservoir_interval_n: {reservoir_interval_n}", flush=True)

    # Keep one optimizer for the entire run to preserve momentum/state.
    optimizer = _build_persistent_optimizer(optimizer_ctor, model)

    global_epoch = 0
    phase_idx = 0

    while global_epoch < max_global_epochs:
        if target_end_n is not None and all(done.values()):
            print("[continual] All tasks reached target N. Stopping.", flush=True)
            break

        active_task = plan[phase_idx % len(plan)]
        start_n_for_task = task_specs[active_task].get("start_n", 2)

        stats["plan_executed"].append(active_task)
        stats["phase_boundaries"].append(global_epoch)

        print(f"\n[continual] === Phase {phase_idx}: {active_task} ===", flush=True)

        # Per-phase logs
        for tname in task_names:
            stats[f"acc_{tname}_phase{phase_idx}"] = []
            stats[f"N_{tname}_phase{phase_idx}"] = []
        stats[f"loss_phase{phase_idx}"] = []

        # Restart active task at saved restart point
        Ns[active_task] = [solved_N[active_task]]
        phase_Ns = Ns[active_task]
        phase_start_n = phase_Ns[-1]

        # Decide what trains this phase
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

        elif reservoir_mode == "per_task_once":
            if task_warmup_done[active_task]:
                train_rnn_core = False
                train_cb = True
            else:
                train_rnn_core = True
                train_cb = False

        elif reservoir_mode == "periodic_refresh":
            if not task_warmup_done[active_task]:
                train_rnn_core = True
                train_cb = False
            else:
                refresh_now = _is_periodic_refresh_n(
                    n_value=phase_start_n,
                    start_n=start_n_for_task,
                    interval_n=reservoir_interval_n,
                )
                train_rnn_core = refresh_now
                train_cb = True
        else:
            train_rnn_core = True
            train_cb = True

        _set_active_task(
            model=model,
            active_task=active_task,
            task_names=task_names,
            reservoir_mode=reservoir_mode,
            train_rnn_core=train_rnn_core,
            train_cb=train_cb,
        )

        optimizer.zero_grad(set_to_none=True)

        print(
            f"  [{active_task}] Restarting from N={phase_start_n} | "
            f"train_rnn_core={train_rnn_core} | train_cb={train_cb} | "
            f"global_rnn_frozen_forever={global_rnn_frozen_forever} | "
            f"task_warmup_done={task_warmup_done[active_task]}",
            flush=True
        )

        for epoch in range(epochs_per_phase):
            if global_epoch >= max_global_epochs:
                break

            current_global_epoch = global_epoch + 1

            # ---- training ----
            mean_loss = _train_phase(
                model=model,
                task_name=active_task,
                task_names=task_names,
                task_specs=task_specs,
                optimizer=optimizer,
                Ns=phase_Ns,
                batch_size=batch_size,
                training_steps=training_steps,
                device=device,
                rnn_eat=rnn_eat,
                rnn_eat_lambda=rnn_eat_lambda,
                task_id_cue=task_id_cue,
            )
            stats[f"loss_phase{phase_idx}"].append(mean_loss)

            # ---- eval all heads ----
            metrics = _eval_all(
                model=model,
                task_names=task_names,
                task_specs=task_specs,
                Ns_dict=Ns,
                batch_size=batch_size,
                test_steps=test_steps,
                device=device,
                task_id_cue=task_id_cue,
            )

            for tname in task_names:
                stats[f"acc_{tname}_phase{phase_idx}"].append(metrics[tname]["score"])
                stats[f"N_{tname}_phase{phase_idx}"].append(Ns[tname][-1])

            print(
                f"  [global {current_global_epoch}/{max_global_epochs} | "
                f"phase {phase_idx} | {active_task} | epoch {epoch+1}/{epochs_per_phase}] "
                f"loss={mean_loss:.3f} | "
                + " | ".join(
                    f"{t}: {metrics[t]['score']:.1f}% N={Ns[t][-1]}"
                    for t in task_names
                ),
                flush=True,
            )

            # ---- curriculum advance for active task only ----
            if ct_advance_threshold is None:
                should_advance = task_specs[active_task]["advance_fn"](metrics[active_task])
            else:
                should_advance = metrics[active_task]["score"] >= ct_advance_threshold

            if should_advance:
                streaks[active_task] += 1
            else:
                streaks[active_task] = 0

            if streaks[active_task] >= advance_patience:
                streaks[active_task] = 0
                old = Ns[active_task][-1]
                reservoir_routing_changed = False

                _save_advance_checkpoint(
                    model=model,
                    subdir=subdir,
                    task_name=active_task,
                    epoch_idx=current_global_epoch,
                    solved_n=old,
                )

                # Reservoir state update happens exactly once when advancing
                # out of the initial N for the relevant regime.
                if reservoir_mode == "global_once":
                    if (not global_rnn_frozen_forever) and (old <= start_n_for_task):
                        global_rnn_frozen_forever = True
                        train_rnn_core = False
                        train_cb = True
                        reservoir_routing_changed = True
                        print(
                            f"  [reservoir/global_once] Warmup complete at N={old}. "
                            f"RNN core now frozen forever.",
                            flush=True
                        )

                elif reservoir_mode == "per_task_once":
                    if (not task_warmup_done[active_task]) and (old <= start_n_for_task):
                        task_warmup_done[active_task] = True
                        train_rnn_core = False
                        train_cb = True
                        reservoir_routing_changed = True
                        print(
                            f"  [reservoir/per_task_once] Warmup complete for task '{active_task}' at N={old}. "
                            f"RNN core will stay frozen for this task from now on.",
                            flush=True
                        )

                elif reservoir_mode == "periodic_refresh":
                    if (not task_warmup_done[active_task]) and (old <= start_n_for_task):
                        task_warmup_done[active_task] = True
                        train_rnn_core = False
                        train_cb = True
                        reservoir_routing_changed = True
                        print(
                            f"  [reservoir/periodic_refresh] Warmup complete for task '{active_task}' at N={old}. "
                            f"CB-only until next refresh N.",
                            flush=True,
                        )

                if reservoir_routing_changed:
                    _set_active_task(
                        model=model,
                        active_task=active_task,
                        task_names=task_names,
                        reservoir_mode=reservoir_mode,
                        train_rnn_core=train_rnn_core,
                        train_cb=train_cb,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        f"  [{active_task}] Immediate routing update | "
                        f"train_rnn_core={train_rnn_core} | train_cb={train_cb}",
                        flush=True,
                    )

                # Save restart point
                solved_N[active_task] = max(
                    task_specs[active_task].get("start_n", 2),
                    old + restart_n_offset
                )

                # Advance N
                if target_end_n is None or Ns[active_task][-1] < target_end_n:
                    proposed_next_n = old + max(1, int(num_add))
                    next_n = min(proposed_next_n, maximum_N)
                    if next_n > old:
                        Ns[active_task] = [next_n]
                        phase_Ns = Ns[active_task]
                        print(f"  [{active_task}] ADVANCE N {old} -> {next_n}", flush=True)
                    else:
                        print(
                            f"  [{active_task}] At maximum_N={maximum_N}; keeping N={old}.",
                            flush=True,
                        )

                if target_end_n is not None and Ns[active_task][-1] >= target_end_n:
                    done[active_task] = True

            global_epoch += 1

        np.save(os.path.join(subdir, "stats_continual.npy"), np.array(stats, dtype=object))
        phase_idx += 1

    # ---- savings analysis ----
    savings = compute_savings(stats, stats["plan_executed"], threshold=savings_threshold)
    stats["savings"] = savings

    if savings:
        print(f"\n[continual] === Savings analysis (threshold={savings_threshold:.1f}%) ===", flush=True)
        for task, s in savings.items():
            print(f"  {task}:", flush=True)
            for i, (etc, fba) in enumerate(zip(s["epochs_to_criterion"], s["fixed_budget_accs"])):
                print(
                    f"    phase {s['phase_indices'][i]}: "
                    f"epochs-to-criterion={etc} | "
                    f"mean acc (first {s['budget_epochs']} epochs)={fba:.1f}%",
                    flush=True,
                )
            print(f"    savings ratio: {s['savings_ratio']:.2f}", flush=True)

    np.save(os.path.join(subdir, "stats_continual.npy"), np.array(stats, dtype=object))
    print("\n[continual] Done.", flush=True)
    return stats