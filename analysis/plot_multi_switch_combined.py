"""
plot_multitask_switching_combined.py

Self-contained plotting utilities for combining:
1. Multitask N-progression from stats_multitask.npy
2. Continual / task-switching N-progression from stats_continual.npy

Designed for your CB-RNN vs RNN-only curriculum-learning experiments.

Main public function:
    plot_multitask_and_switching_combined(...)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =============================================================================
# Style
# =============================================================================

BLUE = "cornflowerblue"
PINK = "salmon"
ORANGE = "orange"
GREEN = "green"
GREY = "gray"

DEFAULT_COLORS = [BLUE, PINK, ORANGE, GREEN, GREY]


# =============================================================================
# Generic utilities
# =============================================================================

def _load(path: Union[str, Path]) -> dict:
    """
    Load a .npy stats file saved as a Python dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist:\n{path}")

    out = np.load(path, allow_pickle=True)

    if isinstance(out, np.ndarray) and out.shape == ():
        return out.item()

    if isinstance(out, dict):
        return out

    try:
        return out.item()
    except Exception as exc:
        raise ValueError(f"Could not load {path} as a stats dictionary.") from exc


def _as_run_dir(path_or_file: Union[str, Path]) -> Path:
    """
    Convert either a stats file path or a run directory path into a run directory.
    """
    p = Path(path_or_file)

    if p.is_file():
        return p.parent

    return p


def _expand_repeat_paths(
    path_or_file: Union[str, Path, Sequence[Union[str, Path]]],
    expected_name: str,
) -> List[Path]:
    """
    Expand a file, directory, or list of files/directories into a list of stats files.

    This is the key function that prevents the old bug:
    multitask calls must pass expected_name="stats_multitask.npy";
    switching calls must pass expected_name="stats_continual.npy".
    """
    if path_or_file is None:
        return []

    if isinstance(path_or_file, (list, tuple)):
        all_paths = []
        for p in path_or_file:
            all_paths.extend(_expand_repeat_paths(p, expected_name=expected_name))
        return sorted(all_paths)

    p = Path(path_or_file)

    if p.is_file():
        if p.name != expected_name:
            raise ValueError(
                f"Expected {expected_name}, got {p.name}\n"
                f"Full path:\n{p}"
            )
        return [p]

    if p.is_dir():
        direct = p / expected_name
        if direct.exists():
            return [direct]

        found = sorted(p.rglob(expected_name))
        if len(found) > 0:
            return found

        raise FileNotFoundError(
            f"Could not find {expected_name} inside directory:\n{p}"
        )

    raise FileNotFoundError(f"Path does not exist:\n{p}")


def _safe_array(x) -> np.ndarray:
    """
    Convert stats entries into 1D float arrays.
    """
    arr = np.asarray(x)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    arr = arr.astype(float)
    return arr


def _nanpad_to_length(arr: np.ndarray, T: int) -> np.ndarray:
    """
    Pad a 1D array to length T with NaNs.
    """
    arr = _safe_array(arr)

    if len(arr) >= T:
        return arr[:T]

    out = np.full(T, np.nan, dtype=float)
    out[:len(arr)] = arr
    return out


def _sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    NaN-safe SEM.
    """
    n = np.sum(np.isfinite(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sem = sd / np.sqrt(n)

    sem = np.where(n <= 1, 0.0, sem)
    return sem


def _display_task_name(task: str) -> str:
    t = task.lower()

    if t == "dms":
        return "DMS"
    if t == "parity":
        return "Parity"
    if t == "oddball":
        return "Oddball"

    return task.capitalize()


# =============================================================================
# Multitask aggregation
# =============================================================================

def _discover_multitask_tasks(stats: dict) -> List[str]:
    """
    Discover task names from keys such as:
        acc_dms, acc_parity
        N_dms, N_parity
    """
    tasks = set()

    for k in stats.keys():
        if k.startswith("acc_"):
            tasks.add(k.split("_", 1)[1])
        elif k.startswith("N_"):
            tasks.add(k.split("_", 1)[1])

    return sorted(tasks)


def _aggregate_condition_multitask(
    path_or_file: Union[str, Path, Sequence[Union[str, Path]]],
    task_keys: Sequence[str],
    filename: str = "stats_multitask.npy",
) -> dict:
    """
    Aggregate one multitask condition across one or more runs.
    """
    paths = _expand_repeat_paths(path_or_file, expected_name=filename)

    if len(paths) == 0:
        raise ValueError("No multitask paths found.")

    loaded = [_load(p) for p in paths]

    # Determine maximum length across all requested keys and runs.
    max_T = 0
    for stats in loaded:
        for key in task_keys:
            if key in stats:
                max_T = max(max_T, len(_safe_array(stats[key])))

    if max_T == 0:
        raise ValueError(
            f"None of the requested keys were found in {filename}.\n"
            f"Requested keys: {task_keys}"
        )

    epoch = np.arange(1, max_T + 1)

    out = {
        "paths": paths,
        "n_runs": len(paths),
        "T": max_T,
        "epoch": epoch,
    }

    for key in task_keys:
        arrs = []

        for stats in loaded:
            if key in stats:
                arr = _nanpad_to_length(stats[key], max_T)
            else:
                arr = np.full(max_T, np.nan, dtype=float)

            arrs.append(arr)

        mat = np.vstack(arrs)

        out[key] = {
            "runs": mat,
            "mean": np.nanmean(mat, axis=0),
            "sem": _sem(mat, axis=0),
        }

    return out


# =============================================================================
# Continual / switching aggregation
# =============================================================================

def _discover_continual_tasks(global_stats: dict) -> List[str]:
    tasks = sorted(
        k.split("_", 1)[1]
        for k in global_stats.keys()
        if k.startswith("N_") and "_phase" not in k
    )
    return tasks

def _build_continual_global_dict(stats: dict) -> dict:
    """
    Build global task-level N trajectories from stats_continual.npy.

    Handles both formats:

    Format 1:
        N_dms, N_parity

    Format 2:
        N_dms_phase0, N_dms_phase1, ...
        N_parity_phase0, N_parity_phase1, ...

    The second format is merged into:
        N_dms
        N_parity
    by concatenating phase arrays in phase order.
    """
    out = {}

    n_keys = [k for k in stats.keys() if k.startswith("N_")]

    if len(n_keys) == 0:
        raise ValueError(
            "Could not find any N_* keys in continual stats file."
        )

    parsed = [_parse_continual_N_key(k) for k in n_keys]
    has_phase_keys = any(phase is not None for _, phase in parsed)

    if has_phase_keys:
        tasks = sorted(set(task for task, phase in parsed if task is not None))
        phases = sorted(set(phase for task, phase in parsed if phase is not None))

        # Work out phase lengths from available N_*_phase arrays.
        phase_lengths = {}
        for phase in phases:
            lengths = []
            for task in tasks:
                key = f"N_{task}_phase{phase}"
                if key in stats:
                    lengths.append(len(_safe_array(stats[key])))

            if len(lengths) == 0:
                continue

            phase_lengths[phase] = max(lengths)

        total_T = sum(phase_lengths[p] for p in phases)

        # Merge each task into one global N_task trajectory.
        for task in tasks:
            pieces = []

            last_value = np.nan

            for phase in phases:
                L = phase_lengths[phase]
                key = f"N_{task}_phase{phase}"

                if key in stats:
                    arr = _safe_array(stats[key])

                    if len(arr) < L:
                        arr = _nanpad_to_length(arr, L)

                    if np.any(np.isfinite(arr)):
                        finite = arr[np.isfinite(arr)]
                        if len(finite) > 0:
                            last_value = finite[-1]

                else:
                    # If this task was inactive in this phase, hold previous N.
                    if np.isfinite(last_value):
                        arr = np.full(L, last_value, dtype=float)
                    else:
                        arr = np.full(L, np.nan, dtype=float)

                pieces.append(arr)

            out[f"N_{task}"] = np.concatenate(pieces)

        # Build active task vector from the phase key names.
        active_task = []

        for phase in phases:
            L = phase_lengths[phase]

            active_candidates = []
            for task in tasks:
                key = f"N_{task}_phase{phase}"
                if key in stats:
                    arr = _safe_array(stats[key])
                    if len(arr) > 0 and np.any(np.isfinite(arr)):
                        active_candidates.append(task)

            # Prefer the task whose N changes in this phase.
            chosen = None
            for task in active_candidates:
                key = f"N_{task}_phase{phase}"
                arr = _safe_array(stats[key])
                if len(arr) > 1 and np.nanmax(arr) > np.nanmin(arr):
                    chosen = task
                    break

            if chosen is None:
                chosen = active_candidates[0] if active_candidates else tasks[0]

            active_task.extend([chosen] * L)

        out["task"] = np.asarray(active_task, dtype=object)
        out["epoch"] = np.arange(1, total_T + 1)
        return out

    # Non-phase format: already global.
    for k, v in stats.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v)
                if arr.ndim >= 1:
                    out[k] = arr
            except Exception:
                pass

    task_keys = [k for k in out.keys() if k.startswith("N_")]
    max_T = max(len(np.asarray(out[k])) for k in task_keys)

    if "epoch" not in out:
        out["epoch"] = np.arange(1, max_T + 1)

    for k in task_keys:
        out[k] = _nanpad_to_length(out[k], max_T)

    for k in ["task", "phase", "stage"]:
        if k in out:
            arr = np.asarray(out[k])
            if len(arr) < max_T:
                padded = np.empty(max_T, dtype=object)
                padded[:] = None
                padded[:len(arr)] = arr
                out[k] = padded
            else:
                out[k] = arr[:max_T]

    return out


def _parse_continual_N_key(key: str):
    """
    Parse keys like:
        N_dms
        N_parity
        N_dms_phase0
        N_parity_phase3

    Returns
    -------
    task, phase
        task: e.g. "dms"
        phase: int or None
    """
    if not key.startswith("N_"):
        return None, None

    body = key[2:]

    if "_phase" in body:
        task, phase_str = body.rsplit("_phase", 1)
        try:
            phase = int(phase_str)
        except ValueError:
            phase = None
        return task, phase

    return body, None

def _infer_active_task_from_global(global_stats: dict, task_names: Sequence[str]) -> np.ndarray:
    """
    Infer active task per epoch.

    Priority:
    1. Use stats["task"] if present.
    2. Otherwise infer from changes in N_task.
    """
    T = len(global_stats["epoch"])

    if "task" in global_stats:
        task_arr = np.asarray(global_stats["task"]).astype(str)
        if len(task_arr) >= T:
            return task_arr[:T]

    # Fallback: infer task from which N changes most recently.
    active = np.empty(T, dtype=object)
    active[:] = task_names[0]

    prev_vals = {t: global_stats[f"N_{t}"][0] for t in task_names}

    for i in range(1, T):
        changed = []
        for t in task_names:
            val = global_stats[f"N_{t}"][i]
            prev = prev_vals[t]

            if np.isfinite(val) and np.isfinite(prev) and val != prev:
                changed.append(t)

            prev_vals[t] = val

        if len(changed) > 0:
            active[i] = changed[0]
        else:
            active[i] = active[i - 1]

    return active


def _phase_boundaries_from_task(active_task: np.ndarray) -> Tuple[List[int], List[int], List[str]]:
    """
    Return boundaries, lengths, and task plan from active-task vector.
    """
    if len(active_task) == 0:
        return [], [], []

    boundaries = [0]
    plan = [str(active_task[0])]

    for i in range(1, len(active_task)):
        if str(active_task[i]) != str(active_task[i - 1]):
            boundaries.append(i)
            plan.append(str(active_task[i]))

    lengths = []
    for j, start in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(active_task)
        lengths.append(end - start)

    return boundaries, lengths, plan


def _aggregate_condition_continual(
    path_or_file: Union[str, Path, Sequence[Union[str, Path]]],
    task_keys: Optional[Sequence[str]] = None,
    filename: str = "stats_continual.npy",
) -> dict:
    """
    Aggregate one continual/switching condition across one or more runs.
    """
    paths = _expand_repeat_paths(path_or_file, expected_name=filename)

    if len(paths) == 0:
        raise ValueError("No continual paths found.")

    global_runs = [_build_continual_global_dict(_load(p)) for p in paths]

    if task_keys is None:
        task_keys = [k for k in global_runs[0].keys() if k.startswith("N_")]

    task_names = sorted(k.split("_", 1)[1] for k in task_keys)

    max_T = max(len(g["epoch"]) for g in global_runs)
    epoch = np.arange(1, max_T + 1)

    out = {
        "paths": paths,
        "n_runs": len(paths),
        "T": max_T,
        "epoch": epoch,
        "task_names": task_names,
    }

    for key in task_keys:
        arrs = []

        for g in global_runs:
            if key in g:
                arr = _nanpad_to_length(g[key], max_T)
            else:
                arr = np.full(max_T, np.nan, dtype=float)

            arrs.append(arr)

        mat = np.vstack(arrs)

        out[key] = {
            "runs": mat,
            "mean": np.nanmean(mat, axis=0),
            "sem": _sem(mat, axis=0),
        }

    # Use first run as reference for phase/switch annotations.
    first_global = global_runs[0]
    active_task = _infer_active_task_from_global(first_global, task_names)
    boundaries, lengths, plan = _phase_boundaries_from_task(active_task)

    out["active_task_reference"] = active_task
    out["phase_boundaries"] = boundaries
    out["phase_lengths"] = lengths
    out["plan_executed"] = plan

    return out


def _infer_switch_events(cond: Optional[dict]) -> List[dict]:
    """
    Infer switch events from aggregated continual condition.
    """
    if cond is None:
        return []

    boundaries = cond.get("phase_boundaries", [])
    plan = cond.get("plan_executed", [])

    events = []

    for i in range(1, len(boundaries)):
        events.append(
            {
                "epoch": boundaries[i] + 1,
                "from_task": plan[i - 1] if i - 1 < len(plan) else None,
                "to_task": plan[i] if i < len(plan) else None,
            }
        )

    return events


# =============================================================================
# Optional phase shading
# =============================================================================

def _shade_phases(
    ax,
    phase_boundaries: Sequence[int],
    phase_lengths: Sequence[int],
    plan_executed: Sequence[str],
    alpha: float = 0.06,
):
    """
    Light background shading for switching phases.
    """
    for i, start in enumerate(phase_boundaries):
        length = phase_lengths[i]
        end = start + length

        if i % 2 == 0:
            ax.axvspan(start + 1, end + 1, color="black", alpha=alpha, lw=0)


def _shade_consensus_phases(ax, conds: Sequence[dict], T_common: int, alpha: float = 0.06):
    """
    Conservative phase shading using first condition as reference.
    """
    if len(conds) == 0:
        return

    ref = conds[0]
    _shade_phases(
        ax,
        ref.get("phase_boundaries", []),
        ref.get("phase_lengths", []),
        ref.get("plan_executed", []),
        alpha=alpha,
    )


# =============================================================================
# Plotting: multitask
# =============================================================================

def _plot_shared_N_progression_on_ax(
    ax,
    path_no_cb,
    path_cb=None,
    path_cb_2=None,
    path_cb_3=None,
    path_cb_affix=None,
    path_cb_2_affix=None,
    path_cb_3_affix=None,
    shade_auc=False,
    show_sem=True,
    representative_task=None,
    ylabel=True,
    xlabel=True,
    title="Multitask",
):
    """
    Plot representative multitask N progression on one axis.
    """
    first_paths = _expand_repeat_paths(
        path_no_cb,
        expected_name="stats_multitask.npy",
    )
    first_stats = _load(str(first_paths[0]))
    tasks = _discover_multitask_tasks(first_stats)

    if len(tasks) == 0:
        raise ValueError("No multitask tasks found. Expected keys like acc_dms or N_dms.")

    task = representative_task or tasks[0]

    if task not in tasks:
        raise ValueError(
            f"Representative task '{task}' not found.\n"
            f"Available tasks: {tasks}"
        )

    task_keys = []
    for t in tasks:
        task_keys.extend([f"acc_{t}", f"N_{t}", f"loss_{t}"])

    cond_a = _aggregate_condition_multitask(path_no_cb, task_keys)
    cond_b = _aggregate_condition_multitask(path_cb, task_keys) if path_cb else None
    cond_c = _aggregate_condition_multitask(path_cb_2, task_keys) if path_cb_2 else None
    cond_d = _aggregate_condition_multitask(path_cb_3, task_keys) if path_cb_3 else None

    conditions = [
        (cond_a, "RNN", BLUE),
        (cond_b, path_cb_affix or "CB-RNN", PINK),
        (cond_c, path_cb_2_affix or "Full Reservoir", ORANGE),
        (cond_d, path_cb_3_affix or "Interleaved Reservoir", GREEN),
    ]

    max_epoch = max(c["T"] for c, _, _ in conditions if c is not None)
    n_max_candidates = []

    for cond, label, color in conditions:
        if cond is None:
            continue

        key = f"N_{task}"

        if key not in cond:
            raise ValueError(
                f"Could not find {key} in aggregated multitask condition."
            )

        ep = cond["epoch"]
        n = cond[key]["mean"]
        sem = cond[key]["sem"]

        label_with_n = f"{label}"

        ax.step(
            ep,
            n,
            color=color,
            lw=1,
            where="post",
            label=label_with_n,
            zorder=3,
        )

        if show_sem:
            ax.fill_between(
                ep,
                n - sem,
                n + sem,
                color=color,
                alpha=0.16,
                step="post",
                zorder=2,
            )

        if shade_auc:
            ax.fill_between(
                ep,
                0,
                n,
                color=color,
                alpha=0.10,
                step="post",
                zorder=1,
            )

        n_max_candidates.append(np.nanmax(n))

    label_fs = 8.5
    tick_fs = 8.5
    title_fs = 9.5

    ax.set_ylim(0, max(n_max_candidates) + 3)
    ax.set_xlim(1, max_epoch)

    if ylabel:
        ax.set_ylabel("Task difficulty\n(N)", fontsize=label_fs)

    if xlabel:
        ax.set_xlabel("Global Epoch", fontsize=label_fs)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=title_fs, fontweight="normal")
    ax.legend(fontsize=tick_fs - 2, frameon=False)


# =============================================================================
# Plotting: task switching
# =============================================================================

def _plot_task_switch_N_only_on_axes(
    axes,
    path_no_cb,
    path_cb=None,
    path_cb_2=None,
    affix_cb=None,
    affix_cb_2=None,
    show_sem=True,
    shade_auc=False,
    annotate_switches=True,
    phase_background="none",
    xlim=None,
):
    """
    Plot task-switching N progression across one axis per task.
    """
    first_paths = _expand_repeat_paths(
        path_no_cb,
        expected_name="stats_continual.npy",
    )
    first_stats = _load(str(first_paths[0]))
    first_global = _build_continual_global_dict(first_stats)

    task_keys = [
        k for k in first_global.keys()
        if k.startswith("N_") and "_phase" not in k
        ]
    task_keys = sorted(task_keys)

    if len(task_keys) == 0:
        raise ValueError("No N_* task keys found in stats_continual.npy.")

    cond_a = _aggregate_condition_continual(path_no_cb, task_keys=task_keys)
    cond_b = _aggregate_condition_continual(path_cb, task_keys=task_keys) if path_cb else None
    cond_c = _aggregate_condition_continual(path_cb_2, task_keys=task_keys) if path_cb_2 else None

    conds_present = [c for c in (cond_a, cond_b, cond_c) if c is not None]

    tasks = cond_a["task_names"]

    if len(tasks) != len(axes):
        raise ValueError(
            f"Need exactly {len(tasks)} axes for switching tasks, got {len(axes)}.\n"
            f"Tasks found: {tasks}"
        )

    conditions = [
        (cond_a, "RNN", BLUE),
        (cond_b, affix_cb or "CB-RNN", PINK),
        (cond_c, affix_cb_2 or "Full Reservoir", ORANGE),
    ]

    max_epoch = max(c["T"] for c in conds_present)

    switch_events_a = _infer_switch_events(cond_a)
    switch_events_b = _infer_switch_events(cond_b) if cond_b is not None else []
    switch_events_c = _infer_switch_events(cond_c) if cond_c is not None else []

    switch_events_by_condition = [
        (switch_events_a, BLUE),
        (switch_events_b, PINK),
        (switch_events_c, ORANGE),
    ]

    for col, task in enumerate(tasks):
        ax = axes[col]

        if phase_background == "reference":
            _shade_phases(
                ax,
                cond_a["phase_boundaries"],
                cond_a["phase_lengths"],
                cond_a["plan_executed"],
            )
        elif phase_background == "consensus":
            T_common = min(c["T"] for c in conds_present)
            _shade_consensus_phases(ax, conds_present, T_common)

        n_max_candidates = []

        for cond, label, color in conditions:
            if cond is None:
                continue

            key = f"N_{task}"

            if key not in cond:
                continue

            ep = cond["epoch"]
            n = cond[key]["mean"]
            sem = cond[key]["sem"]

            label_with_n = f"{label}"

            ax.step(
                ep,
                n,
                where="post",
                lw=1,
                color=color,
                label=label_with_n,
                zorder=3,
            )

            if show_sem:
                ax.fill_between(
                    ep,
                    n - sem,
                    n + sem,
                    step="post",
                    color=color,
                    alpha=0.15,
                    zorder=2,
                )

            if shade_auc:
                ax.fill_between(
                    ep,
                    0,
                    n,
                    step="post",
                    color=color,
                    alpha=0.10,
                    zorder=1,
                )

            n_max_candidates.append(np.nanmax(n))

        if annotate_switches:
            for events, color in switch_events_by_condition:
                for ev in events:
                    if task == ev["to_task"]:
                        ax.axvline(
                            ev["epoch"],
                            color=color,
                            lw=0.8,
                            ls="--",
                            alpha=0.5,
                            zorder=0,
                        )

        label_fs = 8.5
        tick_fs = 8.5
        title_fs = 9.5

        ax.set_xlim(1, xlim if xlim is not None else max_epoch)
        ax.set_ylim(0, max(n_max_candidates) + 2)
        ax.set_xlabel("Global Epoch", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

        if col == 0:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
            ax.spines[["top", "right"]].set_visible(False)
        else:
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.spines[["top", "right", "left"]].set_visible(False)

        ax.set_title(
            _display_task_name(task),
            fontweight="normal",
            fontsize=title_fs,
        )

    # Put legend only on first switching axis if needed.
    # Usually the multitask panel already has the full legend, so this is left off.
    # axes[0].legend(fontsize=6, frameon=False)


# =============================================================================
# Optional adaptation summary
# =============================================================================

def _epochs_to_first_advance_after_switch(
    stats_path: Union[str, Path],
    first_task: str = "dms",
    second_task: str = "parity",
    filename: str = "stats_continual.npy",
    max_global_epochs: Optional[int] = None,
) -> Optional[float]:
    """
    Compute simple adaptation speed:
    number of epochs after the first switch into `second_task`
    until N_second_task first increases.

    This is optional and only used if you re-enable the inset.
    """
    p = _expand_repeat_paths(stats_path, expected_name=filename)[0]
    g = _build_continual_global_dict(_load(p))

    task_names = _discover_continual_tasks(g)
    active = _infer_active_task_from_global(g, task_names)

    T = len(g["epoch"])

    if max_global_epochs is not None:
        T = min(T, max_global_epochs)
        active = active[:T]

    # Find first switch into second task.
    switch_idx = None

    for i in range(1, T):
        if str(active[i - 1]) == first_task and str(active[i]) == second_task:
            switch_idx = i
            break

    if switch_idx is None:
        return None

    key = f"N_{second_task}"

    if key not in g:
        return None

    n = np.asarray(g[key], dtype=float)[:T]
    baseline = n[switch_idx]

    for j in range(switch_idx + 1, T):
        if np.isfinite(n[j]) and n[j] > baseline:
            return float(j - switch_idx)

    return None


def _add_switch_adaptation_boxplot_inset(
    ax,
    groups: Dict[str, Sequence[Union[str, Path]]],
    filename: str = "stats_continual.npy",
    first_task: str = "dms",
    second_task: str = "parity",
    max_global_epochs: Optional[int] = 650,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    inset_bounds: Tuple[float, float, float, float] = (0.52, 0.08, 0.43, 0.43),
    show_points: bool = True,
):
    """
    Optional inset showing epochs to first N advance after switch.
    """
    inset = ax.inset_axes(inset_bounds)

    if labels is None:
        labels = list(groups.keys())

    if colors is None:
        colors = DEFAULT_COLORS[:len(labels)]

    data = []
    out = {}

    for label in labels:
        vals = []

        for run_path in groups[label]:
            val = _epochs_to_first_advance_after_switch(
                run_path,
                first_task=first_task,
                second_task=second_task,
                filename=filename,
                max_global_epochs=max_global_epochs,
            )

            if val is not None:
                vals.append(val)

        vals = np.asarray(vals, dtype=float)
        data.append(vals)
        out[label] = vals

    bp = inset.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)

    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("black")
            item.set_linewidth(0.8)

    if show_points:
        rng = np.random.default_rng(0)

        for i, vals in enumerate(data, start=1):
            if len(vals) == 0:
                continue

            x = i + rng.normal(0, 0.035, size=len(vals))
            inset.scatter(x, vals, s=8, color=colors[i - 1], alpha=0.8, zorder=3)

    inset.set_xticks(np.arange(1, len(labels) + 1))
    inset.set_xticklabels(labels, rotation=30, ha="right", fontsize=5)
    inset.tick_params(axis="y", labelsize=5)
    inset.set_ylabel("Epochs", fontsize=6)
    inset.set_title("Adaptation", fontsize=6, pad=1.5)
    inset.spines[["top", "right"]].set_visible(False)

    return inset, out


# =============================================================================
# Main public plotting function
# =============================================================================

def plot_multitask_and_switching_combined(
    # multitask inputs
    mt_path_no_cb,
    mt_path_cb=None,
    mt_path_cb_2=None,
    mt_path_cb_3=None,
    mt_path_cb_affix=None,
    mt_path_cb_2_affix=None,
    mt_path_cb_3_affix=None,
    mt_representative_task=None,

    # switching inputs
    sw_path_no_cb=None,
    sw_path_cb=None,
    sw_path_cb_2=None,
    sw_affix_cb=None,
    sw_affix_cb_2=None,
    sw_dont_merge=False,

    # shared style
    show_sem=True,
    shade_auc=False,
    annotate_switches=True,
    phase_background="none",
    save_path=None,
    width_ratios=(1.5, 1.0, 1.0),

    show_switch_adaptation_inset=False,
    switch_adaptation_groups=None,
    switch_adaptation_labels=None,
    switch_adaptation_colors=None,
    switch_adaptation_inset_bounds=(0.52, 0.08, 0.43, 0.43),

    # figure sizing
    linewidth_pt=397.5,
    fig_height=1.0,
    spacer_ratio=0.15,
    switch_xlim=650,
):
    """
    Combined multitask + task-switching figure.

    Parameters
    ----------
    mt_path_no_cb, mt_path_cb, mt_path_cb_2, mt_path_cb_3:
        Paths to stats_multitask.npy files or run directories containing stats_multitask.npy.

    sw_path_no_cb, sw_path_cb, sw_path_cb_2:
        Paths to stats_continual.npy files or run directories containing stats_continual.npy.

    mt_representative_task:
        Which multitask task to plot, e.g. "dms" or "parity".
        If None, uses the first discovered task.

    save_path:
        If provided, saves SVG/PDF/PNG according to extension or explicit SVG fallback.
    """
    if sw_path_no_cb is None:
        raise ValueError("sw_path_no_cb must be provided for the switching panels.")

    inches_per_pt = 1 / 72.27
    fig_width = linewidth_pt * inches_per_pt

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=(width_ratios[0], spacer_ratio, width_ratios[1], width_ratios[2]),
        wspace=0.08,
    )

    ax_mt = fig.add_subplot(gs[0, 0])
    ax_sw1 = fig.add_subplot(gs[0, 2])
    ax_sw2 = fig.add_subplot(gs[0, 3])

    # Left panel: multitask.
    _plot_shared_N_progression_on_ax(
        ax=ax_mt,
        path_no_cb=mt_path_no_cb,
        path_cb=mt_path_cb,
        path_cb_2=mt_path_cb_2,
        path_cb_3=mt_path_cb_3,
        path_cb_affix=mt_path_cb_affix,
        path_cb_2_affix=mt_path_cb_2_affix,
        path_cb_3_affix=mt_path_cb_3_affix,
        shade_auc=shade_auc,
        show_sem=show_sem,
        representative_task=mt_representative_task,
        ylabel=True,
        xlabel=True,
        title="Multitask",
    )

    # Right panels: switching.
    _plot_task_switch_N_only_on_axes(
        axes=[ax_sw1, ax_sw2],
        path_no_cb=sw_path_no_cb,
        path_cb=sw_path_cb,
        path_cb_2=sw_path_cb_2,
        affix_cb=sw_affix_cb,
        affix_cb_2=sw_affix_cb_2,
        show_sem=show_sem,
        shade_auc=shade_auc,
        annotate_switches=annotate_switches,
        phase_background=phase_background,
        xlim=switch_xlim,
    )

    ax_sw2.set_ylabel("")

    switch_adaptation_out = None

    if show_switch_adaptation_inset:
        if switch_adaptation_groups is None:
            switch_adaptation_groups = {}

            if sw_path_no_cb is not None:
                paths = sw_path_no_cb if isinstance(sw_path_no_cb, (list, tuple)) else [sw_path_no_cb]
                switch_adaptation_groups["RNN"] = [_as_run_dir(p) for p in paths]

            if sw_path_cb is not None:
                label = sw_affix_cb if sw_affix_cb is not None else "CB-RNN"
                paths = sw_path_cb if isinstance(sw_path_cb, (list, tuple)) else [sw_path_cb]
                switch_adaptation_groups[label] = [_as_run_dir(p) for p in paths]

            if sw_path_cb_2 is not None:
                label = sw_affix_cb_2 if sw_affix_cb_2 is not None else "CB-RNN 2"
                paths = sw_path_cb_2 if isinstance(sw_path_cb_2, (list, tuple)) else [sw_path_cb_2]
                switch_adaptation_groups[label] = [_as_run_dir(p) for p in paths]

        if len(switch_adaptation_groups) > 0:
            _, switch_adaptation_out = _add_switch_adaptation_boxplot_inset(
                ax=ax_sw2,
                groups=switch_adaptation_groups,
                filename="stats_continual.npy",
                first_task="dms",
                second_task="parity",
                max_global_epochs=switch_xlim,
                labels=switch_adaptation_labels,
                colors=switch_adaptation_colors,
                inset_bounds=switch_adaptation_inset_bounds,
                show_points=True,
            )

    if save_path:
        save_path = Path(save_path)

        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)

        for ax in [ax_mt, ax_sw1, ax_sw2]:
            ax.set_facecolor("white")
            ax.patch.set_alpha(1.0)

        suffix = save_path.suffix.lower().replace(".", "")

        if suffix == "":
            suffix = "svg"
            save_path = save_path.with_suffix(".svg")

        fig.savefig(
            save_path,
            bbox_inches="tight",
            format=suffix,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            transparent=False,
        )
    else:
        plt.show()

    if show_switch_adaptation_inset:
        return fig, [ax_mt, ax_sw1, ax_sw2], switch_adaptation_out

    return fig, [ax_mt, ax_sw1, ax_sw2]
