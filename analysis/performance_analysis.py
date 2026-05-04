"""
curriculum_metrics.py

Drop-in utilities for analysing curriculum-learning experiments across:
    1. single-task runs
    2. multitask runs
    3. task-switching / continual-switch runs

Core conventions
----------------
- AUC is raw area under the N-over-epoch curve: np.trapz(N, dx=1).
- Half-max speed uses a GLOBAL maximum N across all compared model groups,
  not each model's own maximum.
- For task-switching:
    * switch AUC is computed from reconstructed global task-specific plot lines.
    * inactive-task periods are forward-filled with the last attained N.
    * post-switch adaptation speed is epochs required to advance from phase-start
      N to N+1 for active post-switch phases.

Example usage
-------------
groups = {
    "CB-RNN": ["/path/to/cb_seed1", "/path/to/cb_seed2", "/path/to/cb_seed3"],
    "RNN-only": ["/path/to/rnn_seed1", "/path/to/rnn_seed2", "/path/to/rnn_seed3"],
    "Full reservoir": [...],
    "Interleaved reservoir": [...],
}

single = analyse_single_task_groups(groups, filename="stats.npy", epoch_budget=1000)
multi = analyse_multitask_groups(groups, filename="stats_multitask.npy", task_keys=("N_dms",), epoch_budget=1000)
switch = analyse_switch_groups(groups, filename="stats_continual.npy", max_global_epochs=650)

display(single["summary"])
display(multi["summary"])
display(switch["summary"])
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Generic helpers
# ============================================================

DEFAULT_STATS_CANDIDATES = (
    "stats.npy",
    "stats_multitask.npy",
    "stats_continual.npy",
)


def load_stats(run_path: str | Path, filename: Optional[str] = None) -> dict:
    """
    Load a stats dictionary from a run directory.

    If filename is supplied, loads that file.
    Otherwise tries common names:
        stats.npy
        stats_multitask.npy
        stats_continual.npy
    """
    run_path = Path(run_path)

    if filename is not None:
        f = run_path / filename
        if not f.exists():
            raise FileNotFoundError(f"Could not find {f}")
        return np.load(f, allow_pickle=True).item()

    for fname in DEFAULT_STATS_CANDIDATES:
        f = run_path / fname
        if f.exists():
            return np.load(f, allow_pickle=True).item()

    raise FileNotFoundError(
        f"No stats file found in {run_path}. Tried: {DEFAULT_STATS_CANDIDATES}"
    )


def make_run_specs(groups: Dict[str, Sequence[str | Path]]) -> List[dict]:
    """
    Convert:
        {"CB-RNN": [path1, path2], "RNN-only": [path1, path2]}
    into a list of run specs.
    """
    specs = []
    for model, paths in groups.items():
        for repeat, path in enumerate(paths):
            specs.append({
                "model": model,
                "repeat": repeat,
                "path": str(path),
            })
    return specs


def sem(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    n = int(finite.sum())
    if n <= 1:
        return np.nan
    return float(np.nanstd(x[finite], ddof=1) / np.sqrt(n))


def sd(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    n = int(finite.sum())
    if n <= 1:
        return np.nan
    return float(np.nanstd(x[finite], ddof=1))


def first_epoch_reaching(y: Sequence[float], target: float, one_based: bool = True) -> float:
    """
    First epoch where y >= target.
    Returns 1-based epoch by default.
    """
    y = np.asarray(y, dtype=float)

    if len(y) == 0 or not np.isfinite(target):
        return np.nan

    hits = np.where(y >= target)[0]

    if len(hits) == 0:
        return np.nan

    return float(hits[0] + 1) if one_based else float(hits[0])


def raw_auc(y: Sequence[float]) -> float:
    """
    Raw AUC of N over epoch.
    """
    y = np.asarray(y, dtype=float)

    if len(y) < 2 or np.all(np.isnan(y)):
        return np.nan

    return float(np.trapz(y, dx=1))


def mean_N(y: Sequence[float]) -> float:
    """
    Mean curriculum level.
    Equivalent to length-normalised AUC in N units.
    """
    y = np.asarray(y, dtype=float)

    if len(y) == 0 or np.all(np.isnan(y)):
        return np.nan

    return float(np.nanmean(y))


def pad_with_final_value(
    y: Sequence[float],
    target_length: int,
    use_max: bool = False,
) -> np.ndarray:
    """
    Pad a series to target_length.

    If use_max=False, pads with y[-1].
    If use_max=True, pads with np.nanmax(y). This matches early-stop logic
    where solved runs are held at their maximum attained N.
    """
    y = np.asarray(y, dtype=float)

    if len(y) >= target_length:
        return y

    if len(y) == 0:
        return np.full(target_length, np.nan)

    pad_value = np.nanmax(y) if use_max else y[-1]
    pad_len = target_length - len(y)

    return np.concatenate([y, np.full(pad_len, pad_value)])


def apply_budget(
    y: Sequence[float],
    epoch_budget: Optional[int] = None,
    pad_to_length: Optional[int] = None,
    pad_with_max: bool = False,
) -> np.ndarray:
    """
    Optionally pad, then optionally truncate to epoch_budget.
    """
    y = np.asarray(y, dtype=float)

    if pad_to_length is not None:
        y = pad_with_final_value(
            y,
            target_length=pad_to_length,
            use_max=pad_with_max,
        )

    if epoch_budget is not None:
        y = y[:epoch_budget]

    return y


def summarise_by_model(
    df: pd.DataFrame,
    metrics: Sequence[str],
    model_col: str = "model",
    n_col_name: str = "n_runs",
) -> pd.DataFrame:
    """
    Summarise metric columns by model with mean, SD, SEM.
    """
    rows = []

    for model, g in df.groupby(model_col):
        row = {
            model_col: model,
            n_col_name: g["repeat"].nunique() if "repeat" in g.columns else len(g),
        }

        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)

            row[f"{metric}_mean"] = (
                float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            )
            row[f"{metric}_sd"] = sd(vals)
            row[f"{metric}_sem"] = sem(vals)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Single-task analysis
# ============================================================

def get_single_task_N_series(
    stats: dict,
    preferred_keys: Sequence[str] = (
        "n_task",
        "N",
        "N_task",
        "N_dms",
        "N_parity",
        "N_oddball",
    ),
) -> Tuple[np.ndarray, str]:
    """
    Extract a single curriculum trajectory from a stats dictionary.
    """
    for key in preferred_keys:
        if key in stats:
            return np.asarray(stats[key], dtype=float), key

    candidates = [k for k in stats.keys() if "N" in k or "n_task" in k]

    raise KeyError(
        "Could not find a single-task N trajectory. "
        f"Candidate keys: {candidates}. All keys: {list(stats.keys())}"
    )


def compute_global_max_single_task(
    run_specs: List[dict],
    filename: Optional[str] = None,
    epoch_budget: Optional[int] = None,
    pad_to_length: Optional[int] = None,
    pad_with_max: bool = False,
    preferred_keys: Sequence[str] = (
        "n_task",
        "N",
        "N_task",
        "N_dms",
        "N_parity",
        "N_oddball",
    ),
) -> float:
    vals = []

    for run in run_specs:
        stats = load_stats(run["path"], filename=filename)
        y, _ = get_single_task_N_series(stats, preferred_keys=preferred_keys)
        y = apply_budget(
            y,
            epoch_budget=epoch_budget,
            pad_to_length=pad_to_length,
            pad_with_max=pad_with_max,
        )

        if len(y) > 0 and not np.all(np.isnan(y)):
            vals.append(np.nanmax(y))

    return float(np.nanmax(vals)) if vals else np.nan


def compute_single_task_run_metrics(
    run: dict,
    global_max_N: float,
    filename: Optional[str] = None,
    epoch_budget: Optional[int] = None,
    pad_to_length: Optional[int] = None,
    pad_with_max: bool = False,
    preferred_keys: Sequence[str] = (
        "n_task",
        "N",
        "N_task",
        "N_dms",
        "N_parity",
        "N_oddball",
    ),
) -> dict:
    stats = load_stats(run["path"], filename=filename)
    y_raw, key_used = get_single_task_N_series(stats, preferred_keys=preferred_keys)

    y = apply_budget(
        y_raw,
        epoch_budget=epoch_budget,
        pad_to_length=pad_to_length,
        pad_with_max=pad_with_max,
    )

    half_global = 0.5 * global_max_N if np.isfinite(global_max_N) else np.nan

    return {
        "model": run["model"],
        "repeat": run["repeat"],
        "run_path": run["path"],
        "N_key": key_used,
        "epochs_logged": len(y_raw),
        "epochs_used": len(y),
        "global_max_N": global_max_N,
        "half_global_max_N": half_global,
        "max_N": float(np.nanmax(y)) if len(y) else np.nan,
        "final_N": float(y[-1]) if len(y) else np.nan,
        "epochs_to_half_global_max_N": first_epoch_reaching(y, half_global),
        "raw_AUC": raw_auc(y),
        "mean_N": mean_N(y),
    }


def analyse_single_task_groups(
    groups: Dict[str, Sequence[str | Path]],
    filename: Optional[str] = None,
    epoch_budget: Optional[int] = None,
    pad_to_length: Optional[int] = None,
    pad_with_max: bool = False,
    preferred_keys: Sequence[str] = (
        "n_task",
        "N",
        "N_task",
        "N_dms",
        "N_parity",
        "N_oddball",
    ),
) -> Dict[str, object]:
    """
    Analyse single-task runs across any number of model groups.

    Parameters
    ----------
    groups:
        Dict mapping model label to list of run directories.
    epoch_budget:
        Fixed epoch window to analyse, e.g. 1000.
    pad_to_length:
        Optional length to pad all runs before truncating. Useful for early-stopped runs.
    pad_with_max:
        If padding, use max N rather than final N.

    Returns
    -------
    dict with:
        run_metrics
        summary
        global_max_N
    """
    run_specs = make_run_specs(groups)

    global_max_N = compute_global_max_single_task(
        run_specs,
        filename=filename,
        epoch_budget=epoch_budget,
        pad_to_length=pad_to_length,
        pad_with_max=pad_with_max,
        preferred_keys=preferred_keys,
    )

    rows = [
        compute_single_task_run_metrics(
            run,
            global_max_N=global_max_N,
            filename=filename,
            epoch_budget=epoch_budget,
            pad_to_length=pad_to_length,
            pad_with_max=pad_with_max,
            preferred_keys=preferred_keys,
        )
        for run in run_specs
    ]

    run_metrics = pd.DataFrame(rows)

    summary = summarise_by_model(
        run_metrics,
        metrics=[
            "max_N",
            "final_N",
            "epochs_to_half_global_max_N",
            "raw_AUC",
            "mean_N",
        ],
    )

    return {
        "run_metrics": run_metrics,
        "summary": summary,
        "global_max_N": global_max_N,
    }


# ============================================================
# Multitask analysis
# ============================================================

def get_multitask_N_keys(
    stats: dict,
    task_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Find multitask curriculum keys.

    If task_keys is supplied, uses those exact keys.
    Otherwise returns keys matching N_<task>, excluding phase keys.

    If your multitask training has one shared curriculum stored as N_dms,
    pass task_keys=("N_dms",).
    """
    if task_keys is not None:
        missing = [k for k in task_keys if k not in stats]
        if missing:
            raise KeyError(
                f"Missing requested multitask keys: {missing}. "
                f"Available keys: {list(stats.keys())}"
            )
        return list(task_keys)

    keys = []
    pattern = re.compile(r"^N_(.+)$")

    for k in stats.keys():
        if "_phase" in k:
            continue
        if pattern.match(k):
            keys.append(k)

    if len(keys) == 0:
        for k in ("n_task", "N", "N_task"):
            if k in stats:
                return [k]

    if len(keys) == 0:
        raise KeyError(f"Could not find multitask N keys. Available keys: {list(stats.keys())}")

    return sorted(keys)


def compute_global_max_multitask(
    run_specs: List[dict],
    filename: Optional[str] = "stats_multitask.npy",
    task_keys: Optional[Sequence[str]] = None,
    epoch_budget: Optional[int] = None,
) -> float:
    vals = []

    for run in run_specs:
        stats = load_stats(run["path"], filename=filename)
        keys = get_multitask_N_keys(stats, task_keys=task_keys)

        for key in keys:
            y = apply_budget(stats[key], epoch_budget=epoch_budget)

            if len(y) > 0 and not np.all(np.isnan(y)):
                vals.append(np.nanmax(y))

    return float(np.nanmax(vals)) if vals else np.nan


def compute_multitask_run_metrics(
    run: dict,
    global_max_N: float,
    filename: Optional[str] = "stats_multitask.npy",
    task_keys: Optional[Sequence[str]] = None,
    epoch_budget: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    stats = load_stats(run["path"], filename=filename)
    keys = get_multitask_N_keys(stats, task_keys=task_keys)

    rows = []
    half_global = 0.5 * global_max_N if np.isfinite(global_max_N) else np.nan

    for key in keys:
        y_raw = np.asarray(stats[key], dtype=float)
        y = apply_budget(y_raw, epoch_budget=epoch_budget)

        task = key.replace("N_", "") if key.startswith("N_") else key

        rows.append({
            "model": run["model"],
            "repeat": run["repeat"],
            "run_path": run["path"],
            "task_key": key,
            "task": task,
            "epochs_logged": len(y_raw),
            "epochs_used": len(y),
            "global_max_N": global_max_N,
            "half_global_max_N": half_global,
            "max_N": float(np.nanmax(y)) if len(y) else np.nan,
            "final_N": float(y[-1]) if len(y) else np.nan,
            "epochs_to_half_global_max_N": first_epoch_reaching(y, half_global),
            "raw_AUC": raw_auc(y),
            "mean_N": mean_N(y),
        })

    task_metrics = pd.DataFrame(rows)

    run_summary = {
        "model": run["model"],
        "repeat": run["repeat"],
        "run_path": run["path"],
        "n_task_curves": len(task_metrics),
        "global_max_N": global_max_N,
        "half_global_max_N": half_global,
        "mean_max_N": task_metrics["max_N"].mean(),
        "mean_final_N": task_metrics["final_N"].mean(),
        "mean_epochs_to_half_global_max_N": task_metrics[
            "epochs_to_half_global_max_N"
        ].mean(),
        "mean_AUC": task_metrics["raw_AUC"].mean(),
        "sum_AUC": task_metrics["raw_AUC"].sum(),
        "mean_N": task_metrics["mean_N"].mean(),
    }

    return task_metrics, run_summary


def analyse_multitask_groups(
    groups: Dict[str, Sequence[str | Path]],
    filename: Optional[str] = "stats_multitask.npy",
    task_keys: Optional[Sequence[str]] = None,
    epoch_budget: Optional[int] = None,
) -> Dict[str, object]:
    """
    Analyse multitask runs across any number of model groups.

    If task_keys is None, all N_<task> keys are used.
    If your multitask plot uses only N_dms as a shared curriculum, pass:
        task_keys=("N_dms",)
    """
    run_specs = make_run_specs(groups)

    global_max_N = compute_global_max_multitask(
        run_specs,
        filename=filename,
        task_keys=task_keys,
        epoch_budget=epoch_budget,
    )

    task_dfs = []
    run_rows = []

    for run in run_specs:
        task_df, run_summary = compute_multitask_run_metrics(
            run,
            global_max_N=global_max_N,
            filename=filename,
            task_keys=task_keys,
            epoch_budget=epoch_budget,
        )
        task_dfs.append(task_df)
        run_rows.append(run_summary)

    task_metrics = pd.concat(task_dfs, ignore_index=True)
    run_metrics = pd.DataFrame(run_rows)

    summary = summarise_by_model(
        run_metrics,
        metrics=[
            "mean_max_N",
            "mean_final_N",
            "mean_epochs_to_half_global_max_N",
            "mean_AUC",
            "sum_AUC",
            "mean_N",
        ],
    )

    task_summary = (
        task_metrics
        .groupby(["model", "task"], dropna=False)
        .agg(
            max_N_mean=("max_N", "mean"),
            max_N_sd=("max_N", sd),
            max_N_sem=("max_N", sem),
            final_N_mean=("final_N", "mean"),
            final_N_sd=("final_N", sd),
            final_N_sem=("final_N", sem),
            epochs_to_half_global_max_N_mean=("epochs_to_half_global_max_N", "mean"),
            epochs_to_half_global_max_N_sd=("epochs_to_half_global_max_N", sd),
            epochs_to_half_global_max_N_sem=("epochs_to_half_global_max_N", sem),
            raw_AUC_mean=("raw_AUC", "mean"),
            raw_AUC_sd=("raw_AUC", sd),
            raw_AUC_sem=("raw_AUC", sem),
            mean_N_mean=("mean_N", "mean"),
            mean_N_sd=("mean_N", sd),
            mean_N_sem=("mean_N", sem),
            n=("repeat", "count"),
        )
        .reset_index()
    )

    return {
        "task_metrics": task_metrics,
        "task_summary": task_summary,
        "run_metrics": run_metrics,
        "summary": summary,
        "global_max_N": global_max_N,
    }


# ============================================================
# Task-switching analysis
# ============================================================

def get_switch_N_keys(stats: dict) -> List[Tuple[str, str, int]]:
    """
    Finds keys like:
        N_dms_phase0
        N_parity_phase1

    Returns:
        [(key, task, phase), ...]
    """
    pattern = re.compile(r"^N_(.+)_phase(\d+)$")
    keys = []

    for k in stats.keys():
        m = pattern.match(k)

        if m is not None:
            task = m.group(1)
            phase = int(m.group(2))
            keys.append((k, task, phase))

    if len(keys) == 0:
        raise KeyError(
            "Could not find switch keys like N_dms_phase0. "
            f"Available keys: {list(stats.keys())}"
        )

    return sorted(keys, key=lambda x: (x[2], x[1]))


def is_active_switch_task(
    task: str,
    phase: int,
    first_task: str = "dms",
    second_task: str = "parity",
) -> bool:
    """
    Assumes alternating switch order:
        phase 0 = first_task
        phase 1 = second_task
        phase 2 = first_task
        phase 3 = second_task
        ...

    Default:
        DMS -> parity -> DMS -> parity
    """
    if phase % 2 == 0:
        return task == first_task
    return task == second_task


def reconstruct_global_task_lines(
    stats: dict,
    tasks: Sequence[str] = ("dms", "parity"),
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Reconstruct global plot lines for each task from phase-local arrays.

    Inactive periods are forward-filled with the last attained N.
    This reproduces the logic of plotting DMS/parity on a shared global
    epoch axis where one task is inactive while the other is trained.
    """
    N_keys = get_switch_N_keys(stats)

    phase_rows = []
    cursor = 0

    for key, task, phase in N_keys:
        y = np.asarray(stats[key], dtype=float)
        L = len(y)

        phase_rows.append({
            "key": key,
            "task": task,
            "phase": phase,
            "start": cursor,
            "end": cursor + L,
            "length": L,
        })

        cursor += L

    total_len = cursor

    global_lines = {
        task: np.full(total_len, np.nan)
        for task in tasks
    }

    cursor = 0
    for key, task, phase in N_keys:
        y = np.asarray(stats[key], dtype=float)
        L = len(y)

        if task in global_lines:
            global_lines[task][cursor:cursor + L] = y

        cursor += L

    # Forward-fill inactive periods.
    for task in tasks:
        line = global_lines[task].copy()

        valid = np.where(~np.isnan(line))[0]
        if len(valid) == 0:
            continue

        # Fill leading NaNs with first observed value.
        first_valid = valid[0]
        if first_valid > 0:
            line[:first_valid] = line[first_valid]

        last_val = line[first_valid]
        for i in range(first_valid, len(line)):
            if np.isnan(line[i]):
                line[i] = last_val
            else:
                last_val = line[i]

        global_lines[task] = line

    phase_info = pd.DataFrame(phase_rows)

    return global_lines, phase_info


def compute_global_max_switch_lines(
    run_specs: List[dict],
    filename: Optional[str] = "stats_continual.npy",
    tasks: Sequence[str] = ("dms", "parity"),
    max_global_epochs: Optional[int] = None,
) -> float:
    """
    Global maximum N across reconstructed switch plot lines from all groups.
    """
    vals = []

    for run in run_specs:
        stats = load_stats(run["path"], filename=filename)
        global_lines, _ = reconstruct_global_task_lines(stats, tasks=tasks)

        for task in tasks:
            y = global_lines[task]

            if max_global_epochs is not None:
                y = y[:max_global_epochs]

            if len(y) > 0 and not np.all(np.isnan(y)):
                vals.append(np.nanmax(y))

    return float(np.nanmax(vals)) if vals else np.nan


def compute_switch_line_metrics_for_run(
    run: dict,
    global_max_N: float,
    filename: Optional[str] = "stats_continual.npy",
    tasks: Sequence[str] = ("dms", "parity"),
    max_global_epochs: Optional[int] = 650,
) -> Tuple[pd.DataFrame, dict, Dict[str, np.ndarray], pd.DataFrame]:
    """
    Compute AUC etc. from reconstructed global plot lines.
    """
    stats = load_stats(run["path"], filename=filename)
    global_lines, phase_info = reconstruct_global_task_lines(stats, tasks=tasks)

    half_global = 0.5 * global_max_N if np.isfinite(global_max_N) else np.nan

    rows = []

    for task in tasks:
        y_full = np.asarray(global_lines[task], dtype=float)
        y = y_full[:max_global_epochs] if max_global_epochs is not None else y_full

        rows.append({
            "model": run["model"],
            "repeat": run["repeat"],
            "run_path": run["path"],
            "task": task,
            "epochs_used": len(y),
            "global_max_N": global_max_N,
            "half_global_max_N": half_global,
            "line_max_N": float(np.nanmax(y)) if len(y) else np.nan,
            "line_final_N": float(y[-1]) if len(y) else np.nan,
            "line_AUC": raw_auc(y),
            "line_mean_N": mean_N(y),
            "line_epochs_to_half_global_max_N": first_epoch_reaching(y, half_global),
        })

    line_metrics = pd.DataFrame(rows)

    run_summary = {
        "model": run["model"],
        "repeat": run["repeat"],
        "run_path": run["path"],
        "global_max_N": global_max_N,
        "half_global_max_N": half_global,
        "mean_line_max_N": line_metrics["line_max_N"].mean(),
        "mean_line_final_N": line_metrics["line_final_N"].mean(),
        "mean_line_AUC": line_metrics["line_AUC"].mean(),
        "sum_line_AUC": line_metrics["line_AUC"].sum(),
        "mean_line_mean_N": line_metrics["line_mean_N"].mean(),
        "mean_line_epochs_to_half_global_max_N": line_metrics[
            "line_epochs_to_half_global_max_N"
        ].mean(),
    }

    return line_metrics, run_summary, global_lines, phase_info


def epochs_to_next_N_after_switch(y: Sequence[float], min_increment: int = 1) -> float:
    """
    Number of epochs after phase onset required to advance from starting N to N+1.

    Example:
        y = [5, 5, 5, 6, 6]
        returns 4, because the first epoch with N >= 6 is the 4th epoch.
    """
    y = np.asarray(y, dtype=float)

    if len(y) == 0 or np.all(np.isnan(y)):
        return np.nan

    start_N = y[0]
    target_N = start_N + min_increment

    return first_epoch_reaching(y, target_N, one_based=True)


def compute_switch_adaptation_metrics_for_run(
    run: dict,
    filename: Optional[str] = "stats_continual.npy",
    first_task: str = "dms",
    second_task: str = "parity",
    active_only: bool = True,
    post_switch_only: bool = True,
    max_global_epochs: Optional[int] = 650,
) -> pd.DataFrame:
    """
    Compute post-switch adaptation speed from phase-local arrays.

    Metric:
        epochs_to_next_N = epochs required for active task trajectory to advance
        from phase-start N to N+1.

    Assumes alternating switch order:
        phase 0 = first_task
        phase 1 = second_task
        phase 2 = first_task
        ...

    If max_global_epochs is supplied, phases beginning after the global cutoff
    are excluded; phases crossing the cutoff are clipped.
    """
    stats = load_stats(run["path"], filename=filename)
    N_keys = get_switch_N_keys(stats)

    rows = []
    cursor = 0

    for key, task, phase in N_keys:
        y_raw = np.asarray(stats[key], dtype=float)
        start = cursor
        end = cursor + len(y_raw)
        cursor = end

        if active_only and not is_active_switch_task(
            task,
            phase,
            first_task=first_task,
            second_task=second_task,
        ):
            continue

        if post_switch_only and phase == 0:
            continue

        if max_global_epochs is not None:
            if start >= max_global_epochs:
                continue
            clipped_end = min(end, max_global_epochs)
            usable_len = clipped_end - start
            y = y_raw[:usable_len]
        else:
            y = y_raw

        if len(y) == 0:
            continue

        rows.append({
            "model": run["model"],
            "repeat": run["repeat"],
            "run_path": run["path"],
            "key": key,
            "task": task,
            "phase": phase,
            "global_start": start,
            "global_end": end,
            "phase_len_original": len(y_raw),
            "phase_len_used": len(y),
            "start_N": float(y[0]),
            "next_target_N": float(y[0] + 1),
            "final_N": float(y[-1]),
            "max_N": float(np.nanmax(y)),
            "N_gained": float(np.nanmax(y) - y[0]),
            "epochs_to_next_N": epochs_to_next_N_after_switch(y),
            "phase_AUC": raw_auc(y),
            "phase_mean_N": mean_N(y),
        })

    return pd.DataFrame(rows)


def analyse_switch_groups(
    groups: Dict[str, Sequence[str | Path]],
    filename: Optional[str] = "stats_continual.npy",
    tasks: Sequence[str] = ("dms", "parity"),
    first_task: str = "dms",
    second_task: str = "parity",
    max_global_epochs: Optional[int] = 650,
) -> Dict[str, object]:
    """
    Analyse task-switching runs across any number of model groups.

    Returns
    -------
    dict with:
        line_metrics:
            one row per reconstructed global task line per run.
        run_metrics:
            one row per run, summarising switch AUC/mean N.
        adaptation_metrics:
            one row per active post-switch phase per run.
        adaptation_per_run:
            one row per run, averaging adaptation speed across successful phases.
        summary:
            per-model summary of line AUC, mean N, final N, and adaptation speed.
        global_max_N:
            global max from reconstructed switch lines.
    """
    run_specs = make_run_specs(groups)

    global_max_N = compute_global_max_switch_lines(
        run_specs,
        filename=filename,
        tasks=tasks,
        max_global_epochs=max_global_epochs,
    )

    line_dfs = []
    run_rows = []
    adaptation_dfs = []

    for run in run_specs:
        line_df, run_summary, _, _ = compute_switch_line_metrics_for_run(
            run,
            global_max_N=global_max_N,
            filename=filename,
            tasks=tasks,
            max_global_epochs=max_global_epochs,
        )

        adaptation_df = compute_switch_adaptation_metrics_for_run(
            run,
            filename=filename,
            first_task=first_task,
            second_task=second_task,
            active_only=True,
            post_switch_only=True,
            max_global_epochs=max_global_epochs,
        )

        line_dfs.append(line_df)
        run_rows.append(run_summary)
        adaptation_dfs.append(adaptation_df)

    line_metrics = pd.concat(line_dfs, ignore_index=True)
    run_metrics = pd.DataFrame(run_rows)

    if len(adaptation_dfs) > 0:
        adaptation_metrics = pd.concat(adaptation_dfs, ignore_index=True)
    else:
        adaptation_metrics = pd.DataFrame()

    if len(adaptation_metrics) > 0:
        # Mean ignores NaN by default. This means failed switch phases where N never
        # reaches N+1 are excluded from the speed average but retained in the raw table.
        adaptation_per_run = (
            adaptation_metrics
            .groupby(["model", "repeat", "run_path"], dropna=False)
            .agg(
                adaptation_speed_epochs=("epochs_to_next_N", "mean"),
                n_successful_switch_phases=("epochs_to_next_N", "count"),
                n_total_switch_phases=("phase", "count"),
                mean_N_gained=("N_gained", "mean"),
            )
            .reset_index()
        )
    else:
        adaptation_per_run = pd.DataFrame(
            columns=[
                "model",
                "repeat",
                "run_path",
                "adaptation_speed_epochs",
                "n_successful_switch_phases",
                "n_total_switch_phases",
                "mean_N_gained",
            ]
        )

    auc_summary = summarise_by_model(
        run_metrics,
        metrics=[
            "mean_line_max_N",
            "mean_line_final_N",
            "mean_line_AUC",
            "sum_line_AUC",
            "mean_line_mean_N",
            "mean_line_epochs_to_half_global_max_N",
        ],
    )

    if len(adaptation_per_run) > 0:
        adaptation_summary = summarise_by_model(
            adaptation_per_run,
            metrics=[
                "adaptation_speed_epochs",
                "n_successful_switch_phases",
                "mean_N_gained",
            ],
        )
        summary = auc_summary.merge(adaptation_summary, on=["model"], how="left")
    else:
        summary = auc_summary

    task_line_summary = (
        line_metrics
        .groupby(["model", "task"], dropna=False)
        .agg(
            line_AUC_mean=("line_AUC", "mean"),
            line_AUC_sd=("line_AUC", sd),
            line_AUC_sem=("line_AUC", sem),
            line_mean_N_mean=("line_mean_N", "mean"),
            line_mean_N_sd=("line_mean_N", sd),
            line_mean_N_sem=("line_mean_N", sem),
            line_final_N_mean=("line_final_N", "mean"),
            line_final_N_sd=("line_final_N", sd),
            line_final_N_sem=("line_final_N", sem),
            n=("repeat", "count"),
        )
        .reset_index()
    )

    if len(adaptation_metrics) > 0:
        adaptation_phase_summary = (
            adaptation_metrics
            .groupby(["model", "task", "phase"], dropna=False)
            .agg(
                epochs_to_next_N_mean=("epochs_to_next_N", "mean"),
                epochs_to_next_N_sd=("epochs_to_next_N", sd),
                epochs_to_next_N_sem=("epochs_to_next_N", sem),
                N_gained_mean=("N_gained", "mean"),
                N_gained_sd=("N_gained", sd),
                N_gained_sem=("N_gained", sem),
                n=("repeat", "count"),
            )
            .reset_index()
        )
    else:
        adaptation_phase_summary = pd.DataFrame()

    return {
        "line_metrics": line_metrics,
        "task_line_summary": task_line_summary,
        "run_metrics": run_metrics,
        "adaptation_metrics": adaptation_metrics,
        "adaptation_phase_summary": adaptation_phase_summary,
        "adaptation_per_run": adaptation_per_run,
        "summary": summary,
        "global_max_N": global_max_N,
    }


# ============================================================
# Optional convenience: print compact summaries
# ============================================================

def print_summary_block(title: str, out: Dict[str, object]) -> None:
    """
    Convenience printer for notebook use.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Global max N:", out.get("global_max_N", None))

    if "summary" in out:
        print("\nSummary:")
        print(out["summary"])

    if "run_metrics" in out:
        print("\nRun metrics:")
        print(out["run_metrics"])

    if "adaptation_per_run" in out:
        print("\nAdaptation per run:")
        print(out["adaptation_per_run"])