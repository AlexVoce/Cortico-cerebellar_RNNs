import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Loading / repeat expansion
# ---------------------------------------------------------------------

def _load_stats(path):
    return np.load(path, allow_pickle=True).item()


def _expand_single_task_paths(path_or_dir, expected_name="stats.npy"):
    """
    Accept either:
      - a single stats.npy file
      - a run folder ending in _network_<k>
      - a directory containing repeat subfolders
    """
    p = Path(path_or_dir)

    if p.is_file():
        if p.name != expected_name:
            raise ValueError(f"Expected {expected_name}, got {p.name}")
        run_dir = p.parent
        stats_name = p.name

    elif p.is_dir():
        maybe = p / expected_name
        if maybe.exists():
            return [maybe]

        files = sorted(p.glob(f"**/{expected_name}"))
        if not files:
            raise FileNotFoundError(f"No {expected_name} files found under {p}")
        return files

    else:
        raise FileNotFoundError(f"Could not find file or directory: {p}")

    run_name = run_dir.name
    parent = run_dir.parent

    m = re.match(r"^(.*)_network_(\d+)$", run_name)
    if m is None:
        return [run_dir / stats_name]

    stem = m.group(1)
    candidate_dirs = sorted(parent.glob(f"{stem}_network_*"))

    stats_files = []
    for d in candidate_dirs:
        f = d / stats_name
        if f.exists():
            stats_files.append(f)

    if not stats_files:
        raise FileNotFoundError(f"No repeated {expected_name} files found for stem '{stem}' under {parent}")

    return stats_files


def _get_single_n_series(stats):
    if "n_task" not in stats:
        raise KeyError("Expected key 'n_task' in single-task stats.npy")
    return np.asarray(stats["n_task"], dtype=int)


def _get_single_acc_series(stats):
    for key in ["accuracy", "acc", "score"]:
        if key in stats:
            return np.asarray(stats[key], dtype=float)
    raise KeyError("Could not find single-task accuracy key in stats.npy")


def _auc_curve(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    if len(y) < 2:
        return np.nan
    return np.trapz(y, x=x)


def _first_stable_threshold_epoch(acc, start_idx, threshold=85.0, patience=3):
    """
    First epoch e >= start_idx such that acc[e:e+patience] are all >= threshold.
    Returns epoch index (0-based), or None if never achieved.
    """
    acc = np.asarray(acc, dtype=float)
    last_start = len(acc) - patience + 1
    if last_start <= start_idx:
        return None

    for e in range(start_idx, last_start):
        if np.all(acc[e:e+patience] >= threshold):
            return e
    return None

def _clip_single_run_series(n_series, acc_series=None, loss_series=None, clip_len=None):
    n_series = np.asarray(n_series, dtype=int)
    acc_series = None if acc_series is None else np.asarray(acc_series, dtype=float)
    loss_series = None if loss_series is None else np.asarray(loss_series, dtype=float)

    L = len(n_series)
    if clip_len is not None:
        L = min(L, int(clip_len))

    n_series = n_series[:L]
    if acc_series is not None:
        acc_series = acc_series[:L]
    if loss_series is not None:
        loss_series = loss_series[:L]

    return n_series, acc_series, loss_series

def find_shared_clip_len_for_target_N(
    base_dir,
    groups,
    target_N,
    mode="earliest_mean_reach",
):
    """
    Determine a single shared clip_len for all groups in one task comparison.

    mode:
      - "earliest_mean_reach": earliest epoch where any group mean reaches target_N
      - "latest_mean_reach": latest epoch where all groups that can reach target_N have done so
    """
    base_dir = Path(base_dir)
    all_subdirs = [p for p in base_dir.iterdir() if p.is_dir()]

    group_mean_curves = {}

    for name, cfg in groups.items():
        pat = cfg["pattern"]
        matching = [p for p in all_subdirs if pat in p.name]
        if len(matching) == 0:
            raise RuntimeError(f"No matching runs found for pattern '{pat}' in {base_dir}")

        exemplar = matching[0] / "stats.npy"
        paths = _expand_single_task_paths(exemplar, expected_name="stats.npy")

        n_curves = []
        min_len = None
        for p in paths:
            stats = _load_stats(p)
            n = _get_single_n_series(stats)
            min_len = len(n) if min_len is None else min(min_len, len(n))
            n_curves.append(n)

        n_curves = [n[:min_len] for n in n_curves]
        mean_curve = np.nanmean(np.stack(n_curves, axis=0), axis=0)
        group_mean_curves[name] = mean_curve

    reach_epochs = {}
    for name, mean_curve in group_mean_curves.items():
        reached = np.where(mean_curve >= target_N)[0]
        reach_epochs[name] = None if len(reached) == 0 else int(reached[0] + 1)

    valid_epochs = [e for e in reach_epochs.values() if e is not None]
    if len(valid_epochs) == 0:
        raise RuntimeError(f"No group mean reached target_N={target_N}")

    if mode == "earliest_mean_reach":
        return min(valid_epochs), reach_epochs
    elif mode == "latest_mean_reach":
        return max(valid_epochs), reach_epochs
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
def _load_single_stats_clipped(path, clip_len=None, clip_N=None):
    """
    Load one single-task stats.npy and return a clipped stats dict.
    """
    stats = _load_stats(path)

    n_series = _get_single_n_series(stats)
    acc_series = _get_single_acc_series(stats)
    loss_series = np.asarray(stats["loss"], dtype=float) if "loss" in stats else None

    n_series, acc_series, loss_series = _clip_single_run_series(
        n_series,
        acc_series=acc_series,
        loss_series=loss_series,
        clip_len=clip_len,
    )

    clipped = dict(stats)  # shallow copy
    clipped["n_task"] = n_series

    # preserve whichever accuracy key exists
    for key in ["accuracy", "acc", "score"]:
        if key in clipped:
            clipped[key] = acc_series
            break

    if loss_series is not None and "loss" in clipped:
        clipped["loss"] = loss_series

    return clipped
# ---------------------------------------------------------------------
# Metrics for one single-task run
# ---------------------------------------------------------------------

def compute_solve_times_for_single_run(stats, threshold=85.0, patience=3):
    """
    Returns dict: N -> epochs-to-solve at that N
    where solve_time = first stable threshold epoch - entry epoch

    This assumes the curriculum only advances after hitting criterion, which matches
    your single-task training setup.
    """
    n_series = _get_single_n_series(stats)
    acc_series = _get_single_acc_series(stats)

    solve_times = {}
    unique_ns = np.unique(n_series)

    for N in unique_ns:
        entry_idxs = np.where(n_series == N)[0]
        if len(entry_idxs) == 0:
            continue
        entry = int(entry_idxs[0])

        solved = _first_stable_threshold_epoch(
            acc_series, start_idx=entry, threshold=threshold, patience=patience
        )

        solve_times[int(N)] = np.nan if solved is None else int(solved - entry)

    return solve_times


def compute_single_run_summary(stats, threshold=98.0, patience=1):
    """
    Summary metrics for one already-clipped single-task run.
    """
    n_series = _get_single_n_series(stats)
    acc_series = _get_single_acc_series(stats)
    solve_times = compute_solve_times_for_single_run(stats, threshold=threshold, patience=patience)

    solved_ns = sorted([n for n, t in solve_times.items() if np.isfinite(t)])

    return {
        "n_series": n_series,
        "acc_series": acc_series,
        "solve_times": solve_times,
        "auc_N": _auc_curve(n_series),
        "final_N": int(n_series[-1]) if len(n_series) > 0 else np.nan,
        "max_solved_N": max(solved_ns) if len(solved_ns) else np.nan,
        "epochs": len(n_series),
    }

# ---------------------------------------------------------------------
# Aggregate one group of repeated runs
# ---------------------------------------------------------------------

def analyse_single_task_group(
    path_or_dir,
    threshold=85.0,
    patience=3,
    clip_len=None,
    clip_N=None,
):
    """
    analyse one repeated condition (e.g. CB or RNN baseline), with optional clipping.
    """
    paths = _expand_single_task_paths(path_or_dir, expected_name="stats.npy")
    run_summaries = []

    for p in paths:
        stats = _load_single_stats_clipped(p, clip_len=clip_len, clip_N=clip_N)
        run_summaries.append(compute_single_run_summary(stats, threshold=threshold, patience=patience))

    # scalar metrics
    auc_vals = np.array([r["auc_N"] for r in run_summaries], dtype=float)
    final_N_vals = np.array([r["final_N"] for r in run_summaries], dtype=float)
    max_solved_vals = np.array([r["max_solved_N"] for r in run_summaries], dtype=float)

    # per-N solve times
    all_ns = sorted(set().union(*[set(r["solve_times"].keys()) for r in run_summaries]))
    solve_time_by_N = {}
    for N in all_ns:
        vals = []
        for r in run_summaries:
            vals.append(r["solve_times"].get(N, np.nan))
        vals = np.asarray(vals, dtype=float)
        solve_time_by_N[N] = {
            "values": vals,
            "mean": np.nanmean(vals),
            "std": np.nanstd(vals),
            "sem": np.nanstd(vals) / np.sqrt(np.sum(np.isfinite(vals))) if np.sum(np.isfinite(vals)) > 0 else np.nan,
            "n": int(np.sum(np.isfinite(vals))),
        }

    return {
        "paths": paths,
        "n_runs": len(paths),
        "runs": run_summaries,
        "clip_len": clip_len,
        "clip_N": clip_N,
        "auc_N": {
            "values": auc_vals,
            "mean": np.nanmean(auc_vals),
            "std": np.nanstd(auc_vals),
            "sem": np.nanstd(auc_vals) / np.sqrt(len(auc_vals)),
        },
        "final_N": {
            "values": final_N_vals,
            "mean": np.nanmean(final_N_vals),
            "std": np.nanstd(final_N_vals),
            "sem": np.nanstd(final_N_vals) / np.sqrt(len(final_N_vals)),
        },
        "max_solved_N": {
            "values": max_solved_vals,
            "mean": np.nanmean(max_solved_vals),
            "std": np.nanstd(max_solved_vals),
            "sem": np.nanstd(max_solved_vals) / np.sqrt(len(max_solved_vals)),
        },
        "solve_time_by_N": solve_time_by_N,
    }
# ---------------------------------------------------------------------
# analyse multiple groups for one task
# ---------------------------------------------------------------------
def analyse_single_task_condition_groups(
    base_dir,
    groups,
    threshold=85.0,
    patience=3,
    clip_len=None,
    clip_N=None,
):
    """
    analyse multiple groups for one task, with optional clipping.
    """
    out = {}

    base_dir = Path(base_dir)
    all_subdirs = [p for p in base_dir.iterdir() if p.is_dir()]

    for name, cfg in groups.items():
        pat = cfg["pattern"]

        matching = [p for p in all_subdirs if pat in p.name]
        if len(matching) == 0:
            raise RuntimeError(f"No matching runs found for pattern '{pat}' in {base_dir}")

        exemplar = matching[0] / "stats.npy"
        res = analyse_single_task_group(
            exemplar,
            threshold=threshold,
            patience=patience,
            clip_len=clip_len,
            clip_N=clip_N,
        )
        res["color"] = cfg.get("color", None)
        res["pattern"] = pat
        out[name] = res

    return out


# ---------------------------------------------------------------------
# fixed-N runs: repeats aren't grouped by N in the dirname, so N has to
# be read out of each run's config.json instead
# ---------------------------------------------------------------------

def group_fixed_n_runs(base_dir, pattern=""):
    """
    Group fixed-N run directories under base_dir by the fixed N they were
    trained at (read from config.json's cli_args['ns_init']), since the
    directory name (e.g. ..._network_<k>) doesn't indicate N and repeats
    of different N values are interleaved in the network numbering.

    pattern : substring that must appear in the run dirname (like
    _collect_runs' include_substr), not a glob pattern - e.g. "noCB" to
    select only no-cerebellum runs. "" (default) matches every subdir.

    Returns {N: [run_dir, ...]}, sorted by N, each list sorted by dirname.
    """
    base_dir = Path(base_dir)
    run_dirs = sorted(
        d for d in base_dir.iterdir() if d.is_dir() and pattern in d.name
    )

    groups = defaultdict(list)
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        N = cfg["cli_args"]["ns_init"]
        groups[N].append(run_dir)

    return dict(sorted(groups.items()))


def _pad_to_len(x, L):
    x = np.asarray(x, dtype=float)
    if x.size >= L:
        return x[:L]
    out = np.full(L, np.nan, dtype=float)
    out[:x.size] = x
    return out


def average_fixed_n_loss_acc(base_dir, pattern="", stats_file="stats.npy"):
    """
    For each fixed-N group under base_dir (see group_fixed_n_runs), load
    stats.npy for every repeat and compute the mean +/- sem loss and
    accuracy over epochs across repeats. Runs missing a stats file (e.g.
    still training) are skipped.

    Returns {N: {"epochs", "loss_mean", "loss_sem", "acc_mean", "acc_sem", "n_runs"}}.
    """
    groups = group_fixed_n_runs(base_dir, pattern=pattern)

    results = {}
    for N, run_dirs in groups.items():
        losses, accs = [], []
        for run_dir in run_dirs:
            stats_path = run_dir / stats_file
            if not stats_path.exists():
                continue
            stats = _load_stats(stats_path)
            losses.append(np.asarray(stats["loss"], dtype=float))
            accs.append(np.asarray(stats["accuracy"], dtype=float))

        if not losses:
            continue

        L = max(x.size for x in losses)
        loss_arr = np.stack([_pad_to_len(x, L) for x in losses])
        acc_arr = np.stack([_pad_to_len(x, L) for x in accs])
        n_valid_loss = np.sum(np.isfinite(loss_arr), axis=0)
        n_valid_acc = np.sum(np.isfinite(acc_arr), axis=0)

        results[N] = {
            "epochs": np.arange(L),
            "loss_mean": np.nanmean(loss_arr, axis=0),
            "loss_sem": np.nanstd(loss_arr, axis=0, ddof=1) / np.sqrt(n_valid_loss),
            "acc_mean": np.nanmean(acc_arr, axis=0),
            "acc_sem": np.nanstd(acc_arr, axis=0, ddof=1) / np.sqrt(n_valid_acc),
            "n_runs": len(losses),
        }

    return results


def plot_fixed_n_loss_and_accuracy(results, title=None, figsize=(6, 6), save_path=None):
    """
    results: output of average_fixed_n_loss_acc, {N: {"epochs","loss_mean",
    "loss_sem","acc_mean","acc_sem"}}. Plots loss (top) and accuracy
    (bottom) vs epoch, one line + shaded sem band per N.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    Ns = sorted(results.keys())
    cmap = plt.get_cmap("viridis")
    colors = {N: cmap(i / max(len(Ns) - 1, 1)) for i, N in enumerate(Ns)}

    for N in Ns:
        d = results[N]
        color = colors[N]
        label = f"N={N}"

        ax_loss.plot(d["epochs"], d["loss_mean"], label=label, color=color, lw=1.5)
        ax_loss.fill_between(
            d["epochs"], d["loss_mean"] - d["loss_sem"], d["loss_mean"] + d["loss_sem"],
            color=color, alpha=0.15,
        )

        ax_acc.plot(d["epochs"], d["acc_mean"], label=label, color=color, lw=1.5)
        ax_acc.fill_between(
            d["epochs"], d["acc_mean"] - d["acc_sem"], d["acc_mean"] + d["acc_sem"],
            color=color, alpha=0.15,
        )

    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    if title:
        ax_loss.set_title(title)

    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format="svg")
    plt.show()

    return fig, (ax_loss, ax_acc)


def plot_fixed_n_comparison(
    group_results,
    colors=None,
    alpha_range=(0.35, 1.0),
    show_sem=True,
    title=None,
    figsize=(6, 6),
    save_path=None,
):
    """
    Overlay multiple average_fixed_n_loss_acc() results on one pair of
    loss/accuracy axes - e.g. CB vs RNN. Each group gets its own base
    color; within a group, alpha increases with N (lowest N = most
    transparent, highest N = most opaque) so curves stay distinguishable
    without a legend entry per N.

    group_results : {group_name: results} where results is the output of
        average_fixed_n_loss_acc().
    colors : {group_name: color}, defaults to {"CB": "salmon", "RNN": "cornflowerblue"}.
    """
    colors = colors or {"CB": "salmon", "RNN": "cornflowerblue"}

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for group_name, results in group_results.items():
        base_color = colors[group_name]
        Ns = sorted(results.keys())
        n_steps = max(len(Ns) - 1, 1)

        for i, N in enumerate(Ns):
            d = results[N]
            alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * (i / n_steps)
            label = f"{group_name}, N={N}"

            ax_loss.plot(d["epochs"], d["loss_mean"], color=base_color, alpha=alpha, lw=1.5, label=label)
            ax_acc.plot(d["epochs"], d["acc_mean"], color=base_color, alpha=alpha, lw=1.5, label=label)

            if show_sem:
                ax_loss.fill_between(
                    d["epochs"], d["loss_mean"] - d["loss_sem"], d["loss_mean"] + d["loss_sem"],
                    color=base_color, alpha=alpha * 0.2, lw=0,
                )
                ax_acc.fill_between(
                    d["epochs"], d["acc_mean"] - d["acc_sem"], d["acc_mean"] + d["acc_sem"],
                    color=base_color, alpha=alpha * 0.2, lw=0,
                )

    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    if title:
        ax_loss.set_title(title)

    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend(frameon=False, fontsize=7, ncol=2)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format="svg")
    plt.show()

    return fig, (ax_loss, ax_acc)