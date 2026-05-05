# scaling.py
import os
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]


from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
CB_COLOR  = "salmon"   # red family
RNN_COLOR = "cornflowerblue"   # blue family

# param counts in ascending order, matched pairs
PARAM_SIZES = [
    {"cb_pattern": "gc64",  "rnn_pattern": "elman_size_128",  "label": "GC64\n/H128"},
    {"cb_pattern": "gc128", "rnn_pattern": "elman_size_160",  "label": "GC128\n/H160"},
    {"cb_pattern": "gc256", "rnn_pattern": "elman_size_202",  "label": "GC256\n/H202"},
    {"cb_pattern": "gc512", "rnn_pattern": "elman_size_272",  "label": "GC512\n/H272"},
]

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})



def _find_runs(base_dir, pattern):
    base_dir = Path(base_dir)
    runs = sorted(p for p in base_dir.iterdir() if p.is_dir() and pattern in p.name)
    if not runs:
        raise RuntimeError(f"No runs found for pattern '{pattern}' in {base_dir}")
    return runs


def _extract_trainable_params(run_dir):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config.json in {run_dir}")
    with cfg_path.open("r") as f:
        cfg = json.load(f)
    return int(cfg["params"]["total"]), cfg


def _extract_max_n_within_epochs(run_dir, max_epochs=1000, min_required_epochs=50):
    stats_path = run_dir / "stats.npy"
    if not stats_path.exists():
        raise RuntimeError(f"Missing stats.npy in {run_dir}")

    stats = np.load(stats_path, allow_pickle=True).item()
    if "n_task" not in stats:
        raise KeyError(f"Expected 'n_task' in {stats_path}")

    n_series = np.asarray(stats["n_task"], dtype=float)
    n_epochs = len(n_series)

    if n_epochs < min_required_epochs:
        return None
    if n_series.size == 0:
        return None

    max_n = float(np.nanmax(n_series[:max_epochs]))
    return max_n, n_epochs


def _extract_epoch_to_reach_n(run_dir, target_N=150, min_required_epochs=50):
    stats_path = run_dir / "stats.npy"
    if not stats_path.exists():
        raise RuntimeError(f"Missing stats.npy in {run_dir}")

    stats = np.load(stats_path, allow_pickle=True).item()
    if "n_task" not in stats:
        raise KeyError(f"Expected 'n_task' in {stats_path}")

    n_series = np.asarray(stats["n_task"], dtype=float)
    n_epochs = len(n_series)

    if n_epochs < min_required_epochs:
        return None

    reached = np.where(n_series >= target_N)[0]
    if len(reached) == 0:
        return None

    return int(reached[0] + 1), n_epochs  # 1-based epoch


def find_max_n_across_runs(base_dir, pattern, max_epochs=1000, min_required_epochs=50):
    runs = _find_runs(base_dir, pattern)
    max_ns = []

    for run_dir in runs:
        result = _extract_max_n_within_epochs(
            run_dir,
            max_epochs=max_epochs,
            min_required_epochs=min_required_epochs,
        )
        if result is not None:
            max_n, _ = result
            max_ns.append(max_n)

    return float(np.nanmax(max_ns)) if len(max_ns) > 0 else None

def find_max_n_across_all_groups(task_dir, groups, max_epochs=1000, min_required_epochs=50):
    global_max_ns = []

    for label, spec in groups.items():
        runs = _find_runs(task_dir, spec["pattern"])
        for run_dir in runs:
            result = _extract_max_n_within_epochs(
                run_dir,
                max_epochs=max_epochs,
                min_required_epochs=min_required_epochs,
            )
            if result is not None:
                max_n, _ = result
                global_max_ns.append(max_n)

    return float(np.nanmax(global_max_ns)) if len(global_max_ns) > 0 else None

def find_epoch_to_fraction_of_max_n(
    run_dir,
    fraction=0.5,
    min_required_epochs=50,
    max_reference=None,
):
    """
    If max_reference is None:
        target = fraction * max N within this run
    else:
        target = fraction * max_reference
    """
    stats_path = run_dir / "stats.npy"
    if not stats_path.exists():
        raise RuntimeError(f"Missing stats.npy in {run_dir}")

    stats = np.load(stats_path, allow_pickle=True).item()
    if "n_task" not in stats:
        raise KeyError(f"Expected 'n_task' in {stats_path}")

    n_series = np.asarray(stats["n_task"], dtype=float)
    n_epochs = len(n_series)

    if n_epochs < min_required_epochs:
        return None
    if n_series.size == 0:
        return None

    if max_reference is None:
        max_n = float(np.nanmax(n_series))
    else:
        max_n = float(max_reference)

    if not np.isfinite(max_n) or max_n <= 0:
        return None

    target_n = fraction * max_n
    reached = np.where(n_series >= target_n)[0]
    if len(reached) == 0:
        return None

    return int(reached[0] + 1), n_epochs, target_n, max_n

def _collect_epoch_metrics(task_dir, pattern, mode, fraction,
                            global_max_n, min_required_epochs):
    """Return list of (n_params, epoch_metric) for all runs matching pattern."""
    runs = _find_runs(task_dir, pattern)
    pts = []
    for run_dir in runs:
        try:
            n_params, _ = _extract_trainable_params(run_dir)
        except RuntimeError:
            continue

        result = find_epoch_to_fraction_of_max_n(
            run_dir,
            fraction=fraction,
            min_required_epochs=min_required_epochs,
            max_reference=global_max_n,
        )
        if result is None:
            continue
        epoch_metric = result[0]
        pts.append((n_params, epoch_metric))
    return pts

def plot_cb_vs_rnn_scaling(
    task_configs,
    param_sizes=PARAM_SIZES,
    mode="half_global_max",
    fraction=0.5,
    fig_height=1.1,
    min_required_epochs=50,
    figsize=(8.2, 4.8),
    sharey=True,
):
    scale_factor = 1
    linewidth= 397.5
    inches_per_pt = 1 / 72.27
    fig_width = linewidth * inches_per_pt

    label_fs = 9 * scale_factor
    tick_fs = 8 * scale_factor
    title_fs = 9 * scale_factor

    n_tasks = len(task_configs)
    if figsize is None:
        figsize = (fig_width*scale_factor, fig_height)

    fig, axes = plt.subplots(1, n_tasks, figsize=figsize, sharey=sharey,constrained_layout=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, cfg in zip(axes, task_configs):
        task_dir  = Path(cfg["task_dir"])
        task_name = cfg["task_name"]

        global_max_n = cfg.get("global_reference_max_n")
        if global_max_n is None:
            global_max_n = find_max_n_across_all_groups(
                task_dir, cfg["groups"],
                min_required_epochs=min_required_epochs,
            )

        cb_means,  cb_x_vals  = [], []
        rnn_means, rnn_x_vals = [], []

        for size in param_sizes:
            # ── CB ────────────────────────────────────────────────────────
            cb_pts = _collect_epoch_metrics(
                task_dir, size["cb_pattern"], mode, fraction,
                global_max_n, min_required_epochs,
            )
            if cb_pts:
                cb_x  = np.mean([p[0] for p in cb_pts])
                cb_ys = [p[1] for p in cb_pts]
                ax.scatter(
                    [cb_x] * len(cb_ys), cb_ys,
                    color=CB_COLOR, s=10, alpha=0.4,
                    edgecolors="k", zorder=3,
                )
                cb_x_vals.append(cb_x)
                cb_means.append(np.mean(cb_ys))

            # ── RNN ───────────────────────────────────────────────────────
            rnn_pts = _collect_epoch_metrics(
                task_dir, size["rnn_pattern"], mode, fraction,
                global_max_n, min_required_epochs,
            )
            if rnn_pts:
                rnn_x  = np.mean([p[0] for p in rnn_pts])
                rnn_ys = [p[1] for p in rnn_pts]
                ax.scatter(
                    [rnn_x] * len(rnn_ys), rnn_ys,
                    color=RNN_COLOR, s=10, alpha=0.4,
                    edgecolors="k", zorder=3,
                )
                rnn_x_vals.append(rnn_x)
                rnn_means.append(np.mean(rnn_ys))


        # ── mean lines ────────────────────────────────────────────────────
        if cb_means:
            ax.plot(cb_x_vals, cb_means, color=CB_COLOR,
                    lw=1.2, marker="o", ms=4, zorder=5, label="CB-RNN")
        if rnn_means:
            ax.plot(rnn_x_vals, rnn_means, color=RNN_COLOR,
                    lw=1.2, marker="o", ms=4, zorder=5, label="RNN")

        ax.set_xlabel("Trainable parameters",fontsize=label_fs)
        ax.set_ylim(90, 575)

        
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5,integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/1e3)}K"))
        ax.tick_params(axis='both', labelsize=tick_fs)

        # ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.spines[["top", "right"]].set_visible(False)

        axes[0].set_ylabel("Epochs to half-max N", fontsize=label_fs)   
    for ax in axes[1:]:
        ax.set_ylabel("")

    # fig.suptitle("CB-RNN learns faster as model scales", fontsize=12, y=1.02)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes
