# analysis/single_task_plotting_utils.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from sklearn.metrics import auc
import seaborn as sns

def _pad_to_len(x, L):
    """Pad 1D array x to length L with NaNs."""
    x = np.asarray(x, dtype=float)
    if x.size >= L:
        return x[:L]
    out = np.full(L, np.nan, dtype=float)
    out[:x.size] = x
    return out


def _pad_to_len_last_value(x, L):
    """
    Pad 1D array x to length L by repeating the final value.
    Used for runs that stopped early because they hit the N ceiling —
    they have 'solved' the task so their N should be held flat rather
    than dropping to NaN (which would drag the group mean down).
    If x is already >= L, it is clipped to L.
    """
    x = np.asarray(x, dtype=float)
    if x.size >= L:
        return x[:L]
    pad_value = x[np.where(np.isfinite(x))[0][-1]] if np.any(np.isfinite(x)) else np.nan
    out = np.full(L, pad_value, dtype=float)
    out[:x.size] = x
    return out


def _collect_runs(base_dir, include_substr, clip_len=None, clip_N=None,
                  stats_file="stats.npy", pad_early_stops=False):
    """
    Load all runs matching include_substr.

    pad_early_stops : bool
        If True, runs whose N series ends below clip_len (i.e. they stopped
        early by hitting the N ceiling) are padded with their final N value
        rather than NaN.  This prevents fast/early-stopping runs from
        dragging the group mean downward in plots and AUC calculations.
        Runs that are simply shorter because they haven't finished are still
        NaN-padded so they don't contribute false signal beyond their length.

    clip_N : int or None
        If set, N values are clipped to this ceiling (y-axis cap only).
        The x-axis / time dimension is never truncated based on clip_N.
    """
    run_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and (include_substr in d)
    ]

    raw = []
    kept, skipped = [], []

    for run in sorted(run_dirs):
        path = os.path.join(base_dir, run, stats_file)
        try:
            stats = np.load(path, allow_pickle=True).item()
        except Exception as e:
            skipped.append((run, f"load_failed:{type(e).__name__}"))
            continue

        missing = [k for k in ("n_task", "loss", "accuracy") if k not in stats]
        if missing:
            skipped.append((run, f"missing_key:{','.join(missing)}"))
            continue

        n = len(stats["n_task"])
        if n == 0:
            skipped.append((run, "empty"))
            continue

        raw.append({
            "run": run,
            "N": np.asarray(stats["n_task"], dtype=float),
            "loss": np.asarray(stats["loss"], dtype=float),
            "acc": np.asarray(stats["accuracy"], dtype=float),
        })
        kept.append(run)

    if len(raw) == 0:
        raise RuntimeError(f"No valid runs found for pattern '{include_substr}' in {base_dir}")

    # Decide final length
    max_len = max(r["N"].size for r in raw)
    L = max_len if clip_len is None else min(int(clip_len), max_len)

    Ns, losses, accs, aucs = [], [], [], []
    for r in raw:
        n_raw = r["N"]

        # Decide padding strategy for this run's N series.
        # A run "finished early" if it's shorter than L AND its final N
        # equals clip_N (it hit the ceiling).  Only apply last-value padding
        # in that case; genuinely short runs still get NaN padding.
        hit_ceiling = (
            pad_early_stops
            and clip_N is not None
            and n_raw.size < L
            and np.nanmax(n_raw) >= clip_N
        )

        if hit_ceiling:
            Np = _pad_to_len_last_value(n_raw, L)
            Lp = _pad_to_len_last_value(r["loss"], L)
            Ap = _pad_to_len_last_value(r["acc"], L)
        else:
            Np = _pad_to_len(n_raw, L)
            Lp = _pad_to_len(r["loss"], L)
            Ap = _pad_to_len(r["acc"], L)

        # Clip N values to ceiling if specified
        if clip_N is not None:
            Np = np.clip(Np, None, clip_N)
            valid = Np <= clip_N
            Np[~valid] = np.nan
            Lp[~valid] = np.nan
            Ap[~valid] = np.nan

        Ns.append(Np)
        losses.append(Lp)
        accs.append(Ap)

        # AUC over valid (non-NaN) prefix only
        valid = np.isfinite(Np)
        if valid.sum() >= 2:
            x = np.arange(L)[valid]
            y = Np[valid]
            aucs.append(auc(x, y))
        else:
            aucs.append(np.nan)

    return {
        "runs": kept,
        "skipped": skipped,
        "clip_len_used": L,
        "N": np.stack(Ns, axis=0),
        "loss": np.stack(losses, axis=0),
        "acc": np.stack(accs, axis=0),
        "auc": np.asarray(aucs, dtype=float),
    }

def _mean_std_nan(x):
    return np.nanmean(x, axis=0), np.nanstd(x, axis=0)

def average_and_plot_runs(
    base_dir,
    groups,
    clip_len=None,
    clip_N=None,
    pad_early_stops=False,
    title="Average learning curves",
    x_label="Epoch",
    figsize=(8, 5),
    plot_metric2="loss",
    log_y2=True,
    show_metric2=True,   # NEW
    show=True,
    fig_height=3,
    save_path=None,
):
    linewidth= 397.5
    inches_per_pt = 1 / 72.27
    fig_width = linewidth * inches_per_pt
    # ---- load + aggregate ----
    agg = {}
    max_L = 0

    for name, cfg in groups.items():
        pat = cfg["pattern"]
        data = _collect_runs(
            base_dir,
            pat,
            clip_len=clip_len,
            clip_N=clip_N,
            pad_early_stops=pad_early_stops,
        )

        L = data["clip_len_used"]
        max_L = max(max_L, L)

        N_mean, N_std = _mean_std_nan(data["N"])
        loss_mean, loss_std = _mean_std_nan(data["loss"])
        acc_mean, acc_std = _mean_std_nan(data["acc"])

        agg[name] = {
            "N_raw": data["N"],
            "loss_raw": data["loss"],
            "acc_raw": data["acc"],
            "N_mean": N_mean,
            "N_std": N_std,
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "auc": data["auc"],
            "runs": data["runs"],
            "skipped": data["skipped"],
            "color": cfg.get("color", None),
            "L": L,
        }

    epochs = np.arange(max_L)


    # ---- plotting ----
    if figsize is None:
        figsize = (fig_width, fig_height)
    label_fs = 9 
    tick_fs = 8 
    title_fs = 9
    if show_metric2:
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
        ax1, ax2 = axs
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        axs = np.array([ax1])
        ax2 = None

    # ---- top subplot: curriculum progression ----
    for name, d in agg.items():
        L = d["L"]
        ax1.plot(
            epochs[:L],
            d["N_mean"],
            lw=1.7,
            label=f"{name}",
            color=d["color"],
        )
        ax1.fill_between(
            epochs[:L],
            d["N_mean"] - d["N_std"],
            d["N_mean"] + d["N_std"],
            alpha=0.12,
            color=d["color"],
        )

    ax1.set_ylabel("Task Difficulty (N)", fontsize=label_fs, fontweight="bold")
    ax1.set_title(title, fontsize=title_fs, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=tick_fs)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax1.set_xlim(0, max_L - 1)
    ax1.legend(loc="lower center", frameon=False, fontsize=tick_fs)

    if not show_metric2:
        ax1.set_xlabel(x_label, fontsize=label_fs, fontweight="bold")

    # ---- optional lower subplot ----
    if show_metric2:
        key_mean = f"{plot_metric2}_mean"
        key_std = f"{plot_metric2}_std"
        ylab = "Loss" if plot_metric2 == "loss" else "Accuracy (%)"

        for name, d in agg.items():
            L = d["L"]
            m = d[key_mean]
            s = d[key_std]

            ax2.plot(
                epochs[:L],
                m,
                lw=1.2,
                label=name,
                color=d["color"],
            )
            ax2.fill_between(
                epochs[:L],
                m - s,
                m + s,
                alpha=0.12,
                color=d["color"],
            )

        ax2.set_ylabel(ylab, fontsize=label_fs, fontweight="bold")
        ax2.set_xlabel(x_label, fontsize=label_fs, fontweight="bold")

        if log_y2 and plot_metric2 == "loss":
            ax2.set_yscale("log")

        ax2.legend(frameon=True, fontsize=tick_fs-2)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format="svg")

    if show:
        plt.show()

    return agg, fig, axs


## for plotting multiple tasks in one figure
def _aggregate_one_task(
    base_dir,
    groups,
    clip_len=None,
    clip_N=None,
    pad_early_stops=False,
):
    """
    Aggregate runs for one task directory.
    Returns the same agg dict structure the plotting code expects.
    """
    agg = {}
    max_L = 0

    for name, cfg in groups.items():
        pat = cfg["pattern"]
        data = _collect_runs(base_dir, pat, clip_len=clip_len, clip_N=clip_N,
                             pad_early_stops=pad_early_stops)

        L = data["clip_len_used"]
        max_L = max(max_L, L)

        N_mean, N_std = _mean_std_nan(data["N"])
        loss_mean, loss_std = _mean_std_nan(data["loss"])
        acc_mean, acc_std = _mean_std_nan(data["acc"])

        agg[name] = {
            "N_raw": data["N"],
            "loss_raw": data["loss"],
            "acc_raw": data["acc"],
            "N_mean": N_mean, "N_std": N_std,
            "loss_mean": loss_mean, "loss_std": loss_std,
            "acc_mean": acc_mean, "acc_std": acc_std,
            "auc": data["auc"],
            "runs": data["runs"],
            "skipped": data["skipped"],
            "color": cfg.get("color", None),
            "L": L,
        }

    return agg, max_L

def average_and_plot_runs_multitask(
    task_dirs,
    groups,
    clip_lens=None,
    clip_Ns=None,
    pad_early_stop_tasks=None,
    task_titles=None,
    figure_title="",
    x_label="Epoch",
    plot_lower_metric=False,
    plot_metric2="loss",
    log_y2=True,
    show=True,
    figsize=None,
    fig_height=1.1,
    sharey_top=False,
    sharey_bottom=False,
    shade_auc=False,
    shade_sem=True,
    save_path=None,
):
    """
    pad_early_stop_tasks : list/set of task names
        For these tasks, runs that hit the N ceiling before the epoch budget
        is exhausted are padded with their final N value (rather than NaN),
        so fast runs don't drag the group mean downward.
        E.g. pad_early_stop_tasks={"Oddball"} to fix oddball whilst leaving
        DMS and Parity behaviour unchanged.
    """
    scale_factor = 1
    linewidth= 397.5
    inches_per_pt = 1 / 72.27
    fig_width = linewidth * inches_per_pt
    task_names = list(task_dirs.keys())
    n_tasks = len(task_names)

    clip_lens = clip_lens or {}
    clip_Ns = clip_Ns or {}
    task_titles = task_titles or {}
    pad_early_stop_tasks = set(pad_early_stop_tasks or [])

    n_rows = 2 if plot_lower_metric else 1
    figsize = (fig_width * scale_factor,fig_height) if figsize is None else figsize
    # figsize = figsize or (4.5 * n_tasks, 8 if plot_lower_metric else 4.5)

    fig, axs = plt.subplots(
        n_rows,
        n_tasks,
        figsize=figsize,
        sharex=False,
        sharey="row" if (sharey_top and sharey_bottom and plot_lower_metric) else False,
        constrained_layout=True
    )

    # normalise axs shape to [n_rows, n_tasks]
    if n_tasks == 1 and n_rows == 1:
        axs = np.array([[axs]])
    elif n_tasks == 1:
        axs = np.array(axs).reshape(n_rows, 1)
    elif n_rows == 1:
        axs = np.array(axs).reshape(1, n_tasks)

    task_aggs = {}

    global_top_min = np.inf
    global_top_max = -np.inf
    global_bottom_min = np.inf
    global_bottom_max = -np.inf

    # First pass: aggregate everything
    meta = {}
    for task in task_names:
        agg, max_L = _aggregate_one_task(
            task_dirs[task],
            groups,
            clip_len=clip_lens.get(task, None),
            clip_N=clip_Ns.get(task, None),
            pad_early_stops=task in pad_early_stop_tasks,
        )
        task_aggs[task] = agg
        meta[task] = {"max_L": max_L}

        for _, d in agg.items():
            global_top_min = min(global_top_min, np.nanmin(d["N_mean"] - d["N_std"]))
            global_top_max = max(global_top_max, np.nanmax(d["N_mean"] + d["N_std"]))

            if plot_lower_metric:
                key_mean = f"{plot_metric2}_mean"
                key_std = f"{plot_metric2}_std"
                global_bottom_min = min(global_bottom_min, np.nanmin(d[key_mean] - d[key_std]))
                global_bottom_max = max(global_bottom_max, np.nanmax(d[key_mean] + d[key_std]))

    # Second pass: plot
    for col, task in enumerate(task_names):
        agg = task_aggs[task]
        max_L = meta[task]["max_L"]
        epochs = np.arange(max_L)

        ax1 = axs[0, col]

        for name, d in agg.items():
            L_plot = min(max_L, d["L"], len(d["N_mean"]))
            x = epochs[:L_plot]
            y = d["N_mean"][:L_plot]
            s = d["N_std"][:L_plot]

            if shade_auc:
                ax1.fill_between(
                    x, 0, y,
                    alpha=0.2,
                    color=d["color"],
                    zorder=1
                )

            ax1.plot(
                x, y,
                lw=1.5,
                label=f"{name}",
                color=d["color"],
                zorder=3
            )
            if shade_sem:
                ax1.fill_between(
                    x, y - s, y + s,
                    alpha=0.12,
                    color=d["color"],
                    zorder=2
                )

        label_fs = 9 * scale_factor
        tick_fs = 8 * scale_factor
        title_fs = 9 * scale_factor
        ax1.set_ylabel("Task Difficulty (N)" if col == 0 else "", fontsize=label_fs, fontweight="bold")
        # do 5 y ticks if possible        
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax1.set_title(task_titles.get(task, task), fontsize=title_fs, fontweight="bold")
        ax1.set_xlim(0, max_L - 1)

        if plot_lower_metric:
            ax1.tick_params(bottom=False, labelbottom=False)
            ax1.spines["bottom"].set_visible(False)
            sns.despine(ax=ax1, bottom=True, top=True, right=True)
        else:
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax1.set_xlabel(x_label, fontsize=label_fs, fontweight="bold")
            sns.despine(ax=ax1, top=True, right=True)
        
        ax1.tick_params(axis="both", labelsize=tick_fs)

        if sharey_top:
            ax1.set_ylim(global_top_min, global_top_max)

        if plot_lower_metric:
            ax2 = axs[1, col]

            key_mean = f"{plot_metric2}_mean"
            key_std = f"{plot_metric2}_std"
            ylab = "Loss" if plot_metric2 == "loss" else "Accuracy (%)"

            for name, d in agg.items():
                m = d[key_mean]
                s = d[key_std]
                L_plot = min(max_L, d["L"], len(m))
                x = epochs[:L_plot]
                y = m[:L_plot]
                sd = s[:L_plot]

                ax2.plot(x, y, lw=1.5, label=name, color=d["color"])
                ax2.fill_between(x, y - sd, y + sd, alpha=0.12, color=d["color"])

            ax2.set_ylabel(ylab if col == 0 else "", fontsize=label_fs, fontweight="bold")
            ax2.set_xlabel(x_label, fontsize=label_fs, fontweight="bold")

            if log_y2 and plot_metric2 == "loss":
                ax2.set_yscale("log")

            if sharey_bottom:
                ax2.set_ylim(global_bottom_min, global_bottom_max)

            sns.despine(ax=ax2, top=True, right=True)

        if col == n_tasks - 1:
            ax1.legend(loc="lower right", frameon=False, fontsize=tick_fs-2)

    fig.suptitle(figure_title, fontsize=15, fontweight="bold", y=0.98)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight",format="svg")

    if show:
        plt.show()

    return task_aggs, fig, axs


# ---------------------------------------------------------------------
# Plot mean epochs to solve each N for one task
# ---------------------------------------------------------------------

def plot_mean_epochs_to_solve_each_N(
    analysis_dict,
    title="Mean epochs to solve each N",
    figsize=(7, 5),
    save_path=None,
    dpi=150,
):
    fig, ax = plt.subplots(figsize=figsize)

    for name, res in analysis_dict.items():
        ns = sorted(res["solve_time_by_N"].keys())
        means = np.array([res["solve_time_by_N"][N]["mean"] for N in ns], dtype=float)
        sems = np.array([res["solve_time_by_N"][N]["sem"] for N in ns], dtype=float)

        ax.plot(ns, means, marker="o", lw=2, label=f"{name} (n={res['n_runs']})", color=res.get("color", None))
        ax.fill_between(ns, means - sems, means + sems, alpha=0.15, color=res.get("color", None))

    ax.set_xlabel("Curriculum level (N)")
    ax.set_ylabel("Mean epochs to solve N")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------
# Summary bar plot for one task
# ---------------------------------------------------------------------

def plot_single_task_summary_bars(
    analysis_dict,
    metrics=("auc_N", "final_N", "max_solved_N"),
    figsize=(10, 4),
    save_path=None,
    dpi=150,
):
    metric_titles = {
        "auc_N": "AUC of N progression",
        "final_N": "Final N reached",
        "max_solved_N": "Max solved N",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    group_names = list(analysis_dict.keys())
    x = np.arange(len(group_names))

    for ax, metric in zip(axes, metrics):
        means = np.array([analysis_dict[g][metric]["mean"] for g in group_names], dtype=float)
        sems = np.array([analysis_dict[g][metric]["sem"] for g in group_names], dtype=float)
        colors = [analysis_dict[g].get("color", None) for g in group_names]

        ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=20, ha="right")
        ax.set_title(metric_titles.get(metric, metric))
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------
# Multi-task figure: mean epochs to solve each N
# ---------------------------------------------------------------------

def plot_mean_epochs_to_solve_each_N_multitask(
    analyses_by_task,
    task_titles=None,
    figsize=None,
    save_path=None,
    dpi=150,
):
    task_names = list(analyses_by_task.keys())
    n_tasks = len(task_names)
    figsize = figsize or (4.5 * n_tasks, 4)

    fig, axes = plt.subplots(1, n_tasks, figsize=figsize, sharey=False)
    if n_tasks == 1:
        axes = [axes]

    task_titles = task_titles or {}

    for ax, task in zip(axes, task_names):
        analysis_dict = analyses_by_task[task]

        for name, res in analysis_dict.items():
            ns = sorted(res["solve_time_by_N"].keys())
            means = np.array([res["solve_time_by_N"][N]["mean"] for N in ns], dtype=float)
            sems = np.array([res["solve_time_by_N"][N]["sem"] for N in ns], dtype=float)

            ax.plot(ns, means, marker="o", lw=2, label=name, color=res.get("color", None))
            ax.fill_between(ns, means - sems, means + sems, alpha=0.15, color=res.get("color", None))

        ax.set_title(task_titles.get(task, task))
        ax.set_xlabel("N")
        ax.set_ylabel("Mean epochs to solve N" if ax is axes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------
# Multi-task figure: summary metrics bars
# ---------------------------------------------------------------------

def plot_single_task_summary_bars_multitask(
    analyses_by_task,
    metric="auc_N",
    task_order=None,
    figsize=(10, 4),
    save_path=None,
    dpi=150,
):
    task_order = task_order or list(analyses_by_task.keys())
    first_task = task_order[0]
    group_names = list(analyses_by_task[first_task].keys())

    x = np.arange(len(task_order))
    width = 0.8 / len(group_names)

    fig, ax = plt.subplots(figsize=figsize)

    for i, group in enumerate(group_names):
        means = np.array([analyses_by_task[t][group][metric]["mean"] for t in task_order], dtype=float)
        sems = np.array([analyses_by_task[t][group][metric]["sem"] for t in task_order], dtype=float)
        color = analyses_by_task[first_task][group].get("color", None)

        ax.bar(
            x + (i - (len(group_names) - 1) / 2) * width,
            means,
            width=width,
            yerr=sems,
            capsize=4,
            label=group,
            color=color,
            alpha=0.85,
        )

    metric_titles = {
        "auc_N": "AUC of N progression",
        "final_N": "Final N reached",
        "max_solved_N": "Max solved N",
    }

    ax.set_xticks(x)
    ax.set_xticklabels(task_order)
    ax.set_ylabel(metric_titles.get(metric, metric))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()

    return fig, ax


def compute_auc_differences_all_pairs(task_aggs, task_dirs, matched_pairs):
    """
    Computes AUC differences for all task x matched-model-pair comparisons.

    Assumes:
        task_aggs[task][model_name]["auc"]

    Returns
    -------
    df_diff : pd.DataFrame
        One row per task x pair.
    """
    rows = []

    for task in task_dirs.keys():
        print(f"\n{task}")

        for pair_name, (cb_model, rnn_model) in matched_pairs.items():
            if cb_model not in task_aggs[task]:
                print(f"Skipping {pair_name}: missing {cb_model}")
                continue

            if rnn_model not in task_aggs[task]:
                print(f"Skipping {pair_name}: missing {rnn_model}")
                continue

            auc_cb = np.asarray(task_aggs[task][cb_model]["auc"], dtype=float)
            auc_rnn = np.asarray(task_aggs[task][rnn_model]["auc"], dtype=float)

            if len(auc_cb) != len(auc_rnn):
                print(
                    f"Warning: {task} {pair_name} has unequal lengths: "
                    f"{cb_model}={len(auc_cb)}, {rnn_model}={len(auc_rnn)}. "
                    "Truncating to shortest."
                )
                n = min(len(auc_cb), len(auc_rnn))
                auc_cb = auc_cb[:n]
                auc_rnn = auc_rnn[:n]

            diff = auc_cb - auc_rnn

            mean_diff = np.mean(diff)
            sd_diff = np.std(diff, ddof=1) if len(diff) > 1 else np.nan

            print(
                f"{pair_name}: "
                f"{mean_diff:.4f} ± {sd_diff:.4f} SD, n={len(diff)}"
            )

            rows.append({
                "task": task,
                "pair": pair_name,
                "cb_model": cb_model,
                "rnn_model": rnn_model,
                "mean_diff": mean_diff,
                "sd_diff": sd_diff,
                "n": len(diff),
                "diffs": diff,
            })

    return pd.DataFrame(rows)
def plot_auc_differences_all_pairs(
    df_auc_diffs,
    pair_order=None,
    task_order=None,
    colors=None,
    figsize=(7, 3.5),
    save_path=None,
):
    """
    Bar plot of CB-RNN minus matched RNN AUC differences.

    Bars show mean difference.
    Error bars show SD across repeats.
    """
    if pair_order is None:
        pair_order = list(df_auc_diffs["pair"].unique())

    if task_order is None:
        task_order = list(df_auc_diffs["task"].unique())

    if colors is None:
        colors = {
            "GC64 - RNN128": "#FFD0CB",
            "GC256 - RNN202": "salmon",
        }

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(task_order))
    n_pairs = len(pair_order)
    width = 0.8 / n_pairs

    for i, pair in enumerate(pair_order):
        means = []
        sds = []

        for task in task_order:
            row = df_auc_diffs[
                (df_auc_diffs["task"] == task) &
                (df_auc_diffs["pair"] == pair)
            ]

            if len(row) == 0:
                means.append(np.nan)
                sds.append(np.nan)
            else:
                means.append(row["mean_diff"].iloc[0])
                sds.append(row["sd_diff"].iloc[0])

        offset = (i - (n_pairs - 1) / 2) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=sds,
            label=pair,
            color=colors.get(pair, "gray"),
            capsize=3,
            linewidth=0.5,
            edgecolor=None,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("AUC difference\nCB-RNN − matched RNN")
    ax.set_xticks(x)
    ax.set_xticklabels(task_order)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()

    return fig, ax