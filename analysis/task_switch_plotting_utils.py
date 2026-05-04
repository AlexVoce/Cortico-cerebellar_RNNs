import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BLUE = "cornflowerblue"
PINK = "salmon"
GREEN = "orange"

PHASE_COLORS = {
    "dms": "#DCEBFA",
    "parity": "#FCE1EA",
    "oddball": "#E6F4EA",
}


def _load(path):
    return np.load(path, allow_pickle=True).item()


def _expand_repeat_paths(path_or_file, expected_name="stats_continual.npy"):
    p = Path(path_or_file)

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
        stats_file = run_dir / stats_name
        if not stats_file.exists():
            raise FileNotFoundError(f"Missing {stats_name} in {run_dir}")
        return [stats_file]

    stem = m.group(1)
    candidate_dirs = sorted(parent.glob(f"{stem}_network_*"))

    stats_files = []
    for d in candidate_dirs:
        f = d / stats_name
        if f.exists():
            stats_files.append(f)

    if not stats_files:
        raise FileNotFoundError(
            f"No repeated {stats_name} files found for stem '{stem}' under {parent}"
        )

    return stats_files


def _smooth(x, w=5):
    if w <= 1 or len(x) < w:
        return np.array(x, dtype=float)
    kernel = np.ones(w) / w
    padded = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def _stats_to_global_series(stats):
    task_names = list(stats["task_names"])
    plan_executed = list(stats["plan_executed"])
    n_phases = len(plan_executed)

    phase_lengths = []
    for i in range(n_phases):
        loss_key = f"loss_phase{i}"
        if loss_key not in stats:
            raise KeyError(f"Missing key {loss_key}")
        phase_lengths.append(len(stats[loss_key]))

    phase_boundaries = []
    running = 0
    for L in phase_lengths:
        phase_boundaries.append(running)
        running += L
    T = running

    out = {
        "task_names": task_names,
        "plan_executed": plan_executed,
        "phase_lengths": phase_lengths,
        "phase_boundaries": phase_boundaries,
        "epoch": np.arange(1, T + 1),
    }

    out["loss"] = np.concatenate(
        [np.asarray(stats[f"loss_phase{i}"], dtype=float) for i in range(n_phases)]
    )

    for task in task_names:
        acc_chunks = []
        n_chunks = []
        for i in range(n_phases):
            acc_key = f"acc_{task}_phase{i}"
            n_key = f"N_{task}_phase{i}"
            if acc_key not in stats:
                raise KeyError(f"Missing key {acc_key}")
            if n_key not in stats:
                raise KeyError(f"Missing key {n_key}")
            acc_chunks.append(np.asarray(stats[acc_key], dtype=float))
            n_chunks.append(np.asarray(stats[n_key], dtype=float))

        out[f"acc_{task}"] = np.concatenate(acc_chunks)
        out[f"N_{task}"] = np.concatenate(n_chunks)

    return out


def _aggregate_condition(path_or_file, dont_merge=False):
    if dont_merge:
        paths = [path_or_file]
    else:
        paths = _expand_repeat_paths(path_or_file, expected_name="stats_continual.npy")

    series_list = [_stats_to_global_series(_load(p)) for p in paths]

    ref_tasks = series_list[0]["task_names"]
    ref_plan = series_list[0]["plan_executed"]

    for s, p in zip(series_list[1:], paths[1:]):
        if s["task_names"] != ref_tasks:
            raise ValueError(f"Task mismatch across repeats: {p}")
        if s["plan_executed"] != ref_plan:
            raise ValueError(f"plan_executed mismatch across repeats: {p}")

    T = min(len(s["epoch"]) for s in series_list)

    out = {
        "n_runs": len(paths),
        "paths": paths,
        "task_names": ref_tasks,
        "plan_executed": ref_plan,
        "epoch": np.arange(1, T + 1),
    }

    ref = series_list[0]
    clipped_boundaries = []
    clipped_lengths = []
    for start, L in zip(ref["phase_boundaries"], ref["phase_lengths"]):
        if start >= T:
            break
        end = min(start + L, T)
        clipped_boundaries.append(start)
        clipped_lengths.append(end - start)
    out["phase_boundaries"] = clipped_boundaries
    out["phase_lengths"] = clipped_lengths

    loss_stack = np.stack([s["loss"][:T] for s in series_list], axis=0)
    out["loss"] = {
        "mean": np.nanmean(loss_stack, axis=0),
        "sem": np.nanstd(loss_stack, axis=0, ddof=0) / np.sqrt(loss_stack.shape[0]),
        "raw": loss_stack,
    }

    for task in ref_tasks:
        acc_stack = np.stack([s[f"acc_{task}"][:T] for s in series_list], axis=0)
        n_stack = np.stack([s[f"N_{task}"][:T] for s in series_list], axis=0)

        out[f"acc_{task}"] = {
            "mean": np.nanmean(acc_stack, axis=0),
            "sem": np.nanstd(acc_stack, axis=0, ddof=0) / np.sqrt(acc_stack.shape[0]),
            "raw": acc_stack,
        }
        out[f"N_{task}"] = {
            "mean": np.nanmean(n_stack, axis=0),
            "sem": np.nanstd(n_stack, axis=0, ddof=0) / np.sqrt(n_stack.shape[0]),
            "raw": n_stack,
        }

    return out


def _shade_phases(ax, phase_boundaries, phase_lengths, plan_executed):
    for i, (start, L) in enumerate(zip(phase_boundaries, phase_lengths)):
        task = plan_executed[i]
        color = PHASE_COLORS.get(task, "#EEEEEE")
        x0 = start + 1
        x1 = start + L
        ax.axvspan(x0, x1, color=color, alpha=0.25, zorder=0)


def _active_mask_from_phases(plan_executed, phase_boundaries, phase_lengths, task, T):
    mask = np.zeros(T, dtype=float)
    for t, start, L in zip(plan_executed, phase_boundaries, phase_lengths):
        if t != task:
            continue
        end = min(start + L, T)
        if start < T:
            mask[start:end] = 1.0
    return mask


def _infer_switch_events(cond):
    events = []
    plan = cond["plan_executed"]
    boundaries = cond["phase_boundaries"]

    for i in range(1, len(boundaries)):
        b = int(boundaries[i])
        if b <= 0:
            continue

        from_task = plan[i - 1]
        to_task = plan[i]

        to_series = cond[f"N_{to_task}"]["mean"]
        from_series = cond[f"N_{from_task}"]["mean"]

        if b >= len(to_series) or b - 1 >= len(from_series):
            continue

        events.append(
            {
                "phase_idx": i,
                "epoch": b + 1,
                "from_task": from_task,
                "to_task": to_task,
                "to_task_start_n": float(to_series[b]),
                "from_task_end_n": float(from_series[b - 1]),
            }
        )

    return events


def _active_task_per_epoch(cond, T):
    active = np.array([None] * T, dtype=object)
    for task, start, L in zip(cond["plan_executed"], cond["phase_boundaries"], cond["phase_lengths"]):
        if start >= T:
            continue
        end = min(start + L, T)
        active[start:end] = task
    return active


def _shade_consensus_phases(ax, conds, T):
    if len(conds) == 0 or T <= 0:
        return

    active_all = [_active_task_per_epoch(c, T) for c in conds]
    consensus = np.array([None] * T, dtype=object)

    for i in range(T):
        task0 = active_all[0][i]
        if task0 is None:
            continue
        same = True
        for j in range(1, len(active_all)):
            if active_all[j][i] != task0:
                same = False
                break
        if same:
            consensus[i] = task0

    i = 0
    while i < T:
        task = consensus[i]
        if task is None:
            i += 1
            continue
        j = i + 1
        while j < T and consensus[j] == task:
            j += 1
        color = PHASE_COLORS.get(task, "#EEEEEE")
        ax.axvspan(i + 1, j, color=color, alpha=0.15, zorder=0)
        i = j


def _draw_switch_markers(ax_list, events, color="gray", alpha=0.5, lw=0.8, ls="--"):
    for ev in events:
        x = ev["epoch"]
        for ax in ax_list:
            ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_task_switch_N_only(
    path_no_cb,
    path_cb=None,
    path_cb_2=None,
    affix_cb=None,
    affix_cb_2=None,
    dont_merge=False,
    figsize=None,
    save_path=None,
    dpi=300,
    show_sem=True,
    shade_auc=False,
    annotate_switches=True,
    phase_background="none",   # "none", "reference", "consensus"
    xlim=None,
):
    cond_a = _aggregate_condition(path_no_cb, dont_merge=dont_merge)
    cond_b = _aggregate_condition(path_cb, dont_merge=dont_merge) if path_cb else None
    cond_c = _aggregate_condition(path_cb_2, dont_merge=dont_merge) if path_cb_2 else None

    conds_present = [c for c in (cond_a, cond_b, cond_c) if c is not None]

    tasks = cond_a["task_names"]
    n_tasks = len(tasks)
    figsize = figsize or (4.2 * n_tasks, 3.8)

    label_a = f"RNN-only (n={cond_a['n_runs']})" if cond_b is not None else f"model (n={cond_a['n_runs']})"
    label_b = f"{affix_cb if affix_cb is not None else 'CB'} (n={cond_b['n_runs']})" if cond_b is not None else None
    label_c = f"{affix_cb_2 if affix_cb_2 is not None else 'CB 2'} (n={cond_c['n_runs']})" if cond_c is not None else None

    max_epoch = max(
        len(cond_a["epoch"]),
        len(cond_b["epoch"]) if cond_b is not None else 0,
        len(cond_c["epoch"]) if cond_c is not None else 0,
    )

    fig, axes = plt.subplots(1, n_tasks, figsize=figsize, sharex=False, sharey=False)
    if n_tasks == 1:
        axes = np.array([axes])

    switch_events_a = _infer_switch_events(cond_a)
    switch_events_b = _infer_switch_events(cond_b) if cond_b is not None else []
    switch_events_c = _infer_switch_events(cond_c) if cond_c is not None else []

    for col, task in enumerate(tasks):
        ax = axes[col]

        # optional background shading
        if phase_background == "reference":
            _shade_phases(
                ax,
                cond_a["phase_boundaries"],
                cond_a["phase_lengths"],
                cond_a["plan_executed"],
            )
        elif phase_background == "consensus":
            T_common = min(len(c["epoch"]) for c in conds_present)
            _shade_consensus_phases(ax, conds_present, T_common)

        n_max_candidates = []

        # ----- condition A -----
        ep_a = cond_a["epoch"]
        n_a = cond_a[f"N_{task}"]["mean"]
        sem_a = cond_a[f"N_{task}"]["sem"]

        ax.step(ep_a, n_a, where="post", lw=2.2, color=BLUE, label=label_a, zorder=3)
        # if show_sem:
            # ax.fill_between(ep_a, n_a - sem_a, n_a + sem_a, step="post", color=BLUE, alpha=0.15, zorder=2)
        if shade_auc:
            ax.fill_between(ep_a, 0, n_a, step="post", color=BLUE, alpha=0.2, zorder=1)
        n_max_candidates.append(np.nanmax(n_a))

        # ----- condition B -----
        if cond_b is not None:
            ep_b = cond_b["epoch"]
            n_b = cond_b[f"N_{task}"]["mean"]
            sem_b = cond_b[f"N_{task}"]["sem"]

            ax.step(ep_b, n_b, where="post", lw=2.2, color=PINK, label=label_b, zorder=3)
            # if show_sem:
                # ax.fill_between(ep_b, n_b - sem_b, n_b + sem_b, step="post", color=PINK, alpha=0.15, zorder=2)
            if shade_auc:
                ax.fill_between(ep_b, 0, n_b, step="post", color=PINK, alpha=0.2, zorder=1)
            n_max_candidates.append(np.nanmax(n_b))

        # ----- condition C -----
        if cond_c is not None:
            ep_c = cond_c["epoch"]
            n_c = cond_c[f"N_{task}"]["mean"]
            sem_c = cond_c[f"N_{task}"]["sem"]

            ax.step(ep_c, n_c, where="post", lw=2.2, color=GREEN, label=label_c, zorder=3)
            # if show_sem:
                # ax.fill_between(ep_c, n_c - sem_c, n_c + sem_c, step="post", color=GREEN, alpha=0.15, zorder=2)
            if shade_auc:
                ax.fill_between(ep_c, 0, n_c, step="post", color=GREEN, alpha=0.2, zorder=1)
            n_max_candidates.append(np.nanmax(n_c))

        # optional switch annotations
        if annotate_switches:
            ymax = max(n_max_candidates) + 1
            for ev in switch_events_a:
                if task == ev["to_task"]:
                    ax.axvline(ev["epoch"], color=BLUE, lw=0.8, ls="--", alpha=0.5)
            if cond_b is not None:
                for ev in switch_events_b:
                    if task == ev["to_task"]:
                        ax.axvline(ev["epoch"], color=PINK, lw=0.8, ls="--", alpha=0.5)
            if cond_c is not None:
                for ev in switch_events_c:
                    if task == ev["to_task"]:
                        ax.axvline(ev["epoch"], color=GREEN, lw=0.8, ls="--", alpha=0.5)

        ax.set_xlim(1, xlim if xlim is not None else max_epoch)
        ax.set_ylim(0, max(n_max_candidates) + 2)

        # ax.axhline(5, color="gray", lw=0.8, ls=":", alpha=0.7, zorder=0)
        # ax.axhline(10, color="gray", lw=0.8, ls=":", alpha=0.7, zorder=0)

        ax.set_xlabel("Global epoch",fontsize=20)
        ax.set_ylabel("Task difficulty (N)" if col == 0 else "",fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        # capitalise task name for title - DMS full capital, Parity first letter
        if task.lower() == "dms":
            display_task = "DMS"
        else:
            display_task = task.capitalize()
        ax.set_title(display_task, fontweight="normal", fontsize=16)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
        ax.spines[["top", "right"]].set_visible(False)

        # if col == 0:
        #     ax.legend(frameon=False, fontsize=12, loc="lower right")

    fig.tight_layout()

    if save_path:
        # Force white background into the SVG geometry, not just metadata
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        for ax in np.ravel(axes):
            ax.set_facecolor("white")
            ax.patch.set_alpha(1.0)
        
        fig.savefig(
            save_path,
            bbox_inches="tight",
            format="svg",
            facecolor=fig.get_facecolor(),  # pull from the patch you just set
            edgecolor="none",
            transparent=False,
        )
    else:
        plt.show()

    return fig, axes


def plot_task_switch_training_loss(
    path_no_cb,
    path_cb=None,
    path_cb_2=None,
    smooth=0,
    dont_merge=False,
    figsize=(9.2, 4.0),
    save_path=None,
    dpi=150,
    show_sem=True,
    annotate_switches=True,
    phase_background="auto",  # "auto", "none", "reference", "consensus"
    switch_markers="auto",    # "auto", "none", "reference", "all"
):
    cond_a = _aggregate_condition(path_no_cb, dont_merge=dont_merge)
    cond_b = _aggregate_condition(path_cb, dont_merge=dont_merge) if path_cb else None
    cond_c = _aggregate_condition(path_cb_2, dont_merge=dont_merge) if path_cb_2 else None

    conds_present = [c for c in (cond_a, cond_b, cond_c) if c is not None]

    if phase_background == "auto":
        phase_background = "reference" if len(conds_present) == 1 else "none"
    if switch_markers == "auto":
        switch_markers = "reference" if len(conds_present) == 1 else "all"

    label_a = f"RNN-only (n={cond_a['n_runs']})" if cond_b is not None else f"model (n={cond_a['n_runs']})"
    label_b = f"CB (n={cond_b['n_runs']})" if cond_b is not None else None
    label_c = f"CB 2 (n={cond_c['n_runs']})" if cond_c is not None else None

    max_epoch = max(
        len(cond_a["epoch"]),
        len(cond_b["epoch"]) if cond_b is not None else 0,
        len(cond_c["epoch"]) if cond_c is not None else 0,
    )

    switch_events_a = _infer_switch_events(cond_a)
    switch_events_b = _infer_switch_events(cond_b) if cond_b is not None else []
    switch_events_c = _infer_switch_events(cond_c) if cond_c is not None else []

    fig, ax = plt.subplots(figsize=figsize)

    if phase_background == "reference":
        _shade_phases(
            ax,
            cond_a["phase_boundaries"],
            cond_a["phase_lengths"],
            cond_a["plan_executed"],
        )
    elif phase_background == "consensus":
        T_common = min(len(c["epoch"]) for c in conds_present)
        _shade_consensus_phases(ax, conds_present, T_common)

    ep_a = cond_a["epoch"]
    loss_a = _smooth(cond_a["loss"]["mean"], smooth)
    sem_a = cond_a["loss"]["sem"]
    ax.plot(ep_a, loss_a, color=BLUE, lw=1.8, label=label_a)
    if show_sem:
        ax.fill_between(ep_a, loss_a - sem_a, loss_a + sem_a, color=BLUE, alpha=0.18)

    if cond_b is not None:
        ep_b = cond_b["epoch"]
        loss_b = _smooth(cond_b["loss"]["mean"], smooth)
        sem_b = cond_b["loss"]["sem"]
        ax.plot(ep_b, loss_b, color=PINK, lw=1.8, label=label_b)
        if show_sem:
            ax.fill_between(ep_b, loss_b - sem_b, loss_b + sem_b, color=PINK, alpha=0.18)

    if cond_c is not None:
        ep_c = cond_c["epoch"]
        loss_c = _smooth(cond_c["loss"]["mean"], smooth)
        sem_c = cond_c["loss"]["sem"]
        ax.plot(ep_c, loss_c, color=GREEN, lw=1.8, label=label_c)
        if show_sem:
            ax.fill_between(ep_c, loss_c - sem_c, loss_c + sem_c, color=GREEN, alpha=0.18)

    if switch_markers == "reference":
        _draw_switch_markers((ax,), switch_events_a, color="gray", alpha=0.55, lw=0.9)
    elif switch_markers == "all":
        _draw_switch_markers((ax,), switch_events_a, color=BLUE, alpha=0.35, lw=0.9)
        if cond_b is not None:
            _draw_switch_markers((ax,), switch_events_b, color=PINK, alpha=0.35, lw=0.9)
        if cond_c is not None:
            _draw_switch_markers((ax,), switch_events_c, color=GREEN, alpha=0.35, lw=0.9)

    finite_vals = loss_a[np.isfinite(loss_a)]
    y_top = np.max(finite_vals) if finite_vals.size else 1.0

    if annotate_switches:
        for ev in switch_events_a:
            txt = f"{ev['from_task']}->{ev['to_task']} @N{int(round(ev['to_task_start_n']))}"
            ax.text(ev["epoch"] + 0.3, y_top, txt, rotation=90, va="top", ha="left", fontsize=7, color="#555555")

    ax.set_xlim(1, max_epoch)
    ax.set_xlabel("global epoch")
    ax.set_ylabel("training loss")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig, ax
import numpy as np

def make_task_active_mask(cond, task, total_len=None):
    """
    Returns a boolean array of length total_len indicating when `task` is active.
    Assumes:
      - cond["phase_lengths"]
      - cond["plan_executed"]
    where each phase corresponds to one active task.
    """
    phase_lengths = cond["phase_lengths"]
    plan = cond["plan_executed"]

    if total_len is None:
        total_len = len(cond["epoch"])

    mask = np.zeros(total_len, dtype=bool)

    start = 0
    for phase_task, phase_len in zip(plan, phase_lengths):
        end = min(start + phase_len, total_len)
        if phase_task == task:
            mask[start:end] = True
        start = end
        if start >= total_len:
            break

    return mask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def make_task_active_mask(cond, task, total_len=None):
    """
    Boolean mask over epochs: True when `task` is the active task.
    Assumes cond has:
      - "phase_lengths"
      - "plan_executed"
      - "epoch"
    """
    phase_lengths = cond["phase_lengths"]
    plan = cond["plan_executed"]

    if total_len is None:
        total_len = len(cond["epoch"])

    mask = np.zeros(total_len, dtype=bool)

    start = 0
    for phase_task, phase_len in zip(plan, phase_lengths):
        end = min(start + phase_len, total_len)
        if phase_task == task:
            mask[start:end] = True
        start = end
        if start >= total_len:
            break

    return mask


def plot_with_active_inactive_style(
    ax,
    x,
    y,
    active_mask,
    color,
    label=None,
    lw_active=2.4,
    lw_inactive=2.0,
    alpha_active=1.0,
    alpha_inactive=0.25,
    ls_active="-",
    ls_inactive="--",
    where="post",
):
    y_active = np.where(active_mask, y, np.nan)
    y_inactive = np.where(~active_mask, y, np.nan)

    ax.step(
        x, y_inactive,
        where=where,
        color=color,
        lw=lw_inactive,
        ls=ls_inactive,
        alpha=alpha_inactive,
        label=None,
        zorder=2,
    )

    ax.step(
        x, y_active,
        where=where,
        color=color,
        lw=lw_active,
        ls=ls_active,
        alpha=alpha_active,
        label=label,
        zorder=3,
    )


def fill_sem_active_inactive(
    ax,
    x,
    y,
    sem,
    active_mask,
    color,
    alpha_active=0.14,
    alpha_inactive=0.05,
    where="post",
):
    lo = y - sem
    hi = y + sem

    lo_active = np.where(active_mask, lo, np.nan)
    hi_active = np.where(active_mask, hi, np.nan)

    lo_inactive = np.where(~active_mask, lo, np.nan)
    hi_inactive = np.where(~active_mask, hi, np.nan)

    ax.fill_between(
        x, lo_inactive, hi_inactive,
        step=where,
        color=color,
        alpha=alpha_inactive,
        zorder=1,
    )
    ax.fill_between(
        x, lo_active, hi_active,
        step=where,
        color=color,
        alpha=alpha_active,
        zorder=1,
    )


def plot_task_switch_results_shaded_active(
    path_no_cb,
    path_cb=None,
    path_cb_2=None,
    affix_cb=None,
    affix_cb_2=None,
    dont_merge=False,
    figsize=None,
    save_path=None,
    dpi=150,
    show_sem=True,
    annotate_switches=False,
    phase_background="none",   # "none", "reference", "consensus"
    xlim=None,
):
    """
    Plot only N progression for task-switching results.

    Active task periods are shown as solid/opaque lines.
    Inactive task periods are shown as dashed/faint lines.

    Assumes _aggregate_condition, _infer_switch_events, _shade_phases,
    _shade_consensus_phases, and colors BLUE/PINK/GREEN already exist.
    """
    cond_a = _aggregate_condition(path_no_cb, dont_merge=dont_merge)
    cond_b = _aggregate_condition(path_cb, dont_merge=dont_merge) if path_cb else None
    cond_c = _aggregate_condition(path_cb_2, dont_merge=dont_merge) if path_cb_2 else None

    conds_present = [c for c in (cond_a, cond_b, cond_c) if c is not None]

    tasks = cond_a["task_names"]
    n_tasks = len(tasks)
    figsize = figsize or (4.2 * n_tasks, 3.8)

    label_a = f"RNN-only (n={cond_a['n_runs']})" if cond_b is not None else f"model (n={cond_a['n_runs']})"
    label_b = f"{affix_cb if affix_cb is not None else 'CB'} (n={cond_b['n_runs']})" if cond_b is not None else None
    label_c = f"{affix_cb_2 if affix_cb_2 is not None else 'CB 2'} (n={cond_c['n_runs']})" if cond_c is not None else None

    max_epoch = max(
        len(cond_a["epoch"]),
        len(cond_b["epoch"]) if cond_b is not None else 0,
        len(cond_c["epoch"]) if cond_c is not None else 0,
    )

    fig, axes = plt.subplots(1, n_tasks, figsize=figsize, sharex=False, sharey=False)
    if n_tasks == 1:
        axes = np.array([axes])

    switch_events_a = _infer_switch_events(cond_a)
    switch_events_b = _infer_switch_events(cond_b) if cond_b is not None else []
    switch_events_c = _infer_switch_events(cond_c) if cond_c is not None else []

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
            T_common = min(len(c["epoch"]) for c in conds_present)
            _shade_consensus_phases(ax, conds_present, T_common)

        n_max_candidates = []

        # ----- condition A -----
        ep_a = cond_a["epoch"]
        n_a = cond_a[f"N_{task}"]["mean"]
        sem_a = cond_a[f"N_{task}"]["sem"]
        active_a = make_task_active_mask(cond_a, task, total_len=len(ep_a))

        plot_with_active_inactive_style(
            ax, ep_a, n_a, active_a,
            color=BLUE,
            label=label_a,
        )
        if show_sem:
            fill_sem_active_inactive(
                ax, ep_a, n_a, sem_a, active_a,
                color=BLUE,
            )
        n_max_candidates.append(np.nanmax(n_a))

        # ----- condition B -----
        if cond_b is not None:
            ep_b = cond_b["epoch"]
            n_b = cond_b[f"N_{task}"]["mean"]
            sem_b = cond_b[f"N_{task}"]["sem"]
            active_b = make_task_active_mask(cond_b, task, total_len=len(ep_b))

            plot_with_active_inactive_style(
                ax, ep_b, n_b, active_b,
                color=PINK,
                label=label_b,
            )
            if show_sem:
                fill_sem_active_inactive(
                    ax, ep_b, n_b, sem_b, active_b,
                    color=PINK,
                )
            n_max_candidates.append(np.nanmax(n_b))

        # ----- condition C -----
        if cond_c is not None:
            ep_c = cond_c["epoch"]
            n_c = cond_c[f"N_{task}"]["mean"]
            sem_c = cond_c[f"N_{task}"]["sem"]
            active_c = make_task_active_mask(cond_c, task, total_len=len(ep_c))

            plot_with_active_inactive_style(
                ax, ep_c, n_c, active_c,
                color=GREEN,
                label=label_c,
            )
            if show_sem:
                fill_sem_active_inactive(
                    ax, ep_c, n_c, sem_c, active_c,
                    color=GREEN,
                )
            n_max_candidates.append(np.nanmax(n_c))

        if annotate_switches:
            ymax = max(n_max_candidates) + 1
            for ev in switch_events_a:
                if task == ev["to_task"]:
                    ax.axvline(ev["epoch"], color=BLUE, lw=0.8, ls="--", alpha=0.25)
            if cond_b is not None:
                for ev in switch_events_b:
                    if task == ev["to_task"]:
                        ax.axvline(ev["epoch"], color=PINK, lw=0.8, ls="--", alpha=0.25)
            if cond_c is not None:
                for ev in switch_events_c:
                    if task == ev["to_task"]:
                        ax.axvline(ev["epoch"], color=GREEN, lw=0.8, ls="--", alpha=0.25)

        ax.set_xlim(1, xlim if xlim is not None else max_epoch)
        ax.set_ylim(0, max(n_max_candidates) + 2)
        ax.set_title(task, fontsize=11, fontweight="normal")
        ax.set_xlabel("Global epoch")
        ax.set_ylabel("Task difficulty (N)" if col == 0 else "")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
        ax.spines[["top", "right"]].set_visible(False)

        if col == 0:
            ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig, axes

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def summarize_single_switch_run(stats, switch_start_window=10):
    """
    Summarize one stats_continual dict into one row per phase per task.

    Expected keys in stats:
      - plan_executed
      - phase_boundaries
      - task_names
      - acc_{task}_phase{i}
      - N_{task}_phase{i}
      - loss_phase{i}
    """
    rows = []

    plan_executed = stats["plan_executed"]
    phase_boundaries = stats["phase_boundaries"]

    n_phases = len(plan_executed)

    for phase_idx, active_task in enumerate(plan_executed):
        acc_key = f"acc_{active_task}_phase{phase_idx}"
        N_key = f"N_{active_task}_phase{phase_idx}"
        loss_key = f"loss_phase{phase_idx}"

        if acc_key not in stats:
            continue

        acc = np.asarray(stats[acc_key], dtype=float)
        Nprog = np.asarray(stats[N_key], dtype=float) if N_key in stats else np.full(len(acc), np.nan)
        loss = np.asarray(stats[loss_key], dtype=float) if loss_key in stats else np.full(len(acc), np.nan)

        if len(acc) == 0:
            continue

        x = np.arange(len(acc))

        row = {
            "phase_idx": phase_idx,
            "task": active_task,
            "phase_start_epoch": phase_boundaries[phase_idx],
            "phase_len": len(acc),
            "acc_start": float(acc[0]),
            "acc_end": float(acc[-1]),
            "acc_mean": float(np.nanmean(acc)),
            "acc_auc": float(np.trapz(acc, x)),
            "N_start": float(Nprog[0]) if len(Nprog) else np.nan,
            "N_end": float(Nprog[-1]) if len(Nprog) else np.nan,
            "N_mean": float(np.nanmean(Nprog)),
            "N_auc": float(np.trapz(Nprog, x)),
            "loss_start": float(loss[0]) if len(loss) else np.nan,
            "loss_end": float(loss[-1]) if len(loss) else np.nan,
            "loss_mean": float(np.nanmean(loss)),
            "switch_start_acc": float(np.nanmean(acc[:min(switch_start_window, len(acc))])),
            "switch_start_N": float(np.nanmean(Nprog[:min(switch_start_window, len(Nprog))])) if len(Nprog) else np.nan,
        }

        rows.append(row)

    # add total training length if possible
    if len(rows) > 0:
        total_epochs = sum(r["phase_len"] for r in rows)
        for r in rows:
            r["total_epochs"] = total_epochs

    return pd.DataFrame(rows)

def summarize_single_switch_condition(path_pattern, switch_start_window=10):
    run_paths = _expand_repeat_paths(path_pattern)
    dfs = []

    for rp in run_paths:
        stats = _load(str(rp))
        df = summarize_single_switch_run(stats, switch_start_window=switch_start_window)
        if len(df) == 0:
            continue
        df["run_path"] = str(rp)
        df["run_id"] = os.path.basename(str(rp).rstrip("/"))
        dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
def summarize_multiple_single_switch_conditions(
    path_no_cb,
    path_cb=None,
    path_cb_2=None,
    affix_cb=None,
    affix_cb_2=None,
    switch_start_window=10,
):
    dfs = []

    df_a = summarize_single_switch_condition(path_no_cb, switch_start_window=switch_start_window)
    df_a["condition"] = "RNN-only" if path_cb is not None else "model"
    dfs.append(df_a)

    if path_cb is not None:
        df_b = summarize_single_switch_condition(path_cb, switch_start_window=switch_start_window)
        df_b["condition"] = affix_cb if affix_cb is not None else "CB"
        dfs.append(df_b)

    if path_cb_2 is not None:
        df_c = summarize_single_switch_condition(path_cb_2, switch_start_window=switch_start_window)
        df_c["condition"] = affix_cb_2 if affix_cb_2 is not None else "CB 2"
        dfs.append(df_c)

    return pd.concat(dfs, ignore_index=True)

def plot_single_switch_phase_lengths(df_summary, figsize=None, save_path=None):
    phases = sorted(df_summary["phase_idx"].unique())
    conds = list(df_summary["condition"].unique())

    color_map = {
        "RNN-only": "cornflowerblue",
        "CB": "salmon",
        "CB 2": "orange",
    }
    box_colors = [color_map.get(c, "lightgray") for c in conds]

    figsize = figsize or (4.0 * len(phases), 4.0)
    fig, axes = plt.subplots(1, len(phases), figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, phase in zip(axes, phases):
        sub = df_summary[df_summary["phase_idx"] == phase]
        task_name = sub["task"].iloc[0]

        data = [sub[sub["condition"] == c]["phase_len"].dropna().values for c in conds]

        bp = ax.boxplot(
            data,
            labels=conds,
            patch_artist=True,
        )

        for i, box in enumerate(bp["boxes"]):
            c = box_colors[i]
            box.set_facecolor(c)
            box.set_edgecolor(c)
            box.set_alpha(0.6)
            box.set_linewidth(1.5)

        for i, med in enumerate(bp["medians"]):
            med.set_color(box_colors[i])
            med.set_linewidth(1.8)

        # whiskers and caps come in pairs per box
        for i, whisk in enumerate(bp["whiskers"]):
            c = box_colors[i // 2]
            whisk.set_color(c)
            whisk.set_linewidth(1.2)

        for i, cap in enumerate(bp["caps"]):
            c = box_colors[i // 2]
            cap.set_color(c)
            cap.set_linewidth(1.2)

        for i, flier in enumerate(bp["fliers"]):
            c = box_colors[i]
            flier.set_markeredgecolor(c)
            flier.set_markerfacecolor(c)
            flier.set_alpha(0.5)
            flier.set_markersize(4)

        ax.set_title(f"Phase {phase}: {task_name}")
        ax.set_ylabel("Epochs in phase" if phase == phases[0] else "")
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", format="svg")
    plt.show()
    return fig, axes

def plot_single_switch_phase_auc(df_summary, value_col="acc_auc", figsize=None, save_path=None):
    phases = sorted(df_summary["phase_idx"].unique())
    conds = list(df_summary["condition"].unique())

    figsize = figsize or (4.0 * len(phases), 4.0)
    fig, axes = plt.subplots(1, len(phases), figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, phase in zip(axes, phases):
        sub = df_summary[df_summary["phase_idx"] == phase]
        task_name = sub["task"].iloc[0]

        data = [sub[sub["condition"] == c][value_col].dropna().values for c in conds]
        ax.boxplot(data, labels=conds, patch_artist=False)

        ax.set_title(f"Phase {phase}: {task_name}")
        ax.set_ylabel(value_col.replace("_", " ") if phase == phases[0] else "")
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", format="svg")
    plt.show()
    return fig, axes

