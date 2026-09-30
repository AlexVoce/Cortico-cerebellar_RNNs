"""Figure 6 panels, plotted from precomputed CSVs.

To regenerate the CSVs (slow), run from the repo root:
    python -m analysis.run_dms_memory_propagation
    python -m analysis.run_dms_memory_dynamics
    python -m analysis.dms_memory_dynamics_stats
    python -m analysis.run_parity_cb_encoding
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "Figures/ICLR"
DMS_CSV = ROOT / "results/mechanistic_analysis/dms_memory_propagation/dms_memory_delay_averages.csv"
PARITY_CAUSAL_CSV = ROOT / "results/mechanistic_analysis/parity_cb_encoding/parity_cb_swap_scores.csv"

STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.spines.top": False,
    "axes.spines.right": False,
}
LABEL_FS, TICK_FS, LEGEND_FS, PANEL_FS = 8, 7, 7, 9

RNN_COLOR, CB_COLOR, DET_COLOR = "cornflowerblue", "salmon", "#7f7f7f"
DMS_VARIANT = "primary_eval_seed_12345"

PARITY_LABELS = ["Correct", "No CB", "Same", "Opposite"]
PARITY_CONDITIONS = ["A_full", "B_zero", "H_same_next_diff_cond", "I_diff_next_diff_cond"]
PARITY_COLORS = ["black", "#888888", "#756BB1", "#C51B7D"]


def _save(fig, save_dir, name, exts, dpi):
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        path = save_dir / f"{name}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", dpi=dpi if ext == "png" else None)
        print(f"saved {path}")


def _panel_label(ax, label, x=-0.30, y=1.10):
    if label:
        ax.text(x, y, rf"\textbf{{{label}}}", transform=ax.transAxes,
                fontsize=PANEL_FS, va="top", ha="left")


# 6a-b: DMS

def _dms_seed_matrix(df, condition):
    w = df[df.condition == condition].pivot(index="seed", columns="N", values="Q")
    return w.columns.values, w.values


def _dms_draw(ax, df, condition, color, label, ls="-"):
    Ns, x = _dms_seed_matrix(df, condition)
    for row in x:
        ax.plot(Ns, row, color=color, alpha=0.18, linewidth=0.5, ls=ls, zorder=1)
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    ax.fill_between(Ns, mean - sd, mean + sd, color=color, alpha=0.25, linewidth=0, zorder=2)
    ax.plot(Ns, mean, color=color, linewidth=1.4, ls=ls, label=label, zorder=3)


def _dms_style(ax, ylabel, panel):
    ax.set_xlabel(r"DMS curriculum level, N", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xlim(4.5, 30.5)
    _panel_label(ax, panel)


def plot_dms_memory_dynamics(csv_path=DMS_CSV, variant=DMS_VARIANT, figsize=(6.2, 1.1),
                             panel_labels=("a", "b"), save_dir=None,
                             name="figure_dms_memory_dynamics_final"):
    """Figure 6a-b. Returns (fig, (axA, axB))."""
    df = pd.read_csv(csv_path)
    df = df[df.variant == variant]

    with mpl.rc_context(STYLE):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize)

        _dms_draw(axA, df, "RNN-only", RNN_COLOR, "RNN only")
        _dms_draw(axA, df, "CB-RNN (full)", CB_COLOR, "CB RNN (full)")
        _dms_draw(axA, df, "CB-RNN (CB detached)", DET_COLOR, "CB RNN (CB detached)", ls="--")
        axA.axhline(1, color="black", linewidth=0.6, ls=":", zorder=0)
        _dms_style(axA, "Memory-axis\npropagation", panel_labels[0])
        axA.legend(fontsize=LEGEND_FS - 0.5, frameon=False, loc="lower right", bbox_to_anchor=(1.05, -0.02),
                   handlelength=1.3, handletextpad=0.4, labelspacing=0.25)

        _dms_draw(axB, df, "CB contribution", CB_COLOR, None)
        axB.axhline(0, color="black", linewidth=0.6, ls=":", zorder=0)
        _dms_style(axB, "CB contribution\nto propagation", panel_labels[1])

        fig.tight_layout(w_pad=2.0)
        _save(fig, save_dir, name, ("svg", "pdf", "png"), dpi=400)
    return fig, (axA, axB)


# 6c-d: parity causal decomposition

def load_parity_causal(csv_path=PARITY_CAUSAL_CSV, normalise=True, exclude_Ns=()):
    """Per-(seed, N) normalised scores (+1 = correct parity centroid, -1 = wrong)."""
    df = pd.read_csv(csv_path)
    df = df[~df["N"].isin(list(exclude_Ns))]
    prefix = "normscore_" if normalise else "score_"
    cols = {prefix + c: label for c, label in zip(PARITY_CONDITIONS, PARITY_LABELS)}
    return df[["seed", "N", *cols]].rename(columns=cols)


def parity_causal_seed_summary(csv_path=PARITY_CAUSAL_CSV, normalise=True, exclude_Ns=()):
    """Seed-level mean/sd per condition (each seed averaged over N first)."""
    per_seed = load_parity_causal(csv_path, normalise, exclude_Ns).groupby("seed")[PARITY_LABELS].mean()
    return pd.DataFrame({"mean": per_seed.mean(), "sd": per_seed.std(), "n_seeds": len(per_seed)})


def plot_parity_causal_bar(csv_path=PARITY_CAUSAL_CSV, normalise=True, exclude_Ns=(),
                           figsize=(3.1, 1.1), panel_label=None, save_dir=None,
                           name="causal_decomposition_mini"):
    """Figure 6c. Returns (fig, ax)."""
    summary = parity_causal_seed_summary(csv_path, normalise, exclude_Ns)
    print(summary.round(4).to_string())

    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(PARITY_LABELS))
        ax.bar(x - 0.05, summary["mean"].values, yerr=summary["sd"].values, color=PARITY_COLORS,
               capsize=3, width=0.6, error_kw={"linewidth": 1})
        ax.axhline(0, color="black", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(PARITY_LABELS, fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_ylabel("Transition\n score", fontsize=LABEL_FS)
        ax.margins(x=0.1)
        if normalise:
            ax.set_ylim(-1.01, 1.01)
            ax.set_yticks([-1, 0, 1])
        _panel_label(ax, panel_label)

        fig.tight_layout()
        _save(fig, save_dir, name, ("svg", "png"), dpi=300)
    return fig, ax


def plot_parity_causal_by_N(csv_path=PARITY_CAUSAL_CSV, normalise=True, show_seeds=True,
                            figsize=(3.1, 1.25), panel_label=None, save_dir=None,
                            name="causal_decomposition_by_N"):
    """Figure 6d. Returns (fig, ax)."""
    df = load_parity_causal(csv_path, normalise)
    Ns = np.sort(df["N"].unique())

    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        for label, color in zip(PARITY_LABELS, PARITY_COLORS):
            if show_seeds:
                for _, d in df.groupby("seed"):
                    d = d.sort_values("N")
                    ax.plot(d["N"], d[label], color=color, alpha=0.15, linewidth=0.4)
            g = df.groupby("N")[label]
            mean, sd = g.mean().loc[Ns].values, g.std().loc[Ns].values
            ax.plot(Ns, mean, color=color, marker="o", markersize=1, linewidth=0.9, label=label)
            ax.fill_between(Ns, mean - sd, mean + sd, color=color, alpha=0.25, linewidth=0)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(Ns[::5])
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_xlabel("N", fontsize=LABEL_FS)
        _panel_label(ax, panel_label)

        fig.tight_layout()
        _save(fig, save_dir, name, ("svg", "png"), dpi=300)
    return fig, ax


def main():
    for plot in (plot_dms_memory_dynamics, plot_parity_causal_bar, plot_parity_causal_by_N):
        fig, _ = plot(save_dir=FIG_DIR)
        plt.close(fig)


if __name__ == "__main__":
    main()
