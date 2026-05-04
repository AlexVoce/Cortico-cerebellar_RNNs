import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Loading
# =============================================================================

def load_timescale_pkls(timescale_dir):
    """
    Load timescale pickle files named timescales_N*.pkl from one run.

    Expected file contents:
        obj["N"]
        obj["shared_tau"]
        obj["modules"][module]["taus_net"]
        obj["modules"][module]["selected_models"]
        obj["modules"][module]["ac_pop"]
    """
    timescale_dir = Path(timescale_dir)

    files = sorted(
        timescale_dir.glob("timescales_N*.pkl"),
        key=lambda p: int(p.stem.split("N")[-1])
    )

    if not files:
        raise FileNotFoundError(
            f"No timescales_N*.pkl files found in:\n{timescale_dir}"
        )

    data = {}

    for f in files:
        with open(f, "rb") as fh:
            obj = pickle.load(fh)

        if "N" not in obj:
            raise KeyError(f"{f} does not contain key 'N'")

        data[int(obj["N"])] = obj

    return data


# =============================================================================
# Helpers
# =============================================================================

def estimate_e_folding_lag(ac, normalize=True):
    """
    Estimate timescale as first lag where autocorrelation <= 1/e.

    Returns np.nan if the curve never crosses 1/e.
    """
    ac = np.asarray(ac, dtype=float)

    if len(ac) == 0:
        return np.nan

    if not np.isfinite(ac[0]) or ac[0] == 0:
        return np.nan

    if normalize:
        ac = ac / ac[0]

    threshold = 1 / np.e
    below = np.where(ac <= threshold)[0]

    if len(below) == 0:
        return np.nan

    return float(below[0])


def _safe_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) <= 1:
        return np.nan

    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _concat_1d_arrays(series):
    arrs = []

    for x in series:
        if x is None:
            continue

        arr = np.asarray(x, dtype=float).ravel()

        if len(arr) > 0:
            arrs.append(arr)

    if len(arrs) == 0:
        return np.array([], dtype=float)

    return np.concatenate(arrs, axis=0)


def _stack_equal_length_arrays(series, allow_trim=True):
    """
    Stack arrays across runs.

    If allow_trim=True and lengths differ, all arrays are trimmed to the
    minimum length. This is usually safer for autocorrelation curves.
    """
    arrs = []

    for x in series:
        if x is None:
            continue

        arr = np.asarray(x, dtype=float).ravel()

        if len(arr) > 0:
            arrs.append(arr)

    if len(arrs) == 0:
        return None

    lengths = [len(a) for a in arrs]

    if len(set(lengths)) != 1:
        if not allow_trim:
            raise ValueError(f"Arrays have different lengths: {lengths}")

        min_len = min(lengths)
        arrs = [a[:min_len] for a in arrs]

    return np.stack(arrs, axis=0)  # [n_runs, T]


# =============================================================================
# Per-run summary
# =============================================================================

def summarize_timescale_run(timescale_data):
    """
    Summarise timescale results for one run.

    Keeps both scalar summaries and raw arrays:
        - taus
        - selected_models
        - ac_pop

    These raw arrays are needed later for pooled/averaged plots.
    """
    rows = []

    for N, obj in sorted(timescale_data.items()):
        shared_tau = obj.get("shared_tau", np.nan)

        modules = obj.get("modules", {})
        if not isinstance(modules, dict):
            raise TypeError(f"obj['modules'] for N={N} is not a dictionary.")

        for module, res in modules.items():
            if res is None:
                continue

            taus = np.asarray(res.get("taus_net", []), dtype=float)
            selected = np.asarray(res.get("selected_models", []), dtype=float)

            ac_pop = res.get("ac_pop", None)
            if ac_pop is not None:
                ac_pop = np.asarray(ac_pop, dtype=float)
                pop_ac0 = ac_pop[0] if len(ac_pop) > 0 else np.nan
                pop_e_folding_lag = estimate_e_folding_lag(ac_pop, normalize=True)
            else:
                pop_ac0 = np.nan
                pop_e_folding_lag = np.nan

            valid = np.isfinite(taus)
            taus_valid = taus[valid]

            if len(taus_valid) > 0:
                tau_mean = np.nanmean(taus_valid)
                tau_median = np.nanmedian(taus_valid)
                tau_std = np.nanstd(taus_valid)
                tau_q25 = np.nanpercentile(taus_valid, 25)
                tau_q75 = np.nanpercentile(taus_valid, 75)
            else:
                tau_mean = np.nan
                tau_median = np.nan
                tau_std = np.nan
                tau_q25 = np.nan
                tau_q75 = np.nan

            rows.append({
                "N": int(N),
                "module": module,
                "shared_tau": shared_tau,

                # raw arrays retained for downstream pooled/curve analyses
                "taus": taus,
                "selected_models": selected,
                "ac_pop": ac_pop,

                # scalar summaries
                "n_units": len(taus),
                "frac_valid_tau": valid.mean() if len(valid) else np.nan,
                "tau_mean": tau_mean,
                "tau_median": tau_median,
                "tau_std": tau_std,
                "tau_q25": tau_q25,
                "tau_q75": tau_q75,
                "frac_single_exp": np.mean(selected == 1) if len(selected) else np.nan,
                "frac_double_exp": np.mean(selected == 2) if len(selected) else np.nan,
                "pop_ac0": pop_ac0,
                "pop_e_folding_lag": pop_e_folding_lag,
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["module", "N"])
        .reset_index(drop=True)
    )


def summarize_timescale_multiple_runs(run_paths):
    """
    Summarise timescale results across multiple run directories.

    Each run_path should contain:
        run_path/timescales/timescales_N*.pkl
    """
    all_dfs = []

    for run_path in run_paths:
        run_path = Path(run_path)
        timescale_dir = run_path / "timescales"

        data = load_timescale_pkls(timescale_dir)
        df = summarize_timescale_run(data)

        df["run_id"] = run_path.name
        df["run_path"] = str(run_path)

        all_dfs.append(df)

    if len(all_dfs) == 0:
        raise ValueError("No runs were summarised.")

    return pd.concat(all_dfs, ignore_index=True)

def average_across_runs_timescales(timescale_runs_df, allow_trim_ac=True):
    """
    Average timescale summaries across runs.

    Input should be the output of summarize_timescale_multiple_runs().
    """
    required_cols = {
        "N",
        "module",
        "run_id",
        "tau_median",
        "tau_mean",
        "frac_valid_tau",
        "pop_e_folding_lag",
        "taus",
        "ac_pop",
    }

    missing = required_cols.difference(timescale_runs_df.columns)
    if missing:
        raise KeyError(
            f"timescale_runs_df is missing required columns: {sorted(missing)}"
        )

    rows = []

    for (N, module), subdf in timescale_runs_df.groupby(["N", "module"]):
        ac_stack = _stack_equal_length_arrays(
            subdf["ac_pop"],
            allow_trim=allow_trim_ac,
        )

        if ac_stack is not None:
            ac_pop_mean = np.nanmean(ac_stack, axis=0)

            if ac_stack.shape[0] > 1:
                ac_pop_sem = np.nanstd(ac_stack, axis=0, ddof=1) / np.sqrt(ac_stack.shape[0])
            else:
                ac_pop_sem = np.full(ac_stack.shape[1], np.nan)

            pop_e_folding_from_mean_ac = estimate_e_folding_lag(
                ac_pop_mean,
                normalize=True,
            )
        else:
            ac_pop_mean = np.array([], dtype=float)
            ac_pop_sem = np.array([], dtype=float)
            pop_e_folding_from_mean_ac = np.nan

        taus_pooled = _concat_1d_arrays(subdf["taus"])
        taus_pooled_valid = taus_pooled[np.isfinite(taus_pooled)]

        rows.append({
            "N": int(N),
            "module": module,
            "n_runs": subdf["run_id"].nunique(),

            # averaged population AC curve
            "ac_pop_mean": ac_pop_mean,
            "ac_pop_sem": ac_pop_sem,

            # pooled unit-level taus
            "taus_pooled": taus_pooled,
            "taus_pooled_valid": taus_pooled_valid,
            "tau_pooled_median": np.nanmedian(taus_pooled_valid) if len(taus_pooled_valid) else np.nan,
            "tau_pooled_mean": np.nanmean(taus_pooled_valid) if len(taus_pooled_valid) else np.nan,

            # run-level scalar summaries
            "tau_median_mean": subdf["tau_median"].mean(),
            "tau_median_sem": subdf["tau_median"].sem(),
            "tau_mean_mean": subdf["tau_mean"].mean(),
            "tau_mean_sem": subdf["tau_mean"].sem(),
            "frac_valid_tau_mean": subdf["frac_valid_tau"].mean(),
            "frac_valid_tau_sem": subdf["frac_valid_tau"].sem(),

            # e-folding computed per run, then averaged
            "pop_e_folding_lag_mean": subdf["pop_e_folding_lag"].mean(),
            "pop_e_folding_lag_sem": subdf["pop_e_folding_lag"].sem(),

            # e-folding computed from the averaged AC curve
            "pop_e_folding_from_mean_ac": pop_e_folding_from_mean_ac,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["module", "N"])
        .reset_index(drop=True)
    )

def plot_two_task_full_vs_reservoir_e_folding_2x2(
    avg_df_full_task1,
    avg_df_res_task1,
    avg_df_full_task2,
    avg_df_res_task2,
    task1_title="DMS",
    task2_title="Parity",
    modules=("hidden", "gc", "pc", "cb_bias"),
    show_sem=True,
    linewidth_pt=397.48499,
    title="Population AC 1/e crossing lag across N",
    savepath=None,
    fig_height=2,
    ylim_vals=None,
    figsize=None,
):
    full_color_map = {
        "hidden":  "#20A0C9",
        "gc":      "#0553CF",
        "pc":      "#33BBFF",
        "cb_bias": "#5BD7CD",
    }
    res_color_map = {
        "hidden":  "#FF5F4A",
        "gc":      "#D62E24",
        "pc":      "#FF8200",
        "cb_bias": "#FFC100",
    }
    nicer_names = {
        "hidden": "RNN",
        "gc": "GC",
        "pc": "PC",
        "cb_bias": "CB bias",
    }

    preferred_order = ["hidden", "gc", "pc", "cb_bias"]
    modules = [m for m in preferred_order if m in modules] + [m for m in modules if m not in preferred_order]
    inches_per_pt = 1 / 72.27
    fig_width = linewidth_pt * inches_per_pt
    if figsize is None:
        figsize = (fig_width,fig_height)
    
    panel_specs = [
        (task1_title, "Full model", avg_df_full_task1, full_color_map),
        (task1_title, "Reservoir model", avg_df_res_task1, res_color_map),
        (task2_title, "Full model", avg_df_full_task2, full_color_map),
        (task2_title, "Reservoir model", avg_df_res_task2, res_color_map),
    ]

    fig, axs = plt.subplots(2, 2, figsize=figsize, squeeze=False, sharey="row",sharex=False)
    axs = axs.flatten()

    label_fs = 9
    tick_fs = 8
    title_fs = 9
    legend_fs = 8

    for ax, (task_title, panel_title, df_use, color_map) in zip(axs, panel_specs):
        for module in modules:
            sub = df_use[df_use["module"] == module].sort_values("N")
            if len(sub) == 0:
                continue

            mean = sub["pop_e_folding_lag_mean"].values.astype(float)
            Nvals = sub["N"].values.astype(float)
            color = color_map.get(module, "gray")
            label = nicer_names.get(module, module)

            ax.plot(
                Nvals,
                mean,
                marker="o",
                linewidth=1,
                markersize=1.5,
                label=label,
                color=color,
            )

            if show_sem:
                sem = sub["pop_e_folding_lag_sem"].values.astype(float)
                ax.fill_between(
                    Nvals,
                    mean - sem,
                    mean + sem,
                    alpha=0.18,
                    color=color,
                )


        if ax==axs[0] or ax==axs[1]:
            ax.set_title(f"{panel_title}", fontsize=title_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

        if ylim_vals is not None:
            if isinstance(ylim_vals, dict):
                if panel_title in ylim_vals and ylim_vals[panel_title] is not None:
                    ax.set_ylim(ylim_vals[panel_title])
            else:
                ax.set_ylim(ylim_vals)

    axs[2].set_xlabel("N", fontsize=label_fs)
    axs[3].set_xlabel("N", fontsize=label_fs)
    axs[0].set_ylabel("DMS\n"+ r"$\tau_{pop}$", fontsize=label_fs)
    axs[2].set_ylabel("Parity\n"+ r"$\tau_{pop}$", fontsize=label_fs)

    axs[2].legend(fontsize=legend_fs-2, loc="upper left",frameon=False)
    axs[3].legend(fontsize=legend_fs-2, loc="upper left",frameon=False)

    axs[1].tick_params(axis="y", labelleft=False)
    axs[3].tick_params(axis="y", labelleft=False)
    # also remove ticks and spines from middle column for cleaner look
    axs[1].tick_params(axis="y", left=False)
    axs[1].spines["left"].set_visible(False)
    axs[3].tick_params(axis="y", left=False)
    axs[3].spines["left"].set_visible(False)

    # ax 3 major locator on X for 3 ticks at 5,10,15
    axs[3].xaxis.set_major_locator(plt.FixedLocator([5,10,15]))

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, format="svg")
    plt.show()

    return fig,axs
