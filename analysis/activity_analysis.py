import numpy as np
from sklearn.decomposition import PCA
import torch
import json
from analysis.rebuild_model_utils import find_available_Ns,load_state_dict, load_run_config, build_model_from_config_and_state
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

def flatten_activity_from_output(output, key="hidden"):
    """
    Takes output from collect_activity_for_n and flattens [T,B,D] across all batches
    into one big [samples, D] tensor.
    """
    chunks = []
    for d in output["dynamics"]:
        x = d[key]   # [T,B,D]
        if x is None:
            continue
        T, B, D = x.shape
        chunks.append(x.reshape(T * B, D))
    if len(chunks) == 0:
        raise ValueError(f"No activity found for key={key}")
    return torch.cat(chunks, dim=0)   # [total_samples, D]
def run_pca_on_tensor(X: torch.Tensor):
    """
    X: [samples, D]
    """
    X = X.detach().cpu().numpy()
    X = X - X.mean(axis=0, keepdims=True)

    pca = PCA()
    pca.fit(X)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    eigvals = pca.explained_variance_

    d90 = int(np.searchsorted(cum_evr, 0.90) + 1)
    d95 = int(np.searchsorted(cum_evr, 0.95) + 1)
    participation_ratio = (eigvals.sum() ** 2) / np.sum(eigvals ** 2)

    return {
        "pca": pca,
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum_evr,
        "d90": d90,
        "d95": d95,
        "participation_ratio": participation_ratio,
    }
def collect_activity_for_n(
    model,
    batch_fn,
    eval_n,
    batch_size=64,
    n_batches=10,
    head_idx=0,
    device="cpu",
):
    all_dyn = []
    all_inputs = []
    all_labels = []
    all_logits = []
    all_lengths = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            seqs, labs = batch_fn([eval_n], batch_size)
            seqs = seqs.to(device)
            labs = [l.to(device) for l in labs]

            _, out_heads, dyn = model(
                seqs,
                return_timewise=False,
                return_dynamics=True,
            )

            all_inputs.append(seqs.cpu())                    # list of [T,B,D]
            all_lengths.append(seqs.shape[0])
            all_labels.append(labs[-1].cpu())               # [B]
            all_logits.append(out_heads[head_idx].cpu())    # [B,C]
            all_dyn.append({
                k: v.cpu() if v is not None else None
                for k, v in dyn.items()
            })

    return {
        "inputs": all_inputs,                        # list of [T,B,D]
        "lengths": all_lengths,                     # list of T values
        "labels": torch.cat(all_labels, dim=0),     # [total_B]
        "logits": torch.cat(all_logits, dim=0),     # [total_B,C]
        "dynamics": all_dyn,                        # list of dicts, each [T,B,*]
        "eval_n": eval_n,
    }
def collect_and_pca_one_N(
    run_path,
    N,
    batch_fn,
    batch_size=64,
    n_batches=10,
    head_idx=0,
    device="cpu",
    key="hidden",
):
    cfg = load_run_config(run_path)
    sd = load_state_dict(run_path, N)

    model = build_model_from_config_and_state(
        cfg=cfg,
        state_dict=sd,
        device=device,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()

    output = collect_activity_for_n(
        model=model,
        batch_fn=batch_fn,
        eval_n=N,
        batch_size=batch_size,
        n_batches=n_batches,
        head_idx=head_idx,
        device=device,
    )

    X = flatten_activity_from_output(output, key=key)
    pca_res = run_pca_on_tensor(X)

    return {
        "N": N,
        "output": output,
        "X": X,
        **pca_res,
    }


def pca_across_run(
    run_path,
    batch_fn,
    batch_size=64,
    n_batches=10,
    head_idx=0,
    device="cpu",
    key="hidden",
):
    Ns = find_available_Ns(run_path)
    results = []

    for N in Ns:
        # print every 10 Ns to track progress
        if N % 10 == 0:
            print(f"Running PCA for N={N}")
        res = collect_and_pca_one_N(
            run_path=run_path,
            N=N,
            batch_fn=batch_fn,
            batch_size=batch_size,
            n_batches=n_batches,
            head_idx=head_idx,
            device=device,
            key=key,
        )
        results.append(res)

    return results

def summarize_run_pca_for_multiple_keys(
    run_path,
    batch_fn,
    keys=("hidden", "gc", "cb_bias"),
    batch_size=64,
    n_batches=10,
    head_idx=0,
    device="cpu",
    run_id=None,
):
    if run_id is None:
        run_id = os.path.basename(run_path)

    rows = []

    for key in keys:
        pca_results = pca_across_run(
            run_path=run_path,
            batch_fn=batch_fn,
            batch_size=batch_size,
            n_batches=n_batches,
            head_idx=head_idx,
            device=device,
            key=key,
        )

        for r in pca_results:
            rows.append({
                "run_id": run_id,
                "run_path": run_path,
                "population": key,
                "N": r["N"],
                "d90": r["d90"],
                "d95": r["d95"],
                "participation_ratio": r["participation_ratio"],
            })

    return pd.DataFrame(rows)
import os
import pandas as pd

def summarize_multi_runs_pca_for_multi_keys(
    run_paths,
    batch_fn,
    keys=("hidden", "gc", "cb_bias"),
    batch_size=64,
    n_batches=10,
    head_idx=0,
    device="cpu",
):
    dfs = []

    for i, run_path in enumerate(run_paths):
        run_id = os.path.basename(run_path.rstrip("/"))
        print(f"Processing run {i+1}/{len(run_paths)}: {run_id}")

        df_run = summarize_run_pca_for_multiple_keys(
            run_path=run_path,
            batch_fn=batch_fn,
            keys=keys,
            batch_size=batch_size,
            n_batches=n_batches,
            head_idx=head_idx,
            device=device,
            run_id=run_id,
        )
        dfs.append(df_run)

    return pd.concat(dfs, ignore_index=True)

### plotting function

def plot_two_tasks_full_vs_reservoir_pca_mean_2x2(
    df_full_task1,
    df_res_task1,
    df_full_task2,
    df_res_task2,
    task1_title="DMS",
    task2_title="Parity",
    metric="d90",
    errorbar="sem",
    ylim_vals=None,
    figsize=None,
    save_path=None,
    fig_height=1.1,
    linewidth_pt=397.48499,
    full_color_map=None,
    res_color_map=None,
):
    full_color_map = full_color_map or {
        "hidden":  "#20A0C9",
        "gc":      "#0553CF",
        "pc":      "#33BBFF",
        "cb_bias": "#5BD7CD",
    }

    res_color_map = res_color_map or {
        "hidden":  "#FF5F4A",
        "gc":      "#D62E24",
        "pc":      "#FF8200",
        "cb_bias": "#FFC100",
    }

    preferred_order = ["hidden", "gc", "pc", "cb_bias"]
    nicer_names = {
        "hidden": "RNN",
        "gc": "GC",
        "pc": "PC",
        "cb_bias": "CB bias",
    }

    inches_per_pt = 1 / 72.27
    fig_width = linewidth_pt * inches_per_pt
    if figsize is None:
        figsize = (fig_width, fig_height)

    panel_specs = [
        (task1_title, "Full model", df_full_task1, full_color_map),
        (task1_title, "Reservoir model", df_res_task1, res_color_map),
        (task2_title, "Full model", df_full_task2, full_color_map),
        (task2_title, "Reservoir model", df_res_task2, res_color_map),
    ]

    fig, axs = plt.subplots(2, 2, figsize=figsize, squeeze=False, sharey="row",sharex=False)
    axs = axs.flatten()

    label_fs = 9
    tick_fs = 8
    title_fs = 9
    legend_fs = 6

    for ax, (task_title, panel_title, df_use, color_map) in zip(axs, panel_specs):
        populations = list(df_use["population"].dropna().unique())
        populations = [p for p in preferred_order if p in populations] + [
            p for p in populations if p not in preferred_order
        ]

        grouped = (
            df_use.groupby(["population", "N"], as_index=False)
                 .agg(
                     mean_val=(metric, "mean"),
                     std_val=(metric, "std"),
                     sem_val=(metric, "sem"),
                 )
        )

        for population in populations:
            subdf = grouped[grouped["population"] == population].sort_values("N")
            if len(subdf) == 0:
                continue

            x = subdf["N"].values.astype(float)
            y = subdf["mean_val"].values.astype(float)
            c = color_map.get(population, "gray")
            label = nicer_names.get(population, population)

            ax.plot(
                x, y,
                marker="o",
                linewidth=1.2,
                markersize=1.7,
                label=label,
                color=c,
            )

            if errorbar == "sem":
                yerr = subdf["sem_val"].values.astype(float)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.16, color=c)
            elif errorbar == "std":
                yerr = subdf["std_val"].values.astype(float)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.16, color=c)

        ax.set_xlabel("N", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        if ax == axs[0] or ax == axs[1]:
            ax.set_title(f"{panel_title}", fontsize=title_fs)

        if ylim_vals is not None:
            if isinstance(ylim_vals, dict):
                if task_title in ylim_vals and ylim_vals[task_title] is not None:
                    ax.set_ylim(ylim_vals[task_title])
            elif isinstance(ylim_vals, (list, tuple)):
                ax.set_ylim(ylim_vals)
    if metric == "d90":
        metric_label = r"$d_{90}$"
    elif metric == "d95":
        metric_label = r"$d_{95}$"
    else:
        metric_label = metric
    
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[2].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[3].xaxis.set_major_locator(plt.FixedLocator([5,10,15]))

    axs[0].set_ylabel("DMS\n" + f" {metric_label}", fontsize=label_fs)
    axs[2].set_ylabel("Parity\n" + f" {metric_label}", fontsize=label_fs)

    # one legend only
    axs[2].legend(fontsize=legend_fs, frameon=False, loc="best")
    axs[3].legend(fontsize=legend_fs, frameon=False, loc="upper left")

    # remove repeated y tick labels from right column
    axs[1].tick_params(axis="y", labelleft=False)
    axs[3].tick_params(axis="y", labelleft=False)
    # also remove ticks and spines from middle column for cleaner look
    axs[1].tick_params(axis="y", left=False)
    axs[1].spines["left"].set_visible(False)
    axs[3].tick_params(axis="y", left=False)
    axs[3].spines["left"].set_visible(False)


    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()

    return fig, axs