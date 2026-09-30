import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from analysis.rebuild_model_utils import (
    build_model_from_config_and_state,
    load_run_config,
    load_state_dict,
)


# Checkpoint / N utilities

def find_available_Ns(run_path):
    """Sorted checkpoint Ns from files named rnn_N{N}_N{N}[.pt]."""
    run_path = Path(run_path)
    Ns = []

    for path in run_path.iterdir():
        if not path.is_file():
            continue

        match = re.fullmatch(r"rnn_N(\d+)_N\1(?:\.pt)?", path.name)
        if match is not None:
            Ns.append(int(match.group(1)))

    return sorted(set(Ns))


def _resolve_run_Ns(run_path, Ns=None, skip_last=0):
    """Ns for one run: from checkpoint files, falling back to stats.npy."""
    run_path = Path(run_path)

    if Ns is None:
        Ns = find_available_Ns(run_path)

        if len(Ns) == 0:
            stats_path = run_path / "stats.npy"
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"No checkpoints and no stats.npy found in:\n{run_path}"
                )

            stats = np.load(stats_path, allow_pickle=True).item()
            Ns = sorted({int(n) for n in stats["n_task"]})
    else:
        Ns = sorted({int(n) for n in Ns})

    if skip_last:
        if skip_last >= len(Ns):
            raise ValueError("skip_last removes all available Ns")
        Ns = Ns[:-skip_last]

    return Ns


def _resolve_shared_Ns(run_paths, Ns=None, skip_last=0):
    """
    Resolve Ns shared across all runs.
    """
    if Ns is None:
        shared = None

        for run_path in run_paths:
            run_Ns = _resolve_run_Ns(run_path, Ns=None, skip_last=0)
            run_set = set(run_Ns)
            shared = run_set if shared is None else shared.intersection(run_set)

        Ns = sorted(shared) if shared is not None else []
    else:
        Ns = sorted({int(n) for n in Ns})

    if len(Ns) == 0:
        raise ValueError("No shared Ns found across runs.")

    if skip_last:
        if skip_last >= len(Ns):
            raise ValueError("skip_last removes all shared Ns")
        Ns = Ns[:-skip_last]

    return Ns


# Batch / model loading

def make_fixed_eval_batches(
    batch_fn,
    eval_n,
    batch_size=64,
    n_batches=20,
):
    """Fixed evaluation batches for one N, reusable across models and runs."""
    batches = []

    for _ in range(n_batches):
        seqs, labs = batch_fn([eval_n], batch_size)

        if isinstance(labs, (list, tuple)):
            labs_copy = [
                lab.clone() if hasattr(lab, "clone") else torch.as_tensor(lab)
                for lab in labs
            ]
        else:
            labs_copy = [labs.clone() if hasattr(labs, "clone") else torch.as_tensor(labs)]

        batches.append((seqs.clone(), labs_copy))

    return batches


def load_model_for_run_and_N(run_path, N, device, verify=False):
    cfg = load_run_config(run_path)
    sd = load_state_dict(run_path, N=N)

    model = build_model_from_config_and_state(
        cfg=cfg,
        state_dict=sd,
        device=device,
    )

    model.to(device)
    model.eval()

    if verify:
        model_sd = model.state_dict()

        # old checkpoints may contain removed keys
        ignored_keys = {
            "tau_param",
        }

        bad = []

        for k, v in sd.items():
            if k in ignored_keys:
                continue

            if k not in model_sd:
                bad.append((k, "missing_after_load"))
                continue

            if model_sd[k].shape != v.shape:
                bad.append((k, f"shape mismatch {model_sd[k].shape} vs {v.shape}"))
                continue

            diff = (model_sd[k].detach().cpu() - v.detach().cpu()).abs().max().item()

            if diff > 1e-6:
                bad.append((k, diff))

        if len(bad) > 0:
            raise RuntimeError(
                "Loaded weights do not match checkpoint:\n"
                + "\n".join([f"{k}: {v}" for k, v in bad[:20]])
            )

    return model


# Output helpers

def _get_label_from_labs(labs, device="cpu", label_idx=-1):
    """Labels from batch_fn output."""
    if isinstance(labs, (list, tuple)):
        lbl = labs[label_idx]
    else:
        lbl = labs

    return lbl.long().to(device)


def _get_logits_from_out_heads(out_heads, head_idx=0):
    """Logits for head_idx from a list, tensor or dict of output heads."""
    if torch.is_tensor(out_heads):
        return out_heads

    if isinstance(out_heads, (list, tuple)):
        return out_heads[head_idx]

    if isinstance(out_heads, dict):
        if head_idx in out_heads:
            return out_heads[head_idx]

        if str(head_idx) in out_heads:
            return out_heads[str(head_idx)]

        for key in ["class", "out_class", "logits", "output", "y"]:
            if key in out_heads:
                return out_heads[key]

        for value in out_heads.values():
            if torch.is_tensor(value):
                return value

        raise KeyError(
            f"Could not find tensor logits in out_heads dict. "
            f"Available keys: {list(out_heads.keys())}"
        )

    raise TypeError(f"Unrecognised out_heads type: {type(out_heads)}")


def accuracy_from_output_dict(output_dict):
    """
    Accuracy from output dict returned by run_model_on_fixed_batches.
    """
    logits = output_dict["logits"]
    labels = output_dict["labels"]
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


# Evaluation runner
def run_model_on_fixed_batches(
    model,
    fixed_batches,
    head_idx=0,
    device="cpu",
    ablate_cb=False,
    return_outputs=False,
    collect_dynamics=False,
):
    """Evaluate a model on fixed batches. Returns accuracy, or (return_outputs=True)
        a dict with logits, labels and optionally dynamics.
    
    """
    all_labels = []
    all_logits = []
    all_dynamics = []

    model.eval()
    original_use_cb_bias = getattr(model, "use_cb_bias", None)

    try:
        if ablate_cb:
            if original_use_cb_bias is None:
                raise AttributeError(
                    "Model has no attribute 'use_cb_bias'. "
                    "This should only be used for CB-RNN models."
                )
            model.use_cb_bias = False

        with torch.no_grad():
            for seqs, labs in fixed_batches:
                seqs = seqs.to(device)
                lbl = _get_label_from_labs(labs, device=device)

                if collect_dynamics:
                    hs_out, out_heads, dynamics = model(
                        seqs,
                        return_timewise=False,
                        return_dynamics=True,
                    )
                else:
                    hs_out, out_heads = model(
                        seqs,
                        return_timewise=False,
                    )
                    dynamics = None

                logits = _get_logits_from_out_heads(out_heads, head_idx=head_idx)

                all_labels.append(lbl.detach().cpu())
                all_logits.append(logits.detach().cpu())

                if collect_dynamics:
                    dyn_cpu = {}

                    if isinstance(dynamics, dict):
                        for k, v in dynamics.items():
                            dyn_cpu[k] = v.detach().cpu() if hasattr(v, "detach") else v

                    # keep batch labels with the dynamics (for t-SNE / decoding)
                    dyn_cpu["_batch_labels"] = lbl.detach().cpu()

                    all_dynamics.append(dyn_cpu)

    finally:
        if ablate_cb and original_use_cb_bias is not None:
            model.use_cb_bias = original_use_cb_bias

    output = {
        "logits": torch.cat(all_logits, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }

    if collect_dynamics:
        output["dynamics"] = all_dynamics

    if return_outputs:
        return output

    return accuracy_from_output_dict(output)

def evaluate_accuracy_for_n(
    model,
    batch_fn,
    eval_n,
    batch_size=64,
    n_batches=20,
    head_idx=0,
    device="cpu",
):
    """
    Evaluate model accuracy for one N using newly sampled batches.
    """
    fixed_batches = make_fixed_eval_batches(
        batch_fn=batch_fn,
        eval_n=eval_n,
        batch_size=batch_size,
        n_batches=n_batches,
    )

    return run_model_on_fixed_batches(
        model=model,
        fixed_batches=fixed_batches,
        head_idx=head_idx,
        device=device,
        ablate_cb=False,
        return_outputs=False,
        collect_dynamics=False,
    )


def evaluate_accuracy_with_cb_ablated(
    model,
    batch_fn,
    eval_n,
    batch_size=64,
    n_batches=20,
    head_idx=0,
    device="cpu",
):
    """
    Evaluate a CB-RNN model with CB disabled.
    """
    fixed_batches = make_fixed_eval_batches(
        batch_fn=batch_fn,
        eval_n=eval_n,
        batch_size=batch_size,
        n_batches=n_batches,
    )

    return run_model_on_fixed_batches(
        model=model,
        fixed_batches=fixed_batches,
        head_idx=head_idx,
        device=device,
        ablate_cb=True,
        return_outputs=False,
        collect_dynamics=False,
    )


# CB ablation evaluation

def compare_cb_ablation_for_n(
    model,
    batch_fn,
    eval_n,
    batch_size=64,
    n_batches=20,
    head_idx=0,
    device="cpu",
    fixed_batches=None,
    return_outputs=False,
    collect_dynamics=False,
):
    """
    Compare full CB-RNN vs CB-ablated model for one N.
    """
    if fixed_batches is None:
        fixed_batches = make_fixed_eval_batches(
            batch_fn=batch_fn,
            eval_n=eval_n,
            batch_size=batch_size,
            n_batches=n_batches,
        )

    out_full = run_model_on_fixed_batches(
        model=model,
        fixed_batches=fixed_batches,
        head_idx=head_idx,
        device=device,
        ablate_cb=False,
        return_outputs=return_outputs,
        collect_dynamics=collect_dynamics,
    )

    out_no_cb = run_model_on_fixed_batches(
        model=model,
        fixed_batches=fixed_batches,
        head_idx=head_idx,
        device=device,
        ablate_cb=True,
        return_outputs=return_outputs,
        collect_dynamics=collect_dynamics,
    )

    if return_outputs:
        acc_full = accuracy_from_output_dict(out_full)
        acc_no_cb = accuracy_from_output_dict(out_no_cb)
    else:
        acc_full = out_full
        acc_no_cb = out_no_cb

    result = {
        "N": int(eval_n),
        "acc_full": acc_full,
        "acc_cb_ablated": acc_no_cb,
        "acc_drop": acc_full - acc_no_cb,
    }

    if return_outputs:
        return result, {"full": out_full, "ablated": out_no_cb}

    return result


def compare_cb_ablation_across_Ns(
    run_path,
    batch_fn,
    batch_size=64,
    n_batches=20,
    head_idx=0,
    device="cpu",
    Ns=None,
    skip_last=0,
    fixed_batches_by_N=None,
    return_outputs=False,
    collect_dynamics=False,
):
    """
    Compare full vs CB-ablated model across Ns for one run.
    """
    Ns = _resolve_run_Ns(run_path, Ns=Ns, skip_last=skip_last)

    if fixed_batches_by_N is None:
        fixed_batches_by_N = {
            N: make_fixed_eval_batches(
                batch_fn=batch_fn,
                eval_n=N,
                batch_size=batch_size,
                n_batches=n_batches,
            )
            for N in Ns
        }

    run_id = os.path.basename(str(run_path).rstrip("/"))
    rows = []
    outputs = {}

    for N in Ns:
        print(f"{run_id} | N={N}")

        model = load_model_for_run_and_N(run_path, N=N, device=device)

        if return_outputs:
            row, out = compare_cb_ablation_for_n(
                model=model,
                batch_fn=batch_fn,
                eval_n=N,
                batch_size=batch_size,
                n_batches=n_batches,
                head_idx=head_idx,
                device=device,
                fixed_batches=fixed_batches_by_N[N],
                return_outputs=True,
                collect_dynamics=collect_dynamics,
            )
            outputs[N] = out
        else:
            row = compare_cb_ablation_for_n(
                model=model,
                batch_fn=batch_fn,
                eval_n=N,
                batch_size=batch_size,
                n_batches=n_batches,
                head_idx=head_idx,
                device=device,
                fixed_batches=fixed_batches_by_N[N],
                return_outputs=False,
                collect_dynamics=False,
            )

        row["run_id"] = run_id
        row["run_path"] = str(run_path)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

    if return_outputs:
        return df, outputs

    return df


def compare_cb_ablation_many_runs(
    run_paths,
    batch_fn,
    batch_size=64,
    n_batches=20,
    head_idx=0,
    device="cpu",
    Ns=None,
    skip_last=0,
    return_outputs=False,
    collect_dynamics=False,
):
    """Full vs CB-ablated accuracy across runs, using shared fixed batches per N."""
    Ns = _resolve_shared_Ns(run_paths, Ns=Ns, skip_last=skip_last)

    fixed_batches_by_N = {
        N: make_fixed_eval_batches(
            batch_fn=batch_fn,
            eval_n=N,
            batch_size=batch_size,
            n_batches=n_batches,
        )
        for N in Ns
    }

    dfs = []
    all_outputs = {}

    for i, run_path in enumerate(run_paths):
        run_id = os.path.basename(str(run_path).rstrip("/"))
        print(f"[{i + 1}/{len(run_paths)}] {run_id}")

        if return_outputs:
            df_run, outputs = compare_cb_ablation_across_Ns(
                run_path=run_path,
                batch_fn=batch_fn,
                batch_size=batch_size,
                n_batches=n_batches,
                head_idx=head_idx,
                device=device,
                Ns=Ns,
                skip_last=0,
                fixed_batches_by_N=fixed_batches_by_N,
                return_outputs=True,
                collect_dynamics=collect_dynamics,
            )
            all_outputs[run_id] = outputs
        else:
            df_run = compare_cb_ablation_across_Ns(
                run_path=run_path,
                batch_fn=batch_fn,
                batch_size=batch_size,
                n_batches=n_batches,
                head_idx=head_idx,
                device=device,
                Ns=Ns,
                skip_last=0,
                fixed_batches_by_N=fixed_batches_by_N,
                return_outputs=False,
                collect_dynamics=False,
            )

        dfs.append(df_run)

    df_all = pd.concat(dfs, ignore_index=True)

    if return_outputs:
        return df_all, all_outputs, fixed_batches_by_N

    return df_all


# alias for older calls
def evaluate_many_runs_over_Ns_with_shared_fixed_batches(
    run_paths,
    batch_fn,
    Ns,
    batch_size=64,
    n_batches=10,
    head_idx=0,
    device="cpu",
    return_outputs=False,
    collect_dynamics=True,
):
    """
    Backwards-compatible wrapper around compare_cb_ablation_many_runs.
    """
    return compare_cb_ablation_many_runs(
        run_paths=run_paths,
        batch_fn=batch_fn,
        batch_size=batch_size,
        n_batches=n_batches,
        head_idx=head_idx,
        device=device,
        Ns=Ns,
        skip_last=0,
        return_outputs=return_outputs,
        collect_dynamics=collect_dynamics,
    )


def summarize_cb_ablation_many_runs(df_all):
    """
    Summarize CB ablation dataframe across runs.
    """
    return (
        df_all.groupby("N", as_index=False)
        .agg(
            acc_full_mean=("acc_full", "mean"),
            acc_full_std=("acc_full", "std"),
            acc_full_sem=("acc_full", "sem"),
            acc_cb_ablated_mean=("acc_cb_ablated", "mean"),
            acc_cb_ablated_std=("acc_cb_ablated", "std"),
            acc_cb_ablated_sem=("acc_cb_ablated", "sem"),
            acc_drop_mean=("acc_drop", "mean"),
            acc_drop_std=("acc_drop", "std"),
            acc_drop_sem=("acc_drop", "sem"),
            n_runs=("run_id", "nunique"),
        )
        .sort_values("N")
        .reset_index(drop=True)
    )


# RNN-only evaluation

def evaluate_rnnonly_acc(
    run_path,
    batch_fn,
    batch_size=64,
    n_batches=20,
    head_idx=0,
    device="cpu",
    Ns=None,
    skip_last=0,
):
    """RNN-only accuracy across Ns."""
    Ns = _resolve_run_Ns(run_path, Ns=Ns, skip_last=skip_last)
    run_id = os.path.basename(str(run_path).rstrip("/"))

    rows = []

    for N in Ns:
        print(f"{run_id} | N={N}")

        model = load_model_for_run_and_N(run_path, N=N, device=device)

        acc = evaluate_accuracy_for_n(
            model=model,
            batch_fn=batch_fn,
            eval_n=N,
            batch_size=batch_size,
            n_batches=n_batches,
            head_idx=head_idx,
            device=device,
        )

        rows.append({
            "run_id": run_id,
            "run_path": str(run_path),
            "N": int(N),
            "accuracy": acc,
        })

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def evaluate_rnnonly_many_runs(
    run_paths,
    batch_fn,
    batch_size=64,
    n_batches=20,
    head_idx=0,
    device="cpu",
    Ns=None,
    skip_last=0,
):
    """
    Evaluate RNN-only models across many runs.
    """
    Ns = _resolve_shared_Ns(run_paths, Ns=Ns, skip_last=skip_last)
    dfs = []

    for i, run_path in enumerate(run_paths):
        run_id = os.path.basename(str(run_path).rstrip("/"))
        print(f"[{i + 1}/{len(run_paths)}] {run_id}")

        df_run = evaluate_rnnonly_acc(
            run_path=run_path,
            batch_fn=batch_fn,
            batch_size=batch_size,
            n_batches=n_batches,
            head_idx=head_idx,
            device=device,
            Ns=Ns,
            skip_last=0,
        )

        dfs.append(df_run)

    return pd.concat(dfs, ignore_index=True)


def summarize_rnnonly_many_runs(df_all):
    """
    Summarize RNN-only accuracy across runs.
    """
    return (
        df_all.groupby("N", as_index=False)
        .agg(
            acc_full_mean=("accuracy", "mean"),
            acc_full_std=("accuracy", "std"),
            acc_full_sem=("accuracy", "sem"),
            n_runs=("run_id", "nunique"),
        )
        .sort_values("N")
        .reset_index(drop=True)
    )


# Plot: CB ablation accuracy

def plot_cb_ablation_two_panel(
    df_dms_ablate,
    df_dms_rnn,
    df_parity_ablate,
    df_parity_rnn,
    save_path=None,
    linewidth_pt=397.48499,
    fig_height=1.45,
    show_sem=True,
    fig_size=None,
    xlim=None,
    ylim=(0.4, 1.02),
    colors=None,
):
    """
    Two-panel CB ablation figure for DMS and Parity.
    """
    if colors is None:
        colors = {
            "cb_full": "salmon",
            "cb_ablated": "#fcaca3",
        }

    inches_per_pt = 1 / 72.27
    plot_block_frac = 0.81
    fig_width = linewidth_pt * inches_per_pt * plot_block_frac

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height) if fig_size is None else fig_size,
        sharey=True,
        constrained_layout=False,
    )

    def _plot_one(ax, df_ablate, df_rnn, title):
        df_ablate = df_ablate.sort_values("N").copy()
        df_rnn = df_rnn.sort_values("N").copy()

        x_cb = df_ablate["N"].to_numpy()
        y_full = df_ablate["acc_full_mean"].to_numpy()
        y_ablate = df_ablate["acc_cb_ablated_mean"].to_numpy()

        x_rnn = df_rnn["N"].to_numpy()
        y_rnn = df_rnn["acc_full_mean"].to_numpy()

        ax.plot(
            x_cb,
            y_full,
            color=colors["cb_full"],
            lw=1,
            marker="o",
            ms=1.2,
            label="CB-RNN",
        )

        ax.plot(
            x_cb,
            y_ablate,
            color=colors["cb_ablated"],
            lw=1,
            marker="o",
            ls="--",
            ms=1.2,
            label="CB ablated",
        )


        if show_sem:
            if "acc_full_sem" in df_ablate:
                sem = df_ablate["acc_full_sem"].to_numpy()
                ax.fill_between(
                    x_cb,
                    y_full - sem,
                    y_full + sem,
                    color=colors["cb_full"],
                    alpha=0.18,
                    linewidth=0,
                )

            if "acc_cb_ablated_sem" in df_ablate:
                sem = df_ablate["acc_cb_ablated_sem"].to_numpy()
                ax.fill_between(
                    x_cb,
                    y_ablate - sem,
                    y_ablate + sem,
                    color=colors["cb_ablated"],
                    alpha=0.18,
                    linewidth=0,
                )

        ax.set_title(title, fontsize=9, fontweight="normal")
        ax.set_xlabel("N", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            all_x = np.concatenate([x_cb, x_rnn])
            ax.set_xlim(np.nanmin(all_x) - 0.5, np.nanmax(all_x) + 0.5)

        if ylim is not None:
            ax.set_ylim(*ylim)

    _plot_one(axs[0], df_dms_ablate, df_dms_rnn, "DMS")
    _plot_one(axs[1], df_parity_ablate, df_parity_rnn, "Parity")

    axs[0].set_ylabel("Accuracy", fontsize=8)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=1,
        frameon=False,
        fontsize=6,
        handlelength=1.5,
        columnspacing=1.0,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.24,
        top=0.82,
        wspace=0.18,
    )

    if save_path is not None:
        save_path = Path(save_path)
        suffix = save_path.suffix.lower().replace(".", "")

        if suffix == "":
            suffix = "svg"
            save_path = save_path.with_suffix(".svg")

        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)

        for ax in axs:
            ax.set_facecolor("white")
            ax.patch.set_alpha(1.0)

        fig.savefig(
            save_path,
            format=suffix,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            transparent=False,
        )

    return fig, axs


def plot_cb_ablation_acc_bars(
    df_dms=None,
    df_parity=None,
    save_path=None,
    figsize=None,
    colors=None,
    bar_width=0.25,
    show_points=True,
    big_font=True,
    ylim=(0.0, 1.05),
):
    """Bar plot of full vs CB-ablated accuracy (mean, SEM across networks; dots are networks).

        Pass df_dms and/or df_parity from compare_cb_ablation_many_runs.
    
    """
    if colors is None:
        colors = {"full": "salmon", "ablated": "#fcaca3"}
    dot_color = "#8f3b30"

    tasks = [
        (name, df)
        for name, df in [("DMS", df_dms), ("Parity", df_parity)]
        if df is not None
    ]
    if not tasks:
        raise ValueError("Pass at least one of df_dms / df_parity.")
    single = len(tasks) == 1
    if figsize is None:
        figsize = (2.6, 2.5) if single else (4.5, 2.5)

    conditions = [
        ("acc_full", "Full", colors["full"], -(bar_width / 2 + 0.01)),
        ("acc_cb_ablated", "CB ablated", colors["ablated"], +(bar_width / 2 + 0.01)),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)

    for ti, (_, df) in enumerate(tasks):
        per_net = df.groupby("run_id")[["acc_full", "acc_cb_ablated"]].mean()
        jitter = rng.uniform(-bar_width * 0.18, bar_width * 0.18, size=len(per_net))

        xs = {}
        for col, label, color, off in conditions:
            vals = per_net[col].to_numpy()
            n = len(vals)
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            x = ti + off
            xs[col] = x + jitter

            ax.bar(
                x,
                mean,
                width=bar_width,
                color=color,
                edgecolor="none",
                label=label if ti == 0 else None,
                zorder=2,
            )
            ax.errorbar(
                x,
                mean,
                yerr=sem,
                fmt="none",
                ecolor="0.2",
                elinewidth=0.9,
                capsize=2.5,
                capthick=0.9,
                zorder=4,
            )

        if show_points:
            ax.plot(
                [xs["acc_full"], xs["acc_cb_ablated"]],
                [per_net["acc_full"], per_net["acc_cb_ablated"]],
                color=dot_color,
                lw=0.4,
                alpha=0.35,
                zorder=3,
            )
            for col, _, _, _ in conditions:
                ax.scatter(
                    xs[col],
                    per_net[col],
                    s=12,
                    color=dot_color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.9,
                    zorder=3,
                )

    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels([name for name, _ in tasks], fontsize=12 if big_font else 8)
    ax.set_xlim(-0.5, 0.5) if single else ax.set_xlim(-0.4, len(tasks) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Accuracy", fontsize=12 if big_font else 8)
    ax.tick_params(axis="y", labelsize=10 if big_font else 7)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=10 if big_font else 8,
        handlelength=1.0,
        columnspacing=1.2,
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        suffix = save_path.suffix.lower().replace(".", "")

        if suffix == "":
            suffix = "svg"
            save_path = save_path.with_suffix(".svg")

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        fig.savefig(
            save_path,
            format=suffix,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            transparent=True,
            bbox_inches="tight",
            dpi=900
        )

    return fig, ax

# t-SNE of hidden-state class separation

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator

def extract_final_module_points(activity_output, module_key="hidden"):
    X_all = []
    y_all = []

    for dyn in activity_output["dynamics"]:
        if module_key not in dyn:
            raise KeyError(
                f"module_key='{module_key}' not found in dynamics. "
                f"Available keys: {list(dyn.keys())}"
            )
        if "_batch_labels" not in dyn:
            raise KeyError(
                "'_batch_labels' not found in dynamics. "
                "Re-run evaluation with the updated run_model_on_fixed_batches."
            )

        X_mod = dyn[module_key]
        if hasattr(X_mod, "detach"):
            X_mod = X_mod.detach().cpu().numpy()
        else:
            X_mod = np.asarray(X_mod)

        if X_mod.ndim != 3:
            raise ValueError(
                f"Expected dyn['{module_key}'] shape [T, B, D], got {X_mod.shape}."
            )

        X_all.append(X_mod[-1])          # [B, D] final timestep
        y_all.append(dyn["_batch_labels"].numpy())

    if len(X_all) == 0:
        raise ValueError("No dynamics found. Run with collect_dynamics=True.")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def project_full_vs_ablated_final_points_tsne(
    output_full,
    output_ablated,
    module_key="hidden",
    n_components=2,
    perplexity=30,
    random_state=0,
    init="pca",
    learning_rate="auto",
):
    """
    Jointly project full and CB-ablated hidden states into the same t-SNE space.
    """
    X_full, y_full = extract_final_module_points(output_full, module_key=module_key)
    X_abl, y_abl = extract_final_module_points(output_ablated, module_key=module_key)

    X_joint = np.concatenate([X_full, X_abl], axis=0)

    scaler = StandardScaler()
    X_joint_z = scaler.fit_transform(X_joint)

    # t-SNE requires perplexity < n_samples.
    max_safe_perplexity = max(2, (len(X_joint_z) - 1) // 3)
    perplexity = min(perplexity, max_safe_perplexity)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init=init,
        learning_rate=learning_rate,
    )

    X_joint_tsne = tsne.fit_transform(X_joint_z)

    n_full = len(X_full)

    return {
        "X_full": X_joint_tsne[:n_full],
        "y_full": y_full,
        "X_abl": X_joint_tsne[n_full:],
        "y_abl": y_abl,
        "tsne": tsne,
        "scaler": scaler,
    }


def plot_hidden_class_points_multiple_Ns_tsne(
    all_outputs,
    run_id,
    Ns,
    module_key="hidden",
    perplexity=30,
    random_state=0,
    class_colors=None,
    figsize=None,
    save_path=None,
):
    """Full vs CB-ablated t-SNE across Ns (rows = Ns, columns = Full / Ablated)."""
    if class_colors is None:
        class_colors = {
            0: "crimson",
            1: "mediumblue",
        }

    if figsize is None:
        figsize = (5.5, 2.0 * len(Ns))

    fig, axs = plt.subplots(
        len(Ns),
        2,
        figsize=figsize,
        sharex="row",
        sharey="row",
        squeeze=False,
    )

    for i, N in enumerate(Ns):
        output_full = all_outputs[run_id][N]["full"]
        output_abl = all_outputs[run_id][N]["ablated"]

        proj = project_full_vs_ablated_final_points_tsne(
            output_full,
            output_abl,
            module_key=module_key,
            perplexity=perplexity,
            random_state=random_state,
        )

        panels = [
            (proj["X_full"], proj["y_full"], "Full"),
            (proj["X_abl"], proj["y_abl"], "CB ablated"),
        ]

        for j, (X, y, title) in enumerate(panels):
            ax = axs[i, j]

            for cls in sorted(np.unique(y)):
                cls_int = int(cls)
                pts = X[y == cls]
                color = class_colors.get(cls_int, "gray")

                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    alpha=0.45,
                    s=4,
                    color=color,
                    label=f"class {cls_int}",
                    rasterized=True,
                )

                if len(pts) > 0:
                    mu = pts.mean(axis=0)
                    ax.scatter(
                        mu[0],
                        mu[1],
                        s=30,
                        marker="X",
                        color=color,
                        edgecolor="black",
                        linewidth=0.3,
                    )

            ax.set_title(f"N={N} | {title}", fontsize=8, fontweight="normal")
            ax.set_xlabel("t-SNE 1", fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

            if j == 0:
                ax.set_ylabel("t-SNE 2", fontsize=7)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles, labels, fontsize=6, frameon=False, loc="best")

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.94,
        wspace=0.12,
        hspace=0.45,
    )

    if save_path is not None:
        save_path = Path(save_path)
        suffix = save_path.suffix.lower().replace(".", "")

        if suffix == "":
            suffix = "svg"
            save_path = save_path.with_suffix(".svg")

        fig.savefig(
            save_path,
            format=suffix,
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )

    return fig, axs


def plot_hidden_class_points_two_tasks_tsne_1row4(
    all_outputs_task1,
    run_id_task1,
    N_task1,
    all_outputs_task2,
    run_id_task2,
    N_task2,
    module_key="hidden",
    perplexity=30,
    random_state=0,
    task1_title="DMS",
    task2_title="Parity",
    linewidth_pt=397.48499,
    plot_block_frac=0.81,
    figsize=None,
    big_fonts=False,
    fig_height=1.0,
    save_path=None,
    class_colors=None,
):
    """One-row t-SNE figure: DMS full | DMS ablated | Parity full | Parity ablated.

        Full and ablated are embedded jointly within each task.
    
    """
    if class_colors is None:
        class_colors = {
            0: "orange",
            1: "#009E73",
        }

    inches_per_pt = 1 / 72.27
    fig_width = linewidth_pt * inches_per_pt * plot_block_frac

    if figsize is None:
        figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[1, 1, 0.22, 1, 1],
        wspace=0.08,
    )

    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 4]),
    ]

    axs[1].sharex(axs[0])
    axs[1].sharey(axs[0])
    axs[3].sharex(axs[2])
    axs[3].sharey(axs[2])

    output1_full = all_outputs_task1[run_id_task1][N_task1]["full"]
    output1_abl = all_outputs_task1[run_id_task1][N_task1]["ablated"]

    output2_full = all_outputs_task2[run_id_task2][N_task2]["full"]
    output2_abl = all_outputs_task2[run_id_task2][N_task2]["ablated"]

    proj1 = project_full_vs_ablated_final_points_tsne(
        output1_full,
        output1_abl,
        module_key=module_key,
        perplexity=perplexity,
        random_state=random_state,
    )

    proj2 = project_full_vs_ablated_final_points_tsne(
        output2_full,
        output2_abl,
        module_key=module_key,
        perplexity=perplexity,
        random_state=random_state,
    )

    panel_specs = [
        (proj1["X_full"], proj1["y_full"], "Full"),
        (proj1["X_abl"], proj1["y_abl"], "Ablated"),
        (proj2["X_full"], proj2["y_full"], "Full"),
        (proj2["X_abl"], proj2["y_abl"], "Ablated"),
    ]
    if big_fonts:
        label_fs = 12
        tick_fs = 10
        title_fs = 12
        legend_fs = 10
    else:
        label_fs = 8
        tick_fs = 7
        title_fs = 8
        legend_fs = 6

    for ax, (X, y, cond_title) in zip(axs, panel_specs):
        for cls in sorted(np.unique(y)):
            cls_int = int(cls)
            pts = X[y == cls]
            color = class_colors.get(cls_int, "gray")

            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                alpha=0.6,
                s=3.5,
                color=color,
                linewidths=0,
                label=f"class {cls_int}",
                rasterized=False,
            )

        ax.set_title(cond_title, fontsize=title_fs, fontweight="normal")
        ax.set_xlabel("t-SNE 1", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    axs[0].set_ylabel("t-SNE 2", fontsize=label_fs)

    for ax in [axs[1], axs[3]]:
        ax.tick_params(axis="y", left=False, labelleft=False)

    handles, labels = axs[0].get_legend_handles_labels()
    labels_caps = [lab.capitalize() for lab in labels]
    axs[1].legend(
        handles,
        labels_caps,
        fontsize=legend_fs,
        frameon=False,
        loc="upper left",
        handletextpad=0.2,
        borderpad=0.1,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.25,
        top=0.78,
        wspace=0.08,
    )

    if save_path is not None:
        save_path = Path(save_path)
        suffix = save_path.suffix.lower().replace(".", "")

        if suffix == "":
            suffix = "svg"
            save_path = save_path.with_suffix("svg")

        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)

        for ax in axs:
            ax.set_facecolor("white")
            ax.patch.set_alpha(1.0)

        fig.savefig(
            save_path,
            format=suffix,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            transparent=False,
            dpi=900,
        )

    return fig, axs


def _view_separability_2d(P, g0, g1):
    """Line separability of two classes in a 2D projection. Returns (balanced acc, Mahalanobis^2)."""
    P0, P1 = P[g0], P[g1]
    m0, m1 = P0.mean(axis=0), P1.mean(axis=0)
    Sw = np.cov(P0.T) * (len(P0) - 1) + np.cov(P1.T) * (len(P1) - 1)
    Sw = Sw + 1e-9 * (np.trace(Sw) + 1e-12) * np.eye(2)

    w = np.linalg.solve(Sw, m1 - m0)
    u = P @ w
    thr = 0.5 * (u[g0].mean() + u[g1].mean())
    pred1 = u > thr

    acc = 0.5 * (pred1[g1].mean() + (~pred1[g0]).mean())
    return float(acc), float((m1 - m0) @ w)


def _view_score_on_axes(ax, X, g0, g1):
    """(balanced acc, Mahalanobis^2) of line separability at the axes' current view."""
    from mpl_toolkits.mplot3d import proj3d

    xs, ys, _ = proj3d.proj_transform(X[:, 0], X[:, 1], X[:, 2], ax.get_proj())
    return _view_separability_2d(np.column_stack([xs, ys]), g0, g1)


def get_tsne_3d_views(axs):
    """(elev, azim) per task from the 3D plot's axes. Returns ((elev1, elev2), (azim1, azim2))."""
    return (axs[0].elev, axs[2].elev), (axs[0].azim, axs[2].azim)


def _best_view_on_axes(ax, X, y, elev_step=5, azim_step=5, refine_step=1):
    """Grid-search the 3D camera for the view where one line best separates the classes."""
    from mpl_toolkits.mplot3d import proj3d

    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Auto view needs exactly 2 classes, got {classes}.")
    g0, g1 = y == classes[0], y == classes[1]

    def score(e, a):
        ax.view_init(elev=e, azim=a)
        acc, fisher = _view_score_on_axes(ax, X, g0, g1)
        return round(acc, 3), fisher

    def search(elevs, azims):
        best_key, best_view = None, None
        for e in elevs:
            for a in azims:
                key = score(e, a)
                if best_key is None or key > best_key:
                    best_key, best_view = key, (e, a)
        return best_key, best_view

    key, (e0, a0) = search(
        np.arange(-90, 90 + 1e-9, elev_step),
        np.arange(-180, 180, azim_step),
    )
    key, (e0, a0) = search(
        np.clip(np.arange(e0 - elev_step, e0 + elev_step + 1e-9, refine_step), -90, 90),
        np.arange(a0 - azim_step, a0 + azim_step + 1e-9, refine_step),
    )

    ax.view_init(elev=e0, azim=a0)
    return {"elev": float(e0), "azim": float(a0), "accuracy": key[0], "fisher": key[1]}


def compute_two_tasks_tsne_3d_projections(
    all_outputs_task1,
    run_id_task1,
    N_task1,
    all_outputs_task2,
    run_id_task2,
    N_task2,
    module_key="hidden",
    perplexity=30,
    random_state=0,
):
    """3-component t-SNE per task. Returns (proj_task1, proj_task2)."""
    projs = []
    for all_outputs, run_id, N in [
        (all_outputs_task1, run_id_task1, N_task1),
        (all_outputs_task2, run_id_task2, N_task2),
    ]:
        projs.append(
            project_full_vs_ablated_final_points_tsne(
                all_outputs[run_id][N]["full"],
                all_outputs[run_id][N]["ablated"],
                module_key=module_key,
                n_components=3,
                perplexity=perplexity,
                random_state=random_state,
            )
        )
    return tuple(projs)


def project_full_vs_ablated_final_points_supervised(
    output_full,
    output_ablated,
    module_key="hidden",
    C=1.0,
    random_state=0,
):
    """Supervised 3D projection of final hidden states: readout direction plus the top two
        PCs of the remaining Full states. Fitted on Full, applied unchanged to Ablated.
    
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X_full, y_full = extract_final_module_points(output_full, module_key=module_key)
    X_abl, y_abl = extract_final_module_points(output_ablated, module_key=module_key)
    y_fit = np.asarray(y_full).astype(int)

    mu = X_full.mean(axis=0)
    Xc_full, Xc_abl = X_full - mu, X_abl - mu

    clf = LogisticRegression(C=C, max_iter=5000, random_state=random_state)
    clf.fit(Xc_full, y_fit)
    w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])

    # Top-2 PCs of the Full states with the readout axis projected out.
    resid = Xc_full - np.outer(Xc_full @ w, w)
    _, _, Vt = np.linalg.svd(resid, full_matrices=False)
    B = np.column_stack([w, Vt[0], Vt[1]])  # [D, 3], orthonormal

    cv = min(5, int(np.bincount(y_fit).min()))
    acc_full = float(cross_val_score(
        LogisticRegression(C=C, max_iter=5000), Xc_full, y_fit, cv=cv
    ).mean())
    acc_abl = float((clf.predict(Xc_abl) == np.asarray(y_abl).astype(int)).mean())

    return {
        "X_full": Xc_full @ B,
        "y_full": y_full,
        "X_abl": Xc_abl @ B,
        "y_abl": y_abl,
        "axis_labels": ("Decoding axis", "PC 1", "PC 2"),
        "readout_acc_full": acc_full,
        "readout_acc_abl": acc_abl,
    }


def compute_two_tasks_supervised_3d_projections(
    all_outputs_task1,
    run_id_task1,
    N_task1,
    all_outputs_task2,
    run_id_task2,
    N_task2,
    module_key="hidden",
    verbose=True,
):
    """Supervised counterpart of compute_two_tasks_tsne_3d_projections."""
    projs = []
    for name, all_outputs, run_id, N in [
        ("task1", all_outputs_task1, run_id_task1, N_task1),
        ("task2", all_outputs_task2, run_id_task2, N_task2),
    ]:
        proj = project_full_vs_ablated_final_points_supervised(
            all_outputs[run_id][N]["full"],
            all_outputs[run_id][N]["ablated"],
            module_key=module_key,
        )
        if verbose:
            print(
                f"{name}: linear readout acc  Full (CV) = {proj['readout_acc_full']:.3f}, "
                f"Full-fitted classifier on Ablated = {proj['readout_acc_abl']:.3f}"
            )
        projs.append(proj)
    return tuple(projs)


def _panel_extent_px(ax, renderer):
    """Display-space bbox (x0, y0, x1, y1) of one 3D panel including its axis labels."""
    from mpl_toolkits.mplot3d import proj3d

    corners = np.array(
        [[x, y, z] for x in ax.get_xlim3d() for y in ax.get_ylim3d() for z in ax.get_zlim3d()]
    )
    px, py, _ = proj3d.proj_transform(corners[:, 0], corners[:, 1], corners[:, 2], ax.get_proj())
    pts = ax.transData.transform(np.column_stack([px, py]))
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        bb = axis.label.get_window_extent(renderer)
        x0, y0 = min(x0, bb.x0), min(y0, bb.y0)
        x1, y1 = max(x1, bb.x1), max(y1, bb.y1)
    return x0, y0, x1, y1


def _open_gaps_between_3d_panels(fig, axs, adjust, left_reserved_in=0.0, pad_in=0.05, max_iter=40):
    """Widen margins/gaps of a 2x2 grid of 3D panels until nothing collides.
        Returns True if everything is clear.
    
    """
    pad_px = pad_in * fig.dpi
    W, H = fig.get_size_inches() * fig.dpi
    left_reserved_px = left_reserved_in * fig.dpi

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tl, tr, bl, br = (_panel_extent_px(ax, renderer) for ax in axs)
        legend_top = max((lg.get_window_extent(renderer).y1 for lg in fig.legends), default=0.0)

        gap_v = min(tl[1], tr[1]) - max(bl[3], br[3])   # top row bottom vs bottom row top
        gap_h = min(tr[0], br[0]) - max(tl[2], bl[2])   # right col left vs left col right
        need_left = left_reserved_px - min(tl[0], bl[0])
        need_right = max(tr[2], br[2]) - (W - pad_px)
        need_bottom = legend_top + pad_px - min(bl[1], br[1])

        if max(pad_px - gap_v, pad_px - gap_h, need_left, need_right, need_bottom) <= 0:
            return True

        win = axs[0].get_window_extent(renderer)
        if gap_v < pad_px:
            adjust["hspace"] = min(adjust["hspace"] + (pad_px - gap_v) / win.height, 1.5)
        if gap_h < pad_px:
            adjust["wspace"] = min(adjust["wspace"] + (pad_px - gap_h) / win.width, 1.0)
        if need_left > 0:
            adjust["left"] = min(adjust["left"] + need_left / W, 0.35)
        if need_right > 0:
            adjust["right"] = max(adjust["right"] - need_right / W, 0.65)
        if need_bottom > 0:
            adjust["bottom"] = min(adjust["bottom"] + need_bottom / H, 0.35)
        fig.subplots_adjust(**adjust)

    return False


def plot_hidden_class_points_two_tasks_tsne_3d_1row4(
    all_outputs_task1=None,
    run_id_task1=None,
    N_task1=None,
    all_outputs_task2=None,
    run_id_task2=None,
    N_task2=None,
    module_key="hidden",
    perplexity=30,
    random_state=0,
    task1_title="DMS",
    task2_title="Parity",
    linewidth_pt=397.48499,
    plot_block_frac=0.81,
    figsize=None,
    big_fonts=False,
    fig_height=None,
    elev=20,
    azim=45,
    point_size=3.5,
    show_ticks=False,
    show_grid=True,
    save_path=None,
    class_colors=None,
    projections=None,
    interactive=False,
    auto_view=False,
    verbose=True,
    layout="2x2",
    zoom=1.0,
    font_scale=1.0,
    x_stretch=1.0,
):
    """3D version of plot_hidden_class_points_two_tasks_tsne_1row4.

        layout: '2x2' (rows = tasks) or '1x4'; axs is always [t1 full, t1 abl, t2 full, t2 abl].
        zoom: cube size within each panel. x_stretch: stretch along the first axis.
        font_scale: multiplies all font sizes. fig_height: height for '1x4'.
        projections: precomputed (proj_task1, proj_task2); skips t-SNE.
        interactive: link Full/Ablated cameras and show the angles (needs %matplotlib widget).
        elev, azim: one value or a (task1, task2) pair.
        auto_view: pick the best-separating camera on the Full panel.
    
    """
    if class_colors is None:
        class_colors = {
            0: "orange",
            1: "#009E73",
        }

    inches_per_pt = 1 / 72.27
    fig_width = linewidth_pt * inches_per_pt * plot_block_frac

    if layout not in ("2x2", "1x4"):
        raise ValueError(f"layout must be '2x2' or '1x4', got {layout!r}")
    grid_2x2 = layout == "2x2"

    if figsize is None:
        if grid_2x2:
            figsize = (fig_width * 0.75, fig_width * 0.75)
        else:
            figsize = (fig_width, 1.6 if fig_height is None else fig_height)

    fig = plt.figure(figsize=figsize)

    if grid_2x2:
        # gaps are set via fig.subplots_adjust, not the gridspec
        gs = fig.add_gridspec(2, 2)
        cells = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
    else:
        gs = fig.add_gridspec(
            1,
            5,
            width_ratios=[1, 1, 0.12, 1, 1],
            wspace=0.0,
        )
        cells = [gs[0, 0], gs[0, 1], gs[0, 3], gs[0, 4]]

    axs = [fig.add_subplot(cell, projection="3d") for cell in cells]

    if grid_2x2:
        # transparent axes, top row drawn above the bottom row so labels are not covered
        for ax in axs:
            ax.patch.set_visible(False)
        for ax in axs[:2]:
            ax.set_zorder(2)

    if projections is None:
        projections = compute_two_tasks_tsne_3d_projections(
            all_outputs_task1,
            run_id_task1,
            N_task1,
            all_outputs_task2,
            run_id_task2,
            N_task2,
            module_key=module_key,
            perplexity=perplexity,
            random_state=random_state,
        )
    proj1, proj2 = projections

    panel_specs = [
        (proj1["X_full"], proj1["y_full"], "Full"),
        (proj1["X_abl"], proj1["y_abl"], "Ablated"),
        (proj2["X_full"], proj2["y_full"], "Full"),
        (proj2["X_abl"], proj2["y_abl"], "Ablated"),
    ]

    if big_fonts:
        label_fs = 12
        tick_fs = 10
        title_fs = 12
        legend_fs = 10
    else:
        label_fs = 8
        tick_fs = 7
        title_fs = 8
        legend_fs = 6

    label_fs, tick_fs, title_fs, legend_fs = (
        v * font_scale for v in (label_fs, tick_fs, title_fs, legend_fs)
    )

    # shared axis limits within each task
    def _limits(*Xs):
        X = np.concatenate(Xs, axis=0)
        lo, hi = X.min(axis=0), X.max(axis=0)
        pad = 0.05 * (hi - lo)
        return lo - pad, hi + pad

    lims = {
        0: _limits(proj1["X_full"], proj1["X_abl"]),
        1: _limits(proj1["X_full"], proj1["X_abl"]),
        2: _limits(proj2["X_full"], proj2["X_abl"]),
        3: _limits(proj2["X_full"], proj2["X_abl"]),
    }

    for i, (ax, (X, y, cond_title)) in enumerate(zip(axs, panel_specs)):
        for cls in sorted(np.unique(y)):
            cls_int = int(cls)
            pts = X[y == cls]
            color = class_colors.get(cls_int, "gray")

            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                alpha=0.6,
                s=point_size,
                color=color,
                linewidths=0,
                label=f"class {cls_int}",
                depthshade=False,
                rasterized=True,
            )

        lo, hi = lims[i]
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        # stretch x only; rescale zoom to keep y and z lengths
        asp = np.array([4.0 * x_stretch, 4.0, 3.0])
        ax.set_box_aspect(asp, zoom=zoom * np.linalg.norm(asp) / np.linalg.norm([4.0, 4.0, 3.0]))

        # 2x2: column headers on the top row only; 1x4: title on every panel
        if not grid_2x2 or i < 2:
            ax.set_title(cond_title, fontsize=title_fs, fontweight="normal", pad=-2)
        xl, yl, zl = (proj1 if i < 2 else proj2).get(
            "axis_labels", ("t-SNE 1", "t-SNE 2", "t-SNE 3")
        )

        # 1x4: axis labels only on each task's Full panel
        xl = "Decoding axis"
        
        if grid_2x2 or i in (0, 2):
            ax.set_xlabel(xl, fontsize=label_fs, labelpad=-8 if not show_ticks else -10)
            ax.set_ylabel(yl, fontsize=label_fs, labelpad=-8 if not show_ticks else -10)
            ax.set_zlabel(zl, fontsize=label_fs, labelpad=-8 if not show_ticks else -10)
        if show_ticks:
            ax.tick_params(axis="both", labelsize=tick_fs, pad=-4)
            ax.tick_params(axis="z", labelsize=tick_fs, pad=-2)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.tick_params(axis="both", length=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=3))
            axis.pane.set_facecolor("white")
            axis.pane.set_alpha(0.0)
            # soften back-panel gridlines
            if show_grid:
                axis._axinfo["grid"].update(color=(0.0, 0.0, 0.0, 0.1), linewidth=0.3)
            else:
                axis._axinfo["grid"].update(color=(0.0, 0.0, 0.0, 0.0), linewidth=0.0)

    # one camera per task
    def _per_task(v):
        return tuple(v) if isinstance(v, (tuple, list)) else (v, v)

    elevs, azims = _per_task(elev), _per_task(azim)
    task_titles = (task1_title, task2_title)

    for t in range(2):
        ax_full, ax_abl = axs[2 * t], axs[2 * t + 1]
        e, a = elevs[t], azims[t]

        if auto_view:
            X_full, y_full = panel_specs[2 * t][0], panel_specs[2 * t][1]
            best = _best_view_on_axes(ax_full, X_full, y_full)
            e, a = best["elev"], best["azim"]
            if verbose:
                print(
                    f"{task_titles[t]}: best view elev={e:.0f}, azim={a:.0f} "
                    f"(line-separability of Full panel: {best['accuracy']:.3f})"
                )

        ax_full.view_init(elev=e, azim=a)
        ax_abl.view_init(elev=e, azim=a)

        if verbose and not auto_view:
            X_full, y_full = panel_specs[2 * t][0], np.asarray(panel_specs[2 * t][1]).astype(int)
            cls = np.unique(y_full)
            if len(cls) == 2:
                acc, _ = _view_score_on_axes(ax_full, X_full, y_full == cls[0], y_full == cls[1])
                print(
                    f"{task_titles[t]}: view elev={e:.1f}, azim={a:.1f} "
                    f"(line-separability of Full panel: {acc:.3f})"
                )

    handles, labels = axs[0].get_legend_handles_labels()
    labels_caps = [lab.capitalize() for lab in labels]
    fig.legend(
        handles,
        labels_caps,
        fontsize=legend_fs,
        frameon=False,
        loc="lower center",
        ncol=len(labels_caps),
        handletextpad=0.2,
        columnspacing=1.5,
        markerscale=2.5,
        borderpad=0.1,
    )

    if grid_2x2:
        # widen gaps between panels until nothing collides
        adjust = dict(
            left=0.08,
            right=0.99,
            bottom=0.08,
            top=0.94,
            wspace=0.0,
            hspace=0.0,
        )
        fig.subplots_adjust(**adjust)
        clear = _open_gaps_between_3d_panels(
            fig, axs, adjust,
            left_reserved_in=0.02 * fig.get_size_inches()[0] + title_fs / 72 + 0.04,
        )
        if not clear:
            import warnings

            warnings.warn(
                "3D panels still overlap or run off the figure: there is not "
                "enough room at this figsize. Lower zoom and/or x_stretch, "
                "reduce font_scale, or enlarge figsize.",
                stacklevel=2,
            )

        # row labels: task names
        for t in range(2):
            pos = axs[2 * t].get_position()
            fig.text(
                0.02,
                0.5 * (pos.y0 + pos.y1),
                task_titles[t],
                rotation=90,
                ha="center",
                va="center",
                fontsize=title_fs,
            )
    else:
        # headroom for the task titles
        fig.subplots_adjust(
            left=0.03,
            right=0.99,
            bottom=0.12,
            top=0.86,
            wspace=0.0,
        )
        # one task title per Full/Ablated pair
        for t in range(2):
            pos_full = axs[2 * t].get_position()
            pos_abl = axs[2 * t + 1].get_position()
            fig.text(
                0.5 * (pos_full.x0 + pos_abl.x1),
                0.95,
                task_titles[t],
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
            )

    if save_path is not None:
        save_path = Path(save_path)
        suffix = save_path.suffix.lower().replace(".", "")

        if suffix == "":
            suffix = "svg"
            save_path = save_path.with_suffix(".svg")

        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)

        fig.savefig(
            save_path,
            format=suffix,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            transparent=True,
            dpi=900,
        )

    if interactive:
        # link Full/Ablated cameras within each task
        axs[1].shareview(axs[0])
        axs[3].shareview(axs[2])

        readout = fig.text(
            0.99, 0.01, "", ha="right", va="bottom",
            fontsize=9, family="monospace",
        )

        def _show_view(event=None):
            readout.set_text(
                f"{task1_title}: elev={axs[0].elev:.1f}, azim={axs[0].azim:.1f}   |   "
                f"{task2_title}: elev={axs[2].elev:.1f}, azim={axs[2].azim:.1f}"
            )
            fig.canvas.draw_idle()

        _show_view()
        fig.canvas.mpl_connect("motion_notify_event", _show_view)
        fig.canvas.mpl_connect("button_release_event", _show_view)

    return fig, axs


def plot_hidden_class_points_3d_bare(
    X,
    y,
    elev=20,
    azim=45,
    point_size=3.5,
    class_colors=None,
    zoom=1.0,
    x_stretch=1.0,
    figsize=(2.0, 2.0),
    save_path=None,
    dpi=900,
):
    """One 3D point cloud with no axes (for slides).

        X: (n, 3) embedding; y: class labels. Other arguments as in
        plot_hidden_class_points_two_tasks_tsne_3d_1row4.
    
    """
    if class_colors is None:
        class_colors = {
            0: "orange",
            1: "#009E73",
        }

    X = np.asarray(X)
    y = np.asarray(y)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for cls in sorted(np.unique(y)):
        cls_int = int(cls)
        pts = X[y == cls]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            alpha=0.6,
            s=point_size,
            color=class_colors.get(cls_int, "gray"),
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])

    asp = np.array([4.0 * x_stretch, 4.0, 3.0])
    ax.set_box_aspect(asp, zoom=zoom * np.linalg.norm(asp) / np.linalg.norm([4.0, 4.0, 3.0]))
    ax.view_init(elev=elev, azim=azim)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save_path is not None:
        save_path = Path(save_path)
        suffix = save_path.suffix.lower().replace(".", "") or "svg"
        save_path = save_path.with_suffix(f".{suffix}")
        fig.savefig(
            save_path,
            format=suffix,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )

    return fig, ax


def save_bare_panels_two_tasks_3d(
    projections,
    elev,
    azim,
    save_dir,
    prefix="tsne_hidden_points",
    task1_name="dms",
    task2_name="parity",
    **bare_kwargs,
):
    """Save the four Full/Ablated panels as separate axis-free point clouds.
        Returns {"<task>_full": path, "<task>_ablated": path, ...}.
    
    """
    proj1, proj2 = projections
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _per_task(v):
        return tuple(v) if isinstance(v, (tuple, list)) else (v, v)

    elevs, azims = _per_task(elev), _per_task(azim)

    panels = [
        (task1_name, "full", proj1["X_full"], proj1["y_full"], elevs[0], azims[0]),
        (task1_name, "ablated", proj1["X_abl"], proj1["y_abl"], elevs[0], azims[0]),
        (task2_name, "full", proj2["X_full"], proj2["y_full"], elevs[1], azims[1]),
        (task2_name, "ablated", proj2["X_abl"], proj2["y_abl"], elevs[1], azims[1]),
    ]

    ext = bare_kwargs.pop("ext", "svg")
    paths = {}
    for task_name, cond, X, y, e, a in panels:
        path = save_dir / f"{prefix}_{task_name}_{cond}.{ext}"
        plot_hidden_class_points_3d_bare(
            X, y, elev=e, azim=a, save_path=path, **bare_kwargs
        )
        paths[f"{task_name}_{cond}"] = path

    return paths


# Class separability decoding

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def decode_auc_single_output(
    activity_output,
    module_key="hidden",
    cv=5,
    random_state=0,
):
    """Mean CV ROC-AUC decoding the class from final-timestep activity."""
    X, y = extract_final_module_points(
        activity_output,
        module_key=module_key,
    )

    y = np.asarray(y).astype(int)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    auc_scores = cross_val_score(
        clf,
        X,
        y,
        cv=skf,
        scoring="roc_auc",
    )

    acc_scores = cross_val_score(
        clf,
        X,
        y,
        cv=skf,
        scoring="accuracy",
    )

    return {
        "auc_mean_cv": auc_scores.mean(),
        "auc_sd_cv": auc_scores.std(ddof=1),
        "acc_mean_cv": acc_scores.mean(),
        "acc_sd_cv": acc_scores.std(ddof=1),
        "n_samples": len(y),
        "n_features": X.shape[1],
    }

def decode_auc_over_runs_one_N(
    all_outputs,
    N,
    module_key="hidden",
    conditions=("full", "ablated"),
    cv=5,
    random_state=0,
):
    """Decoding AUC per run and condition for one N."""
    rows = []

    for run_id, run_dict in all_outputs.items():
        if N not in run_dict:
            continue

        for condition in conditions:
            if condition not in run_dict[N]:
                continue

            metrics = decode_auc_single_output(
                run_dict[N][condition],
                module_key=module_key,
                cv=cv,
                random_state=random_state,
            )

            metrics["run_id"] = run_id
            metrics["N"] = N
            metrics["condition"] = condition
            metrics["module"] = module_key

            rows.append(metrics)

    return pd.DataFrame(rows)

def summarise_decode_across_runs(df_decode_runs):
    """Mean and SD of decoding AUC across runs."""
    summary = (
        df_decode_runs
        .groupby(["condition", "module", "N"], as_index=False)
        .agg(
            auc_mean=("auc_mean_cv", "mean"),
            auc_sd=("auc_mean_cv", "std"),
            acc_mean=("acc_mean_cv", "mean"),
            acc_sd=("acc_mean_cv", "std"),
            n_runs=("run_id", "nunique"),
        )
    )

    return summary

def decode_auc_over_runs_many_Ns(
    all_outputs,
    Ns=None,
    module_key="hidden",
    conditions=("full", "ablated"),
    cv=5,
    random_state=0,
):
    """Decoding AUC per run x N x condition."""
    rows = []

    for run_id, run_dict in all_outputs.items():

        if Ns is None:
            Ns_use = sorted(run_dict.keys())
        else:
            Ns_use = Ns

        for N in Ns_use:
            if N not in run_dict:
                continue

            for condition in conditions:
                if condition not in run_dict[N]:
                    continue

                metrics = decode_auc_single_output(
                    run_dict[N][condition],
                    module_key=module_key,
                    cv=cv,
                    random_state=random_state,
                )

                metrics["run_id"] = run_id
                metrics["N"] = N
                metrics["condition"] = condition
                metrics["module"] = module_key

                rows.append(metrics)

    return pd.DataFrame(rows)


def summarise_decode_across_runs_over_Ns(df_decode):
    """Mean and SD of decoding AUC across runs for each N and condition."""
    summary = (
        df_decode
        .groupby(["N", "condition", "module"], as_index=False)
        .agg(
            auc_mean=("auc_mean_cv", "mean"),
            auc_sd=("auc_mean_cv", "std"),
            acc_mean=("acc_mean_cv", "mean"),
            acc_sd=("acc_mean_cv", "std"),
            n_runs=("run_id", "nunique"),
            n_samples_mean=("n_samples", "mean"),
            n_features=("n_features", "first"),
        )
        .sort_values(["N", "condition"])
    )

    return summary

def print_auc_for_N(summary_df, task_name, N):
    rows = summary_df[summary_df["N"] == N].copy()

    condition_order = ["full", "ablated"]
    rows["condition"] = pd.Categorical(
        rows["condition"],
        categories=condition_order,
        ordered=True,
    )
    rows = rows.sort_values("condition")

    for _, row in rows.iterrows():
        print(
            f"task={task_name} | {row['condition']} | N={int(row['N'])} | {row['module']}: "
            f"ROC-AUC = {row['auc_mean']:.3f} ± {row['auc_sd']:.3f} SD "
            f"across {int(row['n_runs'])} runs"
        )
        
def plot_decode_auc_two_tasks_over_Ns(
    df_decode_dms,
    df_decode_parity,
    task1_title="DMS",
    task2_title="Parity",
    condition_order=("full", "ablated"),
    condition_labels=None,
    condition_colors=None,
    show_individual_runs=True,
    ylim=(0.4, 1.05),
    linewidth_pt=397.48499,
    fig_height=2.2,
    figsize=None,
    save_path=None,
):
    """Decoding AUC vs N for DMS and Parity (mean, SD and individual runs)."""


    if condition_labels is None:
        condition_labels = {
            "full": "Full",
            "ablated": "CB ablated",
        }

    if condition_colors is None:
        condition_colors = {
            "full": "salmon",
            "ablated": "#fcaca3",
        }

    if figsize is None:
        inches_per_pt = 1 / 72.27
        fig_width = linewidth_pt * inches_per_pt
        figsize = (fig_width, fig_height)

    fig, axs = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )
    axs = axs.flatten()

    label_fs = 8
    tick_fs = 7
    title_fs = 8
    legend_fs = 6

    panel_specs = [
        (axs[0], df_decode_dms, task1_title, True),
        (axs[1], df_decode_parity, task2_title, False),
    ]

    for ax, df_decode, title, show_ylabel in panel_specs:
        for condition in condition_order:
            sub = df_decode[df_decode["condition"] == condition].copy()

            if len(sub) == 0:
                continue

            grouped = (
                sub.groupby("N", as_index=False)
                .agg(
                    auc_mean=("auc_mean_cv", "mean"),
                    auc_sd=("auc_mean_cv", "std"),
                    n_runs=("run_id", "nunique"),
                )
                .sort_values("N")
            )

            color = condition_colors.get(condition, None)
            label = condition_labels.get(condition, condition)

            if show_individual_runs:
                for run_id, run_sub in sub.groupby("run_id"):
                    run_sub = run_sub.sort_values("N")
                    ax.plot(
                        run_sub["N"].values,
                        run_sub["auc_mean_cv"].values,
                        color=color,
                        alpha=0.22,
                        linewidth=0.8,
                        zorder=1,
                    )

            x = grouped["N"].values.astype(float)
            y = grouped["auc_mean"].values.astype(float)
            sd = grouped["auc_sd"].values.astype(float)

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.4,
                markersize=2.5,
                color=color,
                label=label,
                zorder=3,
            )

            ax.fill_between(
                x,
                y - sd,
                y + sd,
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=2,
            )

        ax.axhline(
            0.5,
            linestyle="--",
            linewidth=1,
            color="gray",
            zorder=0,
        )

        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("N", fontsize=label_fs)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.spines[["top", "right"]].set_visible(False)

        if show_ylabel:
            ax.set_ylabel("Cross-validated ROC-AUC", fontsize=label_fs)
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)
            ax.spines["left"].set_visible(False)

    axs[1].legend(
        frameon=False,
        fontsize=legend_fs,
        loc="lower right",
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()

    return fig, axs