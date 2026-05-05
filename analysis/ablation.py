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


# =============================================================================
# Checkpoint / N utilities
# =============================================================================

def find_available_Ns(run_path):
    """
    Return sorted checkpoint Ns from files named like:
        rnn_N12_N12
        rnn_N12_N12.pt
    """
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
    """
    Resolve Ns for one run. If Ns is None, infer from checkpoint files,
    falling back to stats.npy.
    """
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


# =============================================================================
# Batch / model loading
# =============================================================================

def make_fixed_eval_batches(
    batch_fn,
    eval_n,
    batch_size=64,
    n_batches=20,
):
    """
    Generate fixed evaluation batches for one N.
    These can be reused across full vs ablated models and across runs.
    """
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


def load_model_for_run_and_N(run_path, N, device="cpu", verify=True):
    """
    Rebuild and strictly load a model checkpoint for a given run and N.

    Important: strict loading is required because earlier reconstruction was
    silently leaving most weights random.
    """
    cfg = load_run_config(run_path)
    sd = load_state_dict(run_path, N=N)

    model = build_model_from_config_and_state(
        cfg=cfg,
        state_dict=sd,
        device=device,
    )

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    if verify:
        model_sd = model.state_dict()
        bad = []

        for k in sd:
            if k not in model_sd:
                bad.append((k, "missing_after_load"))
                continue

            diff = (
                model_sd[k].detach().cpu()
                - sd[k].detach().cpu()
            ).abs().max().item()

            if diff > 1e-7:
                bad.append((k, diff))

        if len(bad) > 0:
            raise RuntimeError(
                "Loaded weights do not match checkpoint:\n"
                + "\n".join([f"{k}: {v}" for k, v in bad[:20]])
            )

    return model


# =============================================================================
# Output helpers
# =============================================================================

def _get_label_from_labs(labs, device="cpu", label_idx=-1):
    """
    Extract labels from batch_fn output.

    Your current generators return labs as a one-element list, so labs[-1]
    is equivalent to labs[0].
    """
    if isinstance(labs, (list, tuple)):
        lbl = labs[label_idx]
    else:
        lbl = labs

    return lbl.long().to(device)


def _get_logits_from_out_heads(out_heads, head_idx=0):
    """
    Extract logits from output head object.

    Handles:
      - list/tuple: out_heads[head_idx]
      - tensor: out_heads
      - dict: tries head_idx, str(head_idx), common keys, then first tensor
    """
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


# =============================================================================
# Core evaluation runner
# =============================================================================
def run_model_on_fixed_batches(
    model,
    fixed_batches,
    head_idx=0,
    device="cpu",
    ablate_cb=False,
    return_outputs=False,
    collect_dynamics=False,
):
    """
    Core model evaluation function.

    If return_outputs=False:
        returns float accuracy.

    If return_outputs=True:
        returns dict with logits, labels, and optionally dynamics.

    This function supports both:
      1. accuracy-only CB ablation analyses
      2. output/dynamics collection for t-SNE or decoding analyses
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

                    # Store batch labels alongside the dynamics.
                    # This is needed for t-SNE / class-separation analyses.
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


# =============================================================================
# CB ablation evaluation
# =============================================================================

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
    """
    Compare full vs CB-ablated model across many runs.

    Uses shared fixed batches for each N across all runs.
    """
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


# Backwards-compatible alias for your older t-SNE/evaluation call.
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


# =============================================================================
# RNN-only evaluation
# =============================================================================

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
    """
    Evaluate RNN-only model across Ns.

    This does not attempt CB ablation.
    """
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


# =============================================================================
# Plot: CB ablation accuracy
# =============================================================================

def plot_cb_ablation_two_panel(
    df_dms_ablate,
    df_dms_rnn,
    df_parity_ablate,
    df_parity_rnn,
    save_path=None,
    linewidth_pt=397.48499,
    fig_height=1.45,
    show_sem=True,
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
            "rnn": "cornflowerblue",
        }

    inches_per_pt = 1 / 72.27
    plot_block_frac = 0.81
    fig_width = linewidth_pt * inches_per_pt * plot_block_frac

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
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

        ax.plot(
            x_rnn,
            y_rnn,
            color=colors["rnn"],
            lw=1,
            marker="o",
            ms=1.2,
            label="RNN-only",
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

            if "acc_full_sem" in df_rnn:
                sem = df_rnn["acc_full_sem"].to_numpy()
                ax.fill_between(
                    x_rnn,
                    y_rnn - sem,
                    y_rnn + sem,
                    color=colors["rnn"],
                    alpha=0.18,
                    linewidth=0,
                )

        ax.set_title(title, fontsize=9, fontweight="normal")
        ax.set_xlabel("Task difficulty (N)", fontsize=8)
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

# =============================================================================
# t-SNE utilities for hidden-state class separation
# =============================================================================

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator

# extract_final_module_points — replace the entire function body
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
    """
    Plot full vs CB-ablated t-SNE projections across multiple Ns.

    Layout:
        rows = Ns
        columns = Full / CB ablated
    """
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
    fig_height=1.0,
    save_path=None,
    class_colors=None,
):
    """
    One-row four-panel t-SNE figure:

        DMS full | DMS ablated | Parity full | Parity ablated

    Full and ablated are jointly embedded within each task, so each pair is
    directly comparable. DMS and Parity are separate t-SNE spaces.
    """
    if class_colors is None:
        class_colors = {
            0: "orange",  # muted magenta
            1: "#009E73",  # teal green
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
                s=2.4,
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
        # ax.spines["left"].set_visible(False)

    # fig.text(
    #     0.28,
    #     0.98,
    #     f"{task1_title} (N={N_task1})",
    #     ha="center",
    #     va="top",
    #     fontsize=8,
    # )

    # fig.text(
    #     0.73,
    #     0.98,
    #     f"{task2_title} (N={N_task2})",
    #     ha="center",
    #     va="top",
    #     fontsize=8,
    # )

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
        )

    return fig, axs

### Class Separability Decoding

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
    """
    Decode binary class from final-timestep activity for one activity_output.

    Returns mean CV ROC-AUC for this one run/N/condition.
    """
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
    """
    Decode class separability over runs for one N.

    Each run contributes one decoding score per condition.
    """
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
    """
    Summarise decoding scores across runs.

    Reports mean ± SD across runs, not across CV folds.
    """
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
