import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from analysis.ablation import (
    find_available_Ns,
    _resolve_shared_Ns,
    make_fixed_eval_batches,
    load_model_for_run_and_N,
    run_model_on_fixed_batches,
    _get_label_from_labs,
    _get_logits_from_out_heads,
)


# =============================================================================
# Model preparation
# =============================================================================

def freeze_all_but_heads(model):
    """
    Freeze all model parameters except readout/head parameters.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "heads"):
        for head in model.heads:
            for p in head.parameters():
                p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            if "head" in name or "heads" in name or "readout" in name:
                p.requires_grad = True

    return model


def make_cb_disabled_readout_recovery_model(
    cb_run_path,
    N,
    device="cpu",
):
    """
    Load a CB-RNN checkpoint, disable CB online, and freeze all parameters
    except the readout heads.
    """
    model = load_model_for_run_and_N(
        run_path=cb_run_path,
        N=N,
        device=device,
        verify=True,
    )

    if not hasattr(model, "use_cb_bias"):
        raise AttributeError("Model does not have attribute 'use_cb_bias'.")

    model.use_cb_bias = False
    model = freeze_all_but_heads(model)
    model.to(device)
    model.eval()

    return model


# =============================================================================
# Readout-only training
# =============================================================================

def train_readout_only_at_fixed_N(
    model,
    batch_fn,
    eval_n,
    batch_size=64,
    n_train_steps=500,
    lr=1e-2,
    head_idx=0,
    device="cpu",
    print_every=50,
):
    """
    Train only the unfrozen readout/head parameters at a fixed N.
    """
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found. Check freeze_all_but_heads()."
        )

    optimizer = torch.optim.SGD(
        trainable_params,
        lr=lr,
        momentum=0.1,
        nesterov=True,
    )

    losses = []
    accs = []

    for step in range(n_train_steps):
        seqs, labs = batch_fn([eval_n], batch_size)
        seqs = seqs.to(device)
        lbl = _get_label_from_labs(labs, device=device)

        optimizer.zero_grad()

        _, out_heads = model(
            seqs,
            return_timewise=False,
        )

        logits = _get_logits_from_out_heads(out_heads, head_idx=head_idx)
        loss = F.cross_entropy(logits, lbl)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == lbl).float().mean().item()

        losses.append(float(loss.item()))
        accs.append(float(acc))

        if print_every is not None and step % print_every == 0:
            print(f"step {step:4d} | loss={loss.item():.4f} | acc={acc:.4f}")

    return {
        "losses": losses,
        "accs": accs,
        "final_train_acc": float(np.mean(accs[-20:])),
    }


# =============================================================================
# Main recovery analysis
# =============================================================================

def run_cb_disabled_readout_recovery_across_N(
    cb_run_path,
    batch_fn,
    batch_size=64,
    n_train_steps=500,
    lr=1e-2,
    head_idx=0,
    device="cpu",
    eval_batches=50,
    Ns=None,
):
    """
    For each checkpoint N:
      1. Evaluate the intact CB-RNN.
      2. Disable CB and evaluate zero-shot performance.
      3. Train only readout heads with CB disabled.
      4. Re-evaluate recovered CB-disabled performance.

    Full, zero-shot, and recovered evaluations use the same fixed batches.
    """
    if Ns is None:
        Ns = find_available_Ns(cb_run_path)

    rows = []
    run_id = os.path.basename(str(cb_run_path).rstrip("/"))

    for N in Ns:
        print(f"\n===== {run_id} | N={N} =====")

        fixed_eval_batches = make_fixed_eval_batches(
            batch_fn=batch_fn,
            eval_n=N,
            batch_size=batch_size,
            n_batches=eval_batches,
        )

        # Full CB-RNN baseline
        cb_model = load_model_for_run_and_N(
            run_path=cb_run_path,
            N=N,
            device=device,
            verify=True,
        )

        cb_full_acc = run_model_on_fixed_batches(
            model=cb_model,
            fixed_batches=fixed_eval_batches,
            head_idx=head_idx,
            device=device,
            ablate_cb=False,
            return_outputs=False,
            collect_dynamics=False,
        )

        # Same checkpoint, but CB disabled and only heads trainable
        recovery_model = make_cb_disabled_readout_recovery_model(
            cb_run_path=cb_run_path,
            N=N,
            device=device,
        )

        zero_shot_disabled_acc = run_model_on_fixed_batches(
            model=recovery_model,
            fixed_batches=fixed_eval_batches,
            head_idx=head_idx,
            device=device,
            ablate_cb=False,
            return_outputs=False,
            collect_dynamics=False,
        )

        train_stats = train_readout_only_at_fixed_N(
            model=recovery_model,
            batch_fn=batch_fn,
            eval_n=N,
            batch_size=batch_size,
            n_train_steps=n_train_steps,
            lr=lr,
            head_idx=head_idx,
            device=device,
        )

        recovered_disabled_acc = run_model_on_fixed_batches(
            model=recovery_model,
            fixed_batches=fixed_eval_batches,
            head_idx=head_idx,
            device=device,
            ablate_cb=False,
            return_outputs=False,
            collect_dynamics=False,
        )

        rows.append({
            "run_id": run_id,
            "run_path": str(cb_run_path),
            "N": int(N),
            "cb_full_acc": float(cb_full_acc),
            "zero_shot_cb_disabled_acc": float(zero_shot_disabled_acc),
            "readout_recovered_cb_disabled_acc": float(recovered_disabled_acc),
            "final_readout_train_acc": float(train_stats["final_train_acc"]),
            "zero_shot_gap_from_cb": float(cb_full_acc - zero_shot_disabled_acc),
            "recovered_gap_from_cb": float(cb_full_acc - recovered_disabled_acc),
        })

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def run_cb_disabled_readout_recovery_across_multiple_runs(
    cb_run_paths,
    batch_fn,
    batch_size=64,
    n_train_steps=500,
    lr=1e-2,
    head_idx=0,
    device="cpu",
    eval_batches=50,
    Ns=None,
    skip_last=0,
):
    """
    Run CB-disabled readout recovery across multiple runs.

    If Ns is None, uses the shared Ns across all runs.
    """
    shared_Ns = _resolve_shared_Ns(
        run_paths=cb_run_paths,
        Ns=Ns,
        skip_last=skip_last,
    )

    dfs = []

    for i, cb_run_path in enumerate(cb_run_paths):
        run_id = os.path.basename(str(cb_run_path).rstrip("/"))
        print(f"\n[{i + 1}/{len(cb_run_paths)}] {run_id}")

        df_run = run_cb_disabled_readout_recovery_across_N(
            cb_run_path=cb_run_path,
            batch_fn=batch_fn,
            batch_size=batch_size,
            n_train_steps=n_train_steps,
            lr=lr,
            head_idx=head_idx,
            device=device,
            eval_batches=eval_batches,
            Ns=shared_Ns,
        )

        dfs.append(df_run)

    return pd.concat(dfs, ignore_index=True)


def summarize_cb_disabled_readout_recovery_across_runs(df_all):
    """
    Average CB-disabled readout-recovery results across runs.
    """
    return (
        df_all
        .groupby("N", as_index=False)
        .agg(
            cb_full_acc_mean=("cb_full_acc", "mean"),
            cb_full_acc_sem=("cb_full_acc", "sem"),
            zero_shot_cb_disabled_acc_mean=("zero_shot_cb_disabled_acc", "mean"),
            zero_shot_cb_disabled_acc_sem=("zero_shot_cb_disabled_acc", "sem"),
            readout_recovered_cb_disabled_acc_mean=(
                "readout_recovered_cb_disabled_acc", "mean"
            ),
            readout_recovered_cb_disabled_acc_sem=(
                "readout_recovered_cb_disabled_acc", "sem"
            ),
            zero_shot_gap_from_cb_mean=("zero_shot_gap_from_cb", "mean"),
            zero_shot_gap_from_cb_sem=("zero_shot_gap_from_cb", "sem"),
            recovered_gap_from_cb_mean=("recovered_gap_from_cb", "mean"),
            recovered_gap_from_cb_sem=("recovered_gap_from_cb", "sem"),
            n_runs=("run_id", "nunique"),
        )
        .sort_values("N")
        .reset_index(drop=True)
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_cb_disabled_readout_recovery(
    summary_df,
    task_title=None,
    fig_size=(6.4, 3),
    fig_height=0.8,
    save_path=None,
):
    linewidth_pt = 397.48499
    inches_per_pt = 1 / 72.27
    fig_width = linewidth_pt * inches_per_pt

    figsize = (fig_width, fig_height) if fig_size is None else fig_size
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    label_fs = 8
    tick_fs = 7
    title_fs = 8
    legend_fs = 7

    axs[0].plot(
        summary_df["N"],
        summary_df["cb_full_acc_mean"],
        marker="o",
        linewidth=1,
        markersize=2,
        label="CB full model",
        color="salmon",
    )

    axs[0].fill_between(
        summary_df["N"],
        summary_df["cb_full_acc_mean"] - summary_df["cb_full_acc_sem"],
        summary_df["cb_full_acc_mean"] + summary_df["cb_full_acc_sem"],
        alpha=0.3,
        color="salmon",
    )

    axs[0].plot(
        summary_df["N"],
        summary_df["zero_shot_cb_disabled_acc_mean"],
        marker="o",
        linewidth=1,
        markersize=2,
        label="CB disabled zero-shot",
        color="#fcaca3",
    )

    axs[0].fill_between(
        summary_df["N"],
        summary_df["zero_shot_cb_disabled_acc_mean"] - summary_df["zero_shot_cb_disabled_acc_sem"],
        summary_df["zero_shot_cb_disabled_acc_mean"] + summary_df["zero_shot_cb_disabled_acc_sem"],
        alpha=0.3,
        color="#fcaca3",
    )

    axs[0].plot(
        summary_df["N"],
        summary_df["readout_recovered_cb_disabled_acc_mean"],
        marker="o",
        linewidth=1,
        markersize=2,
        label="Readout Training",
        color="green",
    )

    axs[0].fill_between(
        summary_df["N"],
        summary_df["readout_recovered_cb_disabled_acc_mean"] - summary_df["readout_recovered_cb_disabled_acc_sem"],
        summary_df["readout_recovered_cb_disabled_acc_mean"] + summary_df["readout_recovered_cb_disabled_acc_sem"],
        alpha=0.3,
        color="green",
    )

    axs[0].set_xlabel("N", fontsize=label_fs)
    axs[0].set_ylabel("Accuracy", fontsize=label_fs)
    axs[0].set_title(
        "CB-disabled readout recovery" + ("" if task_title is None else f" - {task_title}"),
        fontsize=title_fs,
    )
    axs[0].legend(fontsize=legend_fs, frameon=False,loc="upper right")

    axs[1].plot(
        summary_df["N"],
        summary_df["zero_shot_gap_from_cb_mean"],
        marker="o",
        markersize=2,
        label="Zero-shot gap from CB",
        color="#fcaca3",
    )
    axs[1].fill_between(
        summary_df["N"],
        summary_df["zero_shot_gap_from_cb_mean"] - summary_df["zero_shot_gap_from_cb_sem"],
        summary_df["zero_shot_gap_from_cb_mean"] + summary_df["zero_shot_gap_from_cb_sem"],
        alpha=0.3,
        color="#fcaca3",
    )
    axs[1].plot(
        summary_df["N"],
        summary_df["recovered_gap_from_cb_mean"],
        marker="o",
        markersize=2,
        label="Recovered gap from CB",
        color="green",
    )
    axs[1].fill_between(
        summary_df["N"],
        summary_df["recovered_gap_from_cb_mean"] - summary_df["recovered_gap_from_cb_sem"],
        summary_df["recovered_gap_from_cb_mean"] + summary_df["recovered_gap_from_cb_sem"],
        alpha=0.3,
        color="green",
    )
    axs[1].set_xlabel("N", fontsize=label_fs)
    axs[1].set_ylabel("Accuracy gap", fontsize=label_fs)
    axs[1].set_title(
        "Recovery gap" + ("" if task_title is None else f" - {task_title}"),
        fontsize=title_fs,
    )
    axs[1].legend(fontsize=legend_fs, frameon=False)

    for ax in axs:
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, format="svg", bbox_inches="tight")

    plt.show()

    return fig, axs