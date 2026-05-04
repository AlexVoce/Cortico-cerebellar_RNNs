import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rebuild_model_utils import load_run_config, load_state_dict, build_model_from_config_and_state, find_available_Ns

def freeze_all_but_heads(model):
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
    Load the CB model checkpoint at N, then disable CB online by setting
    use_cb_bias=False. Freeze everything except the heads.
    """
    cfg = load_run_config(cb_run_path)
    sd = load_state_dict(cb_run_path, N=N)

    model = build_model_from_config_and_state(
        cfg=cfg,
        state_dict=sd,
        device=device,
    )

    if not hasattr(model, "use_cb_bias"):
        raise AttributeError("Model does not have attribute 'use_cb_bias'")

    model.use_cb_bias = False
    model = freeze_all_but_heads(model)
    model.to(device)
    model.eval()

    return model
def evaluate_model_at_fixed_N(
    model,
    batch_fn,
    eval_n,
    batch_size=64,
    n_batches=50,
    head_idx=0,
    device="cpu",
):
    model.eval()
    accs = []

    with torch.no_grad():
        for _ in range(n_batches):
            seqs, labs = batch_fn([eval_n], batch_size)
            seqs = seqs.to(device)
            lbl = labs[-1].to(device)

            _, out_heads = model(
                seqs,
                return_timewise=False,
            )

            logits = out_heads[head_idx]
            preds = logits.argmax(dim=1)
            acc = (preds == lbl).float().mean().item()
            accs.append(acc)

    return float(np.mean(accs))
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
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=lr,momentum=0.1,nesterov=True)

    losses = []
    accs = []

    for step in range(n_train_steps):
        seqs, labs = batch_fn([eval_n], batch_size)
        seqs = seqs.to(device)
        lbl = labs[-1].to(device)

        optimizer.zero_grad()

        _, out_heads = model(
            seqs,
            return_timewise=False,
        )

        logits = out_heads[head_idx]
        loss = F.cross_entropy(logits, lbl)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == lbl).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if print_every is not None and step % print_every == 0:
            print(f"step {step:4d} | loss={loss.item():.4f} | acc={acc:.4f}")

    return {
        "losses": losses,
        "accs": accs,
        "final_train_acc": float(np.mean(accs[-20:])),
    }

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
    if Ns is None:
        Ns = find_available_Ns(cb_run_path)

    cfg = load_run_config(cb_run_path)
    rows = []

    for N in Ns:
        print(f"\n===== N={N} =====")

        # Full CB model baseline
        cb_sd = load_state_dict(cb_run_path, N=N)
        cb_model = build_model_from_config_and_state(
            cfg=cfg,
            state_dict=cb_sd,
            device=device,
        )

        cb_full_acc = evaluate_model_at_fixed_N(
            model=cb_model,
            batch_fn=batch_fn,
            eval_n=N,
            batch_size=batch_size,
            n_batches=eval_batches,
            head_idx=head_idx,
            device=device,
        )

        # Same model, CB disabled, only heads trainable
        recovery_model = make_cb_disabled_readout_recovery_model(
            cb_run_path=cb_run_path,
            N=N,
            device=device,
        )

        zero_shot_disabled_acc = evaluate_model_at_fixed_N(
            model=recovery_model,
            batch_fn=batch_fn,
            eval_n=N,
            batch_size=batch_size,
            n_batches=eval_batches,
            head_idx=head_idx,
            device=device,
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

        recovered_disabled_acc = evaluate_model_at_fixed_N(
            model=recovery_model,
            batch_fn=batch_fn,
            eval_n=N,
            batch_size=batch_size,
            n_batches=eval_batches,
            head_idx=head_idx,
            device=device,
        )

        rows.append({
            "run_id": os.path.basename(str(cb_run_path).rstrip("/")),
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
):
    dfs = []

    for i, cb_run_path in enumerate(cb_run_paths):
        run_id = os.path.basename(str(cb_run_path).rstrip("/"))
        print(f"\n[{i+1}/{len(cb_run_paths)}] {run_id}")

        df_run = run_cb_disabled_readout_recovery_across_N(
            cb_run_path=cb_run_path,
            batch_fn=batch_fn,
            batch_size=batch_size,
            n_train_steps=n_train_steps,
            lr=lr,
            head_idx=head_idx,
            device=device,
            eval_batches=eval_batches,
            Ns=Ns,
        ).copy()

        df_run["run_id"] = run_id
        dfs.append(df_run)

    return pd.concat(dfs, ignore_index=True)
def summarize_cb_disabled_readout_recovery_across_runs(df_all):
    return (
        df_all
        .groupby("N", as_index=False)
        .agg(
            cb_full_acc_mean=("cb_full_acc", "mean"),
            cb_full_acc_sem=("cb_full_acc", "sem"),
            zero_shot_cb_disabled_acc_mean=("zero_shot_cb_disabled_acc", "mean"),
            zero_shot_cb_disabled_acc_sem=("zero_shot_cb_disabled_acc", "sem"),
            readout_recovered_cb_disabled_acc_mean=("readout_recovered_cb_disabled_acc", "mean"),
            readout_recovered_cb_disabled_acc_sem=("readout_recovered_cb_disabled_acc", "sem"),
            zero_shot_gap_from_cb_mean=("zero_shot_gap_from_cb", "mean"),
            zero_shot_gap_from_cb_sem=("zero_shot_gap_from_cb", "sem"),
            recovered_gap_from_cb_mean=("recovered_gap_from_cb", "mean"),
            recovered_gap_from_cb_sem=("recovered_gap_from_cb", "sem"),
            n_runs=("run_id", "nunique"),
        )
        .sort_values("N")
        .reset_index(drop=True)
    )
def plot_cb_disabled_readout_recovery(summary_df):

    linewidth_pt = 397.48499
    inches_per_pt = 1 / 72.27
    fig_width = linewidth_pt * inches_per_pt
    fig, axs = plt.subplots(1, 2, figsize=(fig_width,3))

    label_fs = 9
    tick_fs = 8
    title_fs = 9
    legend_fs = 8

    axs[0].errorbar(
        summary_df["N"],
        summary_df["cb_full_acc_mean"],
        yerr=summary_df["cb_full_acc_sem"],
        marker="o",
        markersize=2,
        capsize=2,
        label="CB full model",
        color="salmon"
    )
    axs[0].errorbar(
        summary_df["N"],
        summary_df["zero_shot_cb_disabled_acc_mean"],
        yerr=summary_df["zero_shot_cb_disabled_acc_sem"],
        marker="o",
        markersize=2,
        capsize=2,
        label="CB Ablated zero-shot",
        color="orange"
    )
    axs[0].errorbar(
        summary_df["N"],
        summary_df["readout_recovered_cb_disabled_acc_mean"],
        yerr=summary_df["readout_recovered_cb_disabled_acc_sem"],
        marker="o",
        markersize=2,
        capsize=2,
        label="CB disabled + readout recovery",
        color="green"
    )
    axs[0].set_xlabel("N", fontsize=label_fs)
    axs[0].set_ylabel("Accuracy", fontsize=label_fs)
    axs[0].set_title("CB-disabled readout recovery", fontsize=title_fs)
    axs[0].legend(fontsize=legend_fs)

    axs[1].errorbar(
        summary_df["N"],
        summary_df["zero_shot_gap_from_cb_mean"],
        yerr=summary_df["zero_shot_gap_from_cb_sem"],
        marker="o",
        markersize=2,
        capsize=2,
        label="Zero-shot gap from CB",
        color="orange"
    )
    axs[1].errorbar(
        summary_df["N"],
        summary_df["recovered_gap_from_cb_mean"],
        yerr=summary_df["recovered_gap_from_cb_sem"],
        marker="o",
        markersize=2,
        capsize=2,
        label="Recovered gap from CB",
        color="green"
    )
    axs[1].axhline(0, linestyle="--", linewidth=1)
    axs[1].set_xlabel("N", fontsize=label_fs)
    axs[1].set_ylabel("Accuracy gap", fontsize=label_fs)
    axs[1].set_title("How much is recovered by head-only training?", fontsize=title_fs)
    axs[1].legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()