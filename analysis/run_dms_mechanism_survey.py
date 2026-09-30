"""Post-hoc DMS mechanism analysis (RNN-only vs CB-RNN, 8 seeds each).
Outputs to results/dms_mechanism_analysis/.

Run with: python -m analysis.run_dms_mechanism_survey
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from analysis.model_stepping import (
    RNN_COLOR, CB_COLOR, HIDDEN_COLOR, CB_SIGNAL_COLOR, RNN_RUNS, CB_RUNS, HIDDEN_SIZES,
    default_device, load_model, make_fixed_dms_trials, shared_available_Ns,
)
from analysis.cb_ablation import find_available_Ns
from analysis.predictive_analysis import _select_Ns
from analysis.dms_dynamics import (
    collect_aligned_dynamics, delta_over_time, alignment_cosine,
    decode_sample_over_time, causal_one_step_cb_effect,
    rollout_accuracy_with_cb_condition,
)
from analysis.jacobian import (
    decompose_full_jacobian, directional_gain, propagate_direction,
)
from analysis.rebuild_model_utils import load_state_dict
from analysis.single_task_analysis_utils import (
    _get_single_n_series, _get_single_acc_series, _load_stats,
    compute_solve_times_for_single_run,
)

OUT_DIR = "results/dms_mechanism_analysis"
DATA_DIR = os.path.join(OUT_DIR, "data")
FIG_DIR = os.path.join(OUT_DIR, "figures")

BATCH_SIZE = 64
N_BATCHES_MECH = 6          # for delay-resolved mechanism metrics
N_BATCHES_ABLATE = 10       # for behavioural/ablation accuracy
MAX_NS_MECH = 12            # representative Ns for delay-resolved metrics
MAX_NS_JAC = 7              # representative Ns for Jacobian/propagation
JAC_TRIAL_SUBSAMPLE = 64    # trials used per Jacobian/propagation evaluation
EVAL_SEED = 12345           # fixed seed for deterministic trial reconstruction

DEVICE = "cpu"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Phase 1: behavioural performance vs N

def final_accuracy_per_N(stats):
    """Last-logged accuracy for each curriculum level N."""
    n_series = _get_single_n_series(stats)
    acc_series = _get_single_acc_series(stats)
    out = {}
    for n, a in zip(n_series, acc_series):
        out[int(n)] = float(a)  # keeps the last value per N
    return out


def collect_behavioural_performance():
    """Per-N final accuracy and epochs to reach 98%."""
    rows = []
    for arch, runs in [("rnn_only", RNN_RUNS), ("cb_rnn", CB_RUNS)]:
        for seed_idx, run_path in enumerate(runs, start=1):
            stats = _load_stats(os.path.join(run_path, "stats.npy"))
            acc_per_N = final_accuracy_per_N(stats)
            solve_times = compute_solve_times_for_single_run(stats, threshold=98.0, patience=1)
            for N, acc in acc_per_N.items():
                rows.append(dict(
                    arch=arch, seed=seed_idx, run_path=run_path, N=N, final_acc=acc,
                    solve_time_epochs=solve_times.get(N, np.nan),
                ))
    return pd.DataFrame(rows)


# Phase 2: full-rollout CB ablation across N

def collect_ablation_across_N(conditions=("full", "zero", "mean", "shuffled")):
    rows = []
    for seed_idx, run_path in enumerate(CB_RUNS, start=1):
        Ns = find_available_Ns(run_path)
        log(f"  [ablation] cb_rnn seed {seed_idx}: {len(Ns)} checkpoints")
        for N in Ns:
            model = load_model(run_path, N, device=DEVICE)
            batches = make_fixed_dms_trials(N, batch_size=BATCH_SIZE, n_batches=N_BATCHES_ABLATE, seed=EVAL_SEED)
            row = dict(seed=seed_idx, run_path=run_path, N=N)
            for cond in conditions:
                row[f"acc_{cond}"] = rollout_accuracy_with_cb_condition(model, batches, cond, device=DEVICE, seed=EVAL_SEED)
            rows.append(row)
    return pd.DataFrame(rows)


# Phase 3: delay-resolved mechanism metrics

def collect_mechanism_metrics(shared_Ns):
    Ns_use = _select_Ns(shared_Ns, max_Ns=MAX_NS_MECH)
    log(f"  [mechanism] representative Ns: {Ns_use}")

    delta_rows, decode_rows, causal_rows = [], [], []

    for arch, runs in [("rnn_only", RNN_RUNS), ("cb_rnn", CB_RUNS)]:
        for seed_idx, run_path in enumerate(runs, start=1):
            for N in Ns_use:
                model = load_model(run_path, N, device=DEVICE)
                batches = make_fixed_dms_trials(N, batch_size=BATCH_SIZE, n_batches=N_BATCHES_MECH, seed=EVAL_SEED)
                aligned = collect_aligned_dynamics(model, batches, device=DEVICE)
                by_t = aligned["by_t"]

                dh = delta_over_time(by_t, key="hidden")
                auc_h = decode_sample_over_time(by_t, key="hidden", seed=EVAL_SEED)

                if model.use_cb_bias:
                    db = delta_over_time(by_t, key="cb_bias")
                    auc_cb = decode_sample_over_time(by_t, key="cb_bias", seed=EVAL_SEED)
                    cos_theta = alignment_cosine(dh["delta"], db["delta"])
                else:
                    db = dict(norm=np.full(len(by_t), np.nan))
                    auc_cb = np.full(len(by_t), np.nan)
                    cos_theta = np.full(len(by_t), np.nan)

                for t_rel in range(len(by_t)):
                    h_size = HIDDEN_SIZES[arch]
                    delta_rows.append(dict(
                        arch=arch, seed=seed_idx, N=N, t_rel=t_rel, N_rel=len(by_t),
                        delta_h_norm=dh["norm"][t_rel], delta_bcb_norm=db["norm"][t_rel],
                        cos_theta=cos_theta[t_rel], hidden_size=h_size,
                        delta_h_per_unit=dh["norm"][t_rel] / (h_size ** 0.5),
                        delta_bcb_per_unit=db["norm"][t_rel] / (h_size ** 0.5),
                    ))
                    decode_rows.append(dict(
                        arch=arch, seed=seed_idx, N=N, t_rel=t_rel, N_rel=len(by_t),
                        auc_hidden=auc_h[t_rel], auc_cb=auc_cb[t_rel],
                    ))

                if model.use_cb_bias:
                    for t_rel in range(len(by_t) - 1):
                        eff = causal_one_step_cb_effect(model, by_t, t_rel, device=DEVICE, seed=EVAL_SEED)
                        if eff is None:
                            continue
                        row = dict(arch=arch, seed=seed_idx, N=N, t_rel=t_rel, N_rel=len(by_t))
                        for cond in ("A_full", "B_zero", "C_mean", "D_shuffled"):
                            row[f"delta_next_{cond}"] = eff[cond]["delta_next_norm"]
                            row[f"proj_{cond}"] = eff[cond].get("proj_on_mt", np.nan)
                        row["cb_memory_effect"] = eff["cb_memory_effect"]
                        causal_rows.append(row)
            log(f"  [mechanism] {arch} seed {seed_idx} done")

    return pd.DataFrame(delta_rows), pd.DataFrame(decode_rows), pd.DataFrame(causal_rows)


# Phase 4: Jacobian gain and full-delay propagation

def _subsample_trials(by_t_entry, k, seed):
    n = by_t_entry["hidden"].shape[0]
    if n <= k:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return rng.choice(n, size=k, replace=False)


def collect_jacobian_metrics(shared_Ns):
    Ns_use = _select_Ns(shared_Ns, max_Ns=MAX_NS_JAC)
    log(f"  [jacobian] representative Ns: {Ns_use}")

    gain_rows, prop_rows = [], []

    for arch, runs, modes in [
        ("rnn_only", RNN_RUNS, [("full", False)]),
        ("cb_rnn", CB_RUNS, [("full", False), ("cb_zero", True)]),
    ]:
        for seed_idx, run_path in enumerate(runs, start=1):
            for N in Ns_use:
                model = load_model(run_path, N, device=DEVICE)
                batches = make_fixed_dms_trials(N, batch_size=BATCH_SIZE, n_batches=N_BATCHES_MECH, seed=EVAL_SEED)
                aligned = collect_aligned_dynamics(model, batches, device=DEVICE)
                by_t = aligned["by_t"]
                N_rel = len(by_t)
                if N_rel < 3:
                    continue

                dh = delta_over_time(by_t, key="hidden")

                # early / mid / late delay timepoints
                t_points = sorted(set([1, N_rel // 2, max(N_rel - 2, 1)]))
                t_points = [t for t in t_points if 0 <= t < N_rel - 1]

                for mode_name, force_no_cb in modes:
                    for t_rel in t_points:
                        d = dh["delta"][t_rel]
                        if np.any(np.isnan(d)) or np.linalg.norm(d) < 1e-8:
                            continue
                        m_t = torch.tensor(d / np.linalg.norm(d), dtype=torch.float32, device=DEVICE)

                        idx = _subsample_trials(by_t[t_rel], JAC_TRIAL_SUBSAMPLE, EVAL_SEED)
                        h_t = by_t[t_rel]["hidden"][idx].to(DEVICE)
                        x_next = by_t[t_rel]["x_next"]
                        if x_next is None:
                            continue
                        x_next = x_next[idx].to(DEVICE)

                        if mode_name == "full":
                            J_full, _, _, err = decompose_full_jacobian(model, h_t, x_next)
                            J = J_full
                        else:
                            from analysis.jacobian import batched_step_jacobian
                            J = batched_step_jacobian(model, h_t, x_next, force_no_cb=True)

                        g, _ = directional_gain(J, m_t)
                        gain_rows.append(dict(
                            arch=arch, seed=seed_idx, N=N, N_rel=N_rel, t_rel=t_rel,
                            mode=mode_name, gain_mean=float(g.mean()), gain_sem=float(g.std() / (len(g) ** 0.5)),
                        ))

                # full-delay propagation: sample -> comparison
                d0 = dh["delta"][0]
                if not (np.any(np.isnan(d0)) or np.linalg.norm(d0) < 1e-8):
                    m0 = np.linalg.norm(d0)
                    m0_vec = torch.tensor(d0 / m0, dtype=torch.float32, device=DEVICE)
                    for mode_name, force_no_cb in modes:
                        idx = _subsample_trials(by_t[0], JAC_TRIAL_SUBSAMPLE, EVAL_SEED)
                        h_seq, x_seq = [], []
                        for t in range(0, N_rel - 1):
                            h_seq.append(by_t[t]["hidden"][idx].to(DEVICE))
                            x_next_t = by_t[t]["x_next"]
                            if x_next_t is None:
                                break
                            x_seq.append(x_next_t[idx].to(DEVICE))
                        if len(x_seq) < len(h_seq):
                            h_seq = h_seq[:len(x_seq)]
                        if len(h_seq) == 0:
                            continue
                        log_growth, _ = propagate_direction(model, h_seq, x_seq, m0_vec, force_no_cb=force_no_cb)
                        growth = torch.exp(log_growth).cpu().numpy()
                        prop_rows.append(dict(
                            arch=arch, seed=seed_idx, N=N, N_rel=N_rel, mode=mode_name,
                            growth_mean=float(np.mean(growth)), growth_sem=float(np.std(growth) / (len(growth) ** 0.5)),
                        ))
            log(f"  [jacobian] {arch} seed {seed_idx} done")

    return pd.DataFrame(gain_rows), pd.DataFrame(prop_rows)


# Phase 5: recurrent-weight drift across N

def collect_weight_drift():
    """Recurrent-weight drift from initialisation; compare delta_Wh_per_element across architectures."""
    rows = []
    for arch, runs in [("rnn_only", RNN_RUNS), ("cb_rnn", CB_RUNS)]:
        for seed_idx, run_path in enumerate(runs, start=1):
            h_size = HIDDEN_SIZES[arch]
            Ns = find_available_Ns(run_path)
            sd_init = load_state_dict(run_path, Ns[0])
            W_init = sd_init["hh.weight"].numpy().astype(float)
            for N in Ns:
                sd = load_state_dict(run_path, N)
                W = sd["hh.weight"].numpy().astype(float)
                fro = float(np.linalg.norm(W - W_init))
                rows.append(dict(
                    arch=arch, seed=seed_idx, N=N,
                    delta_Wh_fro_from_init=fro,
                    delta_Wh_per_element=fro / h_size,
                    Wh_fro_norm=float(np.linalg.norm(W)),
                ))
    return pd.DataFrame(rows)


# Main

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    log(f"Device: {DEVICE}")
    log("Phase 1: behavioural performance vs N")
    df_behav = collect_behavioural_performance()
    df_behav.to_csv(os.path.join(DATA_DIR, "behavioural_performance.csv"), index=False)

    log("Phase 2: full-rollout CB ablation across N")
    df_ablate = collect_ablation_across_N()
    df_ablate.to_csv(os.path.join(DATA_DIR, "cb_ablation_across_N.csv"), index=False)

    log("Computing shared checkpoint Ns across all 16 matched runs")
    shared_Ns = shared_available_Ns(RNN_RUNS + CB_RUNS)
    with open(os.path.join(DATA_DIR, "shared_Ns.json"), "w") as f:
        json.dump(shared_Ns, f)
    log(f"  {len(shared_Ns)} shared Ns: {shared_Ns}")

    log("Phase 3: delay-resolved mechanism metrics")
    df_delta, df_decode, df_causal = collect_mechanism_metrics(shared_Ns)
    df_delta.to_csv(os.path.join(DATA_DIR, "delta_h_cb_over_time.csv"), index=False)
    df_decode.to_csv(os.path.join(DATA_DIR, "decode_over_time.csv"), index=False)
    df_causal.to_csv(os.path.join(DATA_DIR, "causal_one_step.csv"), index=False)

    log("Phase 4: Jacobian gain + full-delay propagation")
    df_gain, df_prop = collect_jacobian_metrics(shared_Ns)
    df_gain.to_csv(os.path.join(DATA_DIR, "jacobian_gain.csv"), index=False)
    df_prop.to_csv(os.path.join(DATA_DIR, "jacobian_propagation.csv"), index=False)

    log("Phase 5: intrinsic recurrent-weight drift")
    df_drift = collect_weight_drift()
    df_drift.to_csv(os.path.join(DATA_DIR, "weight_drift.csv"), index=False)

    log("All data collection complete. Saved to " + DATA_DIR)
    return dict(
        behav=df_behav, ablate=df_ablate, delta=df_delta, decode=df_decode,
        causal=df_causal, gain=df_gain, prop=df_prop, drift=df_drift,
        shared_Ns=shared_Ns,
    )


if __name__ == "__main__":
    main()
