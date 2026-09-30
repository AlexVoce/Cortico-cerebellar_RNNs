"""DMS CB-contribution analysis: per-transition memory-axis metrics (Part 1, used in
Fig 6a-b), propagation checks, legacy reproduction and sanity checks.

Run with: python -m analysis.run_dms_memory_propagation
"""
import os
import json
import time
import contextlib
import numpy as np
import pandas as pd
import torch

from analysis.model_stepping import RNN_RUNS, CB_RUNS, load_model, shared_available_Ns
from analysis.dms_dynamics import collect_aligned_dynamics
from analysis.dms_memory_propagation import (
    make_split_trials, unit_directions, delay_transitions, tau_of,
    transition_metrics, trajectory_arrays, linear_propagation, finite_perturbation,
    check_detached_forward_and_decomposition, check_finite_differences,
)
import analysis.run_dms_mechanism_survey as legacy

OUT_DIR = "results/mechanistic_analysis/dms_memory_propagation"
DATA_DIR = OUT_DIR

BATCH_SIZE = 64
N_BATCHES = 32              # first half direction set, second half evaluation set
EVAL_SEED = 12345
N_MIN = 5
PERT_NS = [7, 11, 16, 21, 25, 30]
EPSILONS = [1e-3, 1e-2, 1e-1, 3e-1]
SANITY_NS = [7, 16, 30]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PER_TRIAL_KEYS_CB = ["g_full", "p_full", "g_det", "p_det", "g_zero", "p_zero", "dg", "c_cb", "s_cb",
                     "g_full_insample", "p_full_insample", "g_det_insample", "p_det_insample",
                     "dg_insample", "s_cb_insample"]
PER_TRIAL_KEYS_RNN = ["g_full", "p_full", "g_full_insample", "p_full_insample"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [dms_cb] {msg}", flush=True)


@contextlib.contextmanager
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _cos(a, b):
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return np.nan
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Part 1: local memory-direction metrics

def collect_local_metrics(Ns):
    rows = []
    for arch, runs in [("rnn_only", RNN_RUNS), ("cb_rnn", CB_RUNS)]:
        keys = PER_TRIAL_KEYS_CB if arch == "cb_rnn" else PER_TRIAL_KEYS_RNN
        for seed_idx, run_path in enumerate(runs, start=1):
            for N in Ns:
                model = load_model(run_path, N, device=DEVICE)
                dir_batches, eval_batches = make_split_trials(N, BATCH_SIZE, N_BATCHES, EVAL_SEED)
                by_dir = collect_aligned_dynamics(model, dir_batches, device=DEVICE)["by_t"]
                by_eval = collect_aligned_dynamics(model, eval_batches, device=DEVICE)["by_t"]
                m_dir = unit_directions(by_dir)
                m_in = unit_directions(by_eval)
                delay = set(delay_transitions(N))

                for t_rel in range(0, N - 1):  # delay transitions + the comparison transition
                    h = by_eval[t_rel]["hidden"].to(DEVICE)
                    x = by_eval[t_rel]["x_next"].to(DEVICE)
                    r = transition_metrics(model, h, x, m_dir[t_rel], m_dir[t_rel + 1],
                                           m_in[t_rel], m_in[t_rel + 1])
                    is_delay = t_rel in delay
                    row = dict(
                        arch=arch, seed=seed_idx, N=N, t_rel=t_rel, is_delay=is_delay,
                        tau=tau_of(t_rel, N) if is_delay else np.nan, n_trials=h.shape[0],
                        cos_m_t_m_next=_cos(m_dir[t_rel], m_dir[t_rel + 1]),
                        cos_m_heldout_insample=_cos(m_dir[t_rel], m_in[t_rel]),
                    )
                    for k in keys:
                        row[k] = float(np.mean(r[k]))
                        row[k + "_trial_sd"] = float(np.std(r[k]))
                    rows.append(row)
            log(f"local metrics: {arch} seed {seed_idx} done")
    return pd.DataFrame(rows)


# Part 2: linearised propagation and finite perturbation

def collect_propagation(Ns):
    lin_rows, fd_rows = [], []
    with float64():
        for arch, runs in [("rnn_only", RNN_RUNS), ("cb_rnn", CB_RUNS)]:
            for seed_idx, run_path in enumerate(runs, start=1):
                for N in Ns:
                    model = load_model(run_path, N, device=DEVICE).double()
                    dir_b, eval_b = make_split_trials(N, BATCH_SIZE, N_BATCHES, EVAL_SEED)
                    dir_b = [{**b, "seqs": b["seqs"].double()} for b in dir_b]
                    eval_b = [{**b, "seqs": b["seqs"].double()} for b in eval_b]
                    m_dir = unit_directions(collect_aligned_dynamics(model, dir_b, device=DEVICE)["by_t"])
                    by_eval = collect_aligned_dynamics(model, eval_b, device=DEVICE)["by_t"]

                    t_rels = delay_transitions(N)            # 0 .. N-3
                    hs, xs = trajectory_arrays(by_eval, t_rels, DEVICE, torch.float64)
                    m0 = m_dir[0]
                    ms = [m_dir[t + 1] for t in t_rels]      # direction at the state reached

                    lin_modes = ["full", "det"] if model.use_cb_bias else ["full"]
                    for mode in lin_modes:
                        norms, projs = linear_propagation(model, hs, xs, m0, ms, mode)
                        for k in range(len(t_rels)):
                            lin_rows.append(dict(
                                arch=arch, seed=seed_idx, N=N, lag=k + 1, mode=mode,
                                norm_mean=float(norms[k].mean()),
                                log_norm_mean=float(np.log(norms[k]).mean()),
                                proj_mean=float(projs[k].mean()),
                                absproj_mean=float(np.abs(projs[k]).mean()),
                            ))

                    fd_modes = ["full", "clamped"] if model.use_cb_bias else ["full"]
                    for mode in fd_modes:
                        for eps in EPSILONS:
                            projs, norms = finite_perturbation(model, hs, xs, m0, ms, eps, mode)
                            for k in range(len(t_rels)):
                                fd_rows.append(dict(
                                    arch=arch, seed=seed_idx, N=N, lag=k + 1, mode=mode, eps=eps,
                                    proj_mean=float(projs[k].mean()),
                                    absproj_mean=float(np.abs(projs[k]).mean()),
                                    norm_mean=float(norms[k].mean()),
                                ))
                log(f"propagation: {arch} seed {seed_idx} done")
    return pd.DataFrame(lin_rows), pd.DataFrame(fd_rows)


# Part 3: reproduce the legacy Jacobian metrics

def reproduce_legacy():
    shared_Ns = shared_available_Ns(RNN_RUNS + CB_RUNS)
    return legacy.collect_jacobian_metrics(shared_Ns)


# Part 4: sanity checks

def collect_sanity():
    rows = []
    with float64():
        for arch, runs in [("rnn_only", RNN_RUNS), ("cb_rnn", CB_RUNS)]:
            for seed_idx, run_path in enumerate(runs, start=1):
                for N in SANITY_NS:
                    model = load_model(run_path, N, device="cpu").double()
                    _, eval_b = make_split_trials(N, BATCH_SIZE, N_BATCHES, EVAL_SEED)
                    eval_b = [{**b, "seqs": b["seqs"].double()} for b in eval_b]
                    by = collect_aligned_dynamics(model, eval_b)["by_t"]
                    rng = np.random.RandomState(EVAL_SEED + seed_idx + N)
                    t_rel = int(rng.randint(0, N - 2))
                    idx = rng.choice(by[t_rel]["hidden"].shape[0], size=16, replace=False)
                    h, x = by[t_rel]["hidden"][idx], by[t_rel]["x_next"][idx]
                    row = dict(arch=arch, seed=seed_idx, N=N, t_rel=t_rel)
                    fd = check_finite_differences(model, h, x, seed=seed_idx)
                    row.update({f"fd_rel_err_{k}": v for k, v in fd.items()})
                    if model.use_cb_bias:
                        row.update(check_detached_forward_and_decomposition(model, h, x))
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    shared_Ns = shared_available_Ns(RNN_RUNS + CB_RUNS)
    Ns = [n for n in shared_Ns if n >= N_MIN]
    config = dict(
        rnn_runs=RNN_RUNS, cb_runs=CB_RUNS, shared_Ns=shared_Ns, Ns_local=Ns,
        Ns_perturbation=PERT_NS, epsilons=EPSILONS, batch_size=BATCH_SIZE, n_batches=N_BATCHES,
        direction_batches="first half", evaluation_batches="second half",
        eval_seed=EVAL_SEED, device=DEVICE, torch_version=torch.__version__,
    )
    with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log(f"shared Ns {shared_Ns}; local-metric Ns {Ns}")

    log("Part 4: sanity checks")
    collect_sanity().to_csv(os.path.join(DATA_DIR, "sanity_checks.csv"), index=False)

    log("Part 1: local memory-direction metrics")
    collect_local_metrics(Ns).to_csv(os.path.join(DATA_DIR, "local_gain_by_transition.csv"), index=False)

    log("Part 2: propagation and finite perturbation")
    lin, fd = collect_propagation(PERT_NS)
    lin.to_csv(os.path.join(DATA_DIR, "propagation_linear.csv"), index=False)
    fd.to_csv(os.path.join(DATA_DIR, "propagation_finite_perturbation.csv"), index=False)

    log("Part 3: reproducing the previous Jacobian phase (its own code/settings)")
    g, p = reproduce_legacy()
    g.to_csv(os.path.join(DATA_DIR, "legacy_reproduction_jacobian_gain.csv"), index=False)
    p.to_csv(os.path.join(DATA_DIR, "legacy_reproduction_jacobian_propagation.csv"), index=False)

    log("done")


if __name__ == "__main__":
    main()
