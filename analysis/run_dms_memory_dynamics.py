"""DMS memory-axis propagation q(t) = m_{t+1}^T J_t m_t: primary values, robustness
variants and detachment checks. Outputs to results/mechanistic_analysis/dms_memory_propagation/.

Run with: python -m analysis.run_dms_memory_dynamics
"""
import os
import numpy as np
import pandas as pd
import torch

from analysis.model_stepping import RNN_RUNS, CB_RUNS, load_model, manual_step, cb_bias_at
from analysis.dms_dynamics import collect_aligned_dynamics
from analysis.dms_memory_propagation import (
    make_split_trials, unit_directions, delay_transitions, tau_of,
    batched_jacobian, batched_jvp, step_h_next,
)

OUT_DIR = "results/mechanistic_analysis/dms_memory_propagation"
PRIMARY_CSV = "results/mechanistic_analysis/dms_memory_propagation/local_gain_by_transition.csv"
NS = list(range(5, 31))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANTS = {
    "alt_eval_seed_54321": dict(seed=54321, n_batches=32),
    "fewer_trials_256": dict(seed=12345, n_batches=8),
}
VERIFY_NS, VERIFY_STATES = [5, 16, 30], 32


def q_metrics(model, h, x, m_t, m_next):
    """Per-trial q_full, q_det, q_CB (and q_full only for RNN-only)."""
    mt = torch.as_tensor(m_t, device=h.device, dtype=h.dtype)
    mn = torch.as_tensor(m_next, device=h.device, dtype=h.dtype)
    with torch.no_grad():
        J_full = batched_jacobian(model, h, x, "full")
        out = dict(q_full=(J_full @ mt) @ mn)
        if model.use_cb_bias:
            b = cb_bias_at(model, h, x)
            J_det = batched_jacobian(model, h, x, "det", b)
            out["q_det"] = (J_det @ mt) @ mn
            out["q_cb"] = ((J_full - J_det) @ mt) @ mn
    return {k: v.cpu().numpy() for k, v in out.items()}


def collect_variant(seed, n_batches, tag):
    rows = []
    for arch, runs in [("rnn_only", RNN_RUNS), ("cb_rnn", CB_RUNS)]:
        for s, rp in enumerate(runs, start=1):
            for N in NS:
                model = load_model(rp, N, device=DEVICE)
                dir_b, eval_b = make_split_trials(N, 64, n_batches, seed)
                m = unit_directions(collect_aligned_dynamics(model, dir_b, device=DEVICE)["by_t"])
                by = collect_aligned_dynamics(model, eval_b, device=DEVICE)["by_t"]
                for t in delay_transitions(N):
                    r = q_metrics(model, by[t]["hidden"].to(DEVICE), by[t]["x_next"].to(DEVICE), m[t], m[t + 1])
                    row = dict(variant=tag, arch=arch, seed=s, N=N, t_rel=t, tau=tau_of(t, N),
                               n_trials=len(r["q_full"]))
                    row.update({k: float(v.mean()) for k, v in r.items()})
                    rows.append(row)
        print(f"[{tag}] {arch} done", flush=True)
    return pd.DataFrame(rows)


def verify_detachment():
    """Float64 checks on random evaluation states of every CB-RNN seed."""
    rows = []
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        for s, rp in enumerate(CB_RUNS, start=1):
            for N in VERIFY_NS:
                model = load_model(rp, N).double()
                dir_b, eval_b = make_split_trials(N, 64, 32, 12345)
                dir_b = [{**b, "seqs": b["seqs"].double()} for b in dir_b]
                eval_b = [{**b, "seqs": b["seqs"].double()} for b in eval_b]
                m = unit_directions(collect_aligned_dynamics(model, dir_b)["by_t"])
                by = collect_aligned_dynamics(model, eval_b)["by_t"]
                rng = np.random.RandomState(1000 * s + N)
                t = int(rng.randint(0, N - 2))
                idx = rng.choice(by[t]["hidden"].shape[0], size=VERIFY_STATES, replace=False)
                h, x = by[t]["hidden"][idx], by[t]["x_next"][idx]
                mt, mn = torch.as_tensor(m[t]), torch.as_tensor(m[t + 1])
                with torch.no_grad():
                    b = cb_bias_at(model, h, x)
                    st_full = manual_step(model, h, x)
                    st_det = manual_step(model, h, x, b_cb_override=b)
                    J_full = batched_jacobian(model, h, x, "full")
                    J_det = batched_jacobian(model, h, x, "det", b)
                    v = mt.expand_as(h)
                    jvp_full = batched_jvp(model, h, x, v, "full")
                    jvp_det = batched_jvp(model, h, x, v, "det", b)
                    d = 1e-6
                    fd_full = (step_h_next(model, h + d * v, x, "full") - step_h_next(model, h - d * v, x, "full")) / (2 * d)
                    fd_det = (step_h_next(model, h + d * v, x, "det", b) - step_h_next(model, h - d * v, x, "det", b)) / (2 * d)
                    q_full, q_det = (J_full @ mt) @ mn, (J_det @ mt) @ mn
                    q_cb = ((J_full - J_det) @ mt) @ mn
                rel = lambda a, r: float(((a - r).norm(dim=-1) / r.norm(dim=-1)).max())
                rows.append(dict(
                    seed=s, N=N, t_rel=t, n_states=VERIFY_STATES,
                    max_abs_diff_cb_forward=float((st_full["b_cb"] - st_det["b_cb"]).abs().max()),
                    max_abs_diff_h_next=float((st_full["h_next"] - st_det["h_next"]).abs().max()),
                    max_abs_diff_post=float((st_full["post"] - st_det["post"]).abs().max()),
                    max_abs_J_full_minus_J_det=float((J_full - J_det).abs().max()),
                    mean_fro_J_full_minus_J_det=float((J_full - J_det).flatten(1).norm(dim=1).mean()),
                    max_rel_err_jacrev_vs_jvp_full=rel(J_full @ mt, jvp_full),
                    max_rel_err_jacrev_vs_jvp_det=rel(J_det @ mt, jvp_det),
                    max_rel_err_jacrev_vs_fd_full=rel(J_full @ mt, fd_full),
                    max_rel_err_jacrev_vs_fd_det=rel(J_det @ mt, fd_det),
                    max_abs_err_qcb_identity=float((q_cb - (q_full - q_det)).abs().max()),
                    mean_q_cb=float(q_cb.mean()),
                ))
    finally:
        torch.set_default_dtype(old)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("detachment verification", flush=True)
    verify_detachment().to_csv(os.path.join(OUT_DIR, "verify_detachment.csv"), index=False)

    primary = pd.read_csv(PRIMARY_CSV)
    primary = primary[primary.is_delay & primary.N.isin(NS)]
    prim = primary.rename(columns={"p_full": "q_full", "p_det": "q_det", "s_cb": "q_cb"})[
        ["arch", "seed", "N", "t_rel", "tau", "n_trials", "q_full", "q_det", "q_cb", "cos_m_t_m_next"]]
    prim.insert(0, "variant", "primary_eval_seed_12345")
    frames = [prim]
    for tag, cfg in VARIANTS.items():
        frames.append(collect_variant(cfg["seed"], cfg["n_batches"], tag))
    pd.concat(frames, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "dms_memory_dynamics_by_transition.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
