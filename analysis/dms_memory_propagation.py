"""Delay-resolved CB contribution to DMS memory-direction dynamics.

Jacobians of h_{t+1} w.r.t. h_t at the same state:
    full: CB output and its derivative included
    det:  CB output kept in the forward pass, derivative through CB removed
    zero: CB output removed from the forward pass
"""
import numpy as np
import torch

from analysis.model_stepping import manual_step, cb_bias_at, make_fixed_dms_trials
from analysis.dms_dynamics import collect_aligned_dynamics, delta_over_time


# One-step maps

def step_h_next(model, h, x, mode, b_const=None):
    """h_{t+1} for one CB condition. mode: 'full' | 'det' | 'zero'.
    'det' requires b_const = the real CB output at (h, x)."""
    if mode == "full":
        return manual_step(model, h, x)["h_next"]
    if mode == "det":
        return manual_step(model, h, x, b_cb_override=b_const)["h_next"]
    if mode == "zero":
        return manual_step(model, h, x, force_no_cb=True)["h_next"]
    raise ValueError(mode)


def batched_jacobian(model, h, x, mode, b_const=None):
    """d h_{t+1} / d h_t at each trial's own state. Returns [B, H, H]."""
    if mode == "det":
        def f(h_i, x_i, b_i):
            return step_h_next(model, h_i[None], x_i[None], "det", b_i[None])[0]
        return torch.func.vmap(torch.func.jacrev(f, argnums=0))(h, x, b_const)

    def f(h_i, x_i):
        return step_h_next(model, h_i[None], x_i[None], mode)[0]
    return torch.func.vmap(torch.func.jacrev(f, argnums=0))(h, x)


def batched_jvp(model, h, x, v, mode, b_const=None):
    """J_t v for each trial via forward-mode AD (no H x H matrix formed)."""
    if mode == "det":
        def f(h_i, x_i, v_i, b_i):
            g = lambda hh: step_h_next(model, hh[None], x_i[None], "det", b_i[None])[0]
            return torch.func.jvp(g, (h_i,), (v_i,))[1]
        return torch.func.vmap(f)(h, x, v, b_const)

    def f(h_i, x_i, v_i):
        g = lambda hh: step_h_next(model, hh[None], x_i[None], mode)[0]
        return torch.func.jvp(g, (h_i,), (v_i,))[1]
    return torch.func.vmap(f)(h, x, v)


# Trials and memory directions

def make_split_trials(N, batch_size, n_batches, seed):
    """Fixed trials split by batch: first half for directions, second half for evaluation."""
    batches = make_fixed_dms_trials(N, batch_size=batch_size, n_batches=n_batches, seed=seed)
    half = n_batches // 2
    return batches[:half], batches[half:]


def unit_directions(by_t):
    """Unit memory directions m_t (mean h | sample=0 minus mean h | sample=1)."""
    d = delta_over_time(by_t, key="hidden")["delta"]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 1e-8, d / n, np.nan)
    return m


def delay_transitions(N):
    """t_rel of delay transitions, i.e. 0..N-3."""
    return list(range(0, N - 2))


def tau_of(t_rel, N):
    """Normalised delay position: 0 = first delay step, 1 = last."""
    n_delay = N - 2
    return t_rel / (n_delay - 1) if n_delay > 1 else np.nan


# Local metrics at one transition

def transition_metrics(model, h, x_next, m_t, m_next, m_t_insample=None, m_next_insample=None):
    """Per-trial local gain/propagation metrics for one transition. Returns dict of [B] arrays."""
    dev, dt = h.device, h.dtype
    mt = torch.as_tensor(m_t, device=dev, dtype=dt)
    mn = torch.as_tensor(m_next, device=dev, dtype=dt)

    out = {}
    with torch.no_grad():
        J_full = batched_jacobian(model, h, x_next, "full")
        Jm_full = J_full @ mt
        out["g_full"] = Jm_full.norm(dim=-1)
        out["p_full"] = Jm_full @ mn

        if model.use_cb_bias:
            b_real = cb_bias_at(model, h, x_next)
            J_det = batched_jacobian(model, h, x_next, "det", b_real)
            J_zero = batched_jacobian(model, h, x_next, "zero")
            Jm_det, Jm_zero = J_det @ mt, J_zero @ mt
            Km = Jm_full - Jm_det
            out["g_det"] = Jm_det.norm(dim=-1)
            out["p_det"] = Jm_det @ mn
            out["g_zero"] = Jm_zero.norm(dim=-1)
            out["p_zero"] = Jm_zero @ mn
            out["dg"] = out["g_full"] - out["g_det"]
            out["c_cb"] = Km.norm(dim=-1)
            out["s_cb"] = Km @ mn

        if m_t_insample is not None:
            mti = torch.as_tensor(m_t_insample, device=dev, dtype=dt)
            mni = torch.as_tensor(m_next_insample, device=dev, dtype=dt)
            Jm = J_full @ mti
            out["g_full_insample"] = Jm.norm(dim=-1)
            out["p_full_insample"] = Jm @ mni
            if model.use_cb_bias:
                Jmd = J_det @ mti
                out["g_det_insample"] = Jmd.norm(dim=-1)
                out["p_det_insample"] = Jmd @ mni
                out["dg_insample"] = out["g_full_insample"] - out["g_det_insample"]
                out["s_cb_insample"] = out["p_full_insample"] - out["p_det_insample"]

    return {k: v.detach().cpu().numpy() for k, v in out.items()}


# Propagation and finite perturbation over the delay

def trajectory_arrays(by_t, t_rels, device, dtype):
    """Stack h and x_next for the given t_rels."""
    hs = [by_t[t]["hidden"].to(device=device, dtype=dtype) for t in t_rels]
    xs = [by_t[t]["x_next"].to(device=device, dtype=dtype) for t in t_rels]
    return hs, xs


def linear_propagation(model, hs, xs, m0, ms, mode):
    """Linearised propagation of m0 through the delay. Returns (norm [K,B], proj [K,B])."""
    B = hs[0].shape[0]
    v = torch.as_tensor(m0, device=hs[0].device, dtype=hs[0].dtype).expand(B, -1).clone()
    norms, projs = [], []
    with torch.no_grad():
        for k, (h, x) in enumerate(zip(hs, xs)):
            b = cb_bias_at(model, h, x) if mode == "det" else None
            v = batched_jvp(model, h, x, v, mode, b)
            mk = torch.as_tensor(ms[k], device=v.device, dtype=v.dtype)
            norms.append(v.norm(dim=-1))
            projs.append(v @ mk)
    return torch.stack(norms).cpu().numpy(), torch.stack(projs).cpu().numpy()


def finite_perturbation(model, hs, xs, m0, ms, eps, mode):
    """Finite-perturbation counterpart of linear_propagation. Returns (proj [K,B], norm [K,B])."""
    B = hs[0].shape[0]
    dev, dt = hs[0].device, hs[0].dtype
    h_pert = hs[0] + eps * torch.as_tensor(m0, device=dev, dtype=dt)
    projs, norms = [], []
    with torch.no_grad():
        for k, (h_ref, x) in enumerate(zip(hs, xs)):
            if mode == "clamped":
                b_ref = cb_bias_at(model, h_ref, x)
                h_pert = manual_step(model, h_pert, x, b_cb_override=b_ref)["h_next"]
            else:
                h_pert = manual_step(model, h_pert, x)["h_next"]
            h_ref_next = manual_step(model, h_ref, x)["h_next"]
            d = (h_pert - h_ref_next) / eps
            mk = torch.as_tensor(ms[k], device=dev, dtype=dt)
            projs.append(d @ mk)
            norms.append(d.norm(dim=-1))
    return torch.stack(projs).cpu().numpy(), torch.stack(norms).cpu().numpy()


# Sanity checks

def check_detached_forward_and_decomposition(model, h, x):
    """Check the detached forward pass and Jacobian decomposition. Returns max-abs errors."""
    alpha = 1.0 / model.tau
    with torch.no_grad():
        b = cb_bias_at(model, h, x)
        hn_full = step_h_next(model, h, x, "full")
        hn_det = step_h_next(model, h, x, "det", b)
        J_full = batched_jacobian(model, h, x, "full")
        J_det = batched_jacobian(model, h, x, "det", b)

        post = manual_step(model, h, x)["post"]
        slope = model.afunc.negative_slope
        D = torch.where(post > 0, torch.ones_like(post), torch.full_like(post, slope))
        H = h.shape[1]
        eye = torch.eye(H, device=h.device, dtype=h.dtype)
        J_det_analytic = (1 - alpha) * eye + alpha * D[:, :, None] * model.hh.weight[None]

        def cbf(h_i, x_i):
            return cb_bias_at(model, h_i[None], x_i[None])[0]
        J_cb = torch.func.vmap(torch.func.jacrev(cbf))(h, x)
        K_analytic = alpha * D[:, :, None] * J_cb

    return dict(
        forward_max_abs_diff=float((hn_full - hn_det).abs().max()),
        J_full_minus_J_det_max_abs=float((J_full - J_det).abs().max()),
        J_det_vs_analytic_max_abs_err=float((J_det - J_det_analytic).abs().max()),
        K_vs_analytic_max_abs_err=float(((J_full - J_det) - K_analytic).abs().max()),
    )


def check_finite_differences(model, h, x, n_dirs=5, delta=1e-5, seed=0):
    """Finite differences vs J v for random unit v. Returns max relative error per mode."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    modes = ["full", "det"] if model.use_cb_bias else ["full"]
    res = {}
    with torch.no_grad():
        b = cb_bias_at(model, h, x) if model.use_cb_bias else None
        for mode in modes:
            J = batched_jacobian(model, h, x, mode, b)
            errs = []
            for _ in range(n_dirs):
                v = torch.randn(h.shape, generator=g, dtype=h.dtype).to(h.device)
                v = v / v.norm(dim=-1, keepdim=True)
                fp = step_h_next(model, h + delta * v, x, mode, b)
                fm = step_h_next(model, h - delta * v, x, mode, b)
                fd = (fp - fm) / (2 * delta)
                jv = torch.einsum("bij,bj->bi", J, v)
                errs.append(((fd - jv).norm(dim=-1) / jv.norm(dim=-1).clamp_min(1e-12)).max().item())
            res[mode] = float(max(errs))
    return res
