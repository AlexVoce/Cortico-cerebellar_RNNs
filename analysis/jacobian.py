"""Local Jacobians of the one-step transition, with the CB-mediated term separated:

    J_full = (1 - alpha) I + alpha diag(f'(post)) (W_h + dCB/dh)
"""
import torch

from analysis.model_stepping import manual_step


def _leakyrelu_deriv(afunc_module, post):
    slope = getattr(afunc_module, "negative_slope", 0.01)
    return torch.where(post > 0, torch.ones_like(post), torch.full_like(post, slope))


def batched_step_jacobian(model, h, x_t, force_no_cb=False):
    """d h_{t+1} / d h_t per trial. Returns [B, H, H]. force_no_cb removes CB for this step."""
    def f(h_i, x_i):
        out = manual_step(model, h_i.unsqueeze(0), x_i.unsqueeze(0), force_no_cb=force_no_cb)
        return out["h_next"].squeeze(0)

    jac_fn = torch.func.jacrev(f, argnums=0)
    return torch.func.vmap(jac_fn)(h, x_t)


def batched_cb_output_jacobian(model, h, x_t):
    """d b_cb / d h_t per trial. Returns [B, H, H], or None if the model has no CB."""
    if not model.use_cb_bias:
        return None

    def f(h_i, x_i):
        from analysis.model_stepping import cb_bias_at
        return cb_bias_at(model, h_i.unsqueeze(0), x_i.unsqueeze(0)).squeeze(0)

    jac_fn = torch.func.jacrev(f, argnums=0)
    return torch.func.vmap(jac_fn)(h, x_t)


def decompose_full_jacobian(model, h, x_t, atol=1e-4):
    """J_full via autograd and its intrinsic/CB decomposition, checked to `atol`.

        Returns (J_full, J_intrinsic_term, J_cb_term, max_abs_reconstruction_error).
    
    """
    B, H = h.shape
    alpha = 1.0 / model.tau

    J_full = batched_step_jacobian(model, h, x_t, force_no_cb=False)

    step_full = manual_step(model, h, x_t)
    fprime = _leakyrelu_deriv(model.afunc, step_full["post"])  # [B, H]

    W_h = model.hh.weight  # h_next contribution = W_h @ h

    if model.use_cb_bias:
        J_cb_raw = batched_cb_output_jacobian(model, h, x_t)  # [B, H, H]
    else:
        J_cb_raw = torch.zeros(B, H, H, device=h.device, dtype=h.dtype)

    eye = torch.eye(H, device=h.device, dtype=h.dtype).unsqueeze(0)
    J_intrinsic_term = alpha * fprime.unsqueeze(-1) * W_h.unsqueeze(0)
    J_cb_term = alpha * fprime.unsqueeze(-1) * J_cb_raw
    J_full_reconstructed = (1 - alpha) * eye + J_intrinsic_term + J_cb_term

    err = (J_full - J_full_reconstructed).abs().max().item()
    if err > atol:
        raise RuntimeError(
            f"Jacobian decomposition mismatch: max abs error {err} > {atol}. "
            "The additive CB-mediated-Jacobian decomposition does not hold "
            "for this model/state -- do not trust downstream J_CB reports."
        )

    return J_full, J_intrinsic_term, J_cb_term, err


# Directional gain / preservation

def directional_gain(J, m):
    """g = ||J m|| per trial. Returns [B]."""
    if m.dim() == 1:
        m = m.unsqueeze(0).expand(J.shape[0], -1)
    Jm = torch.einsum("bij,bj->bi", J, m)
    return Jm.norm(dim=-1), Jm


def directional_preservation(Jm, m_next, eps=1e-8):
    """cos(J m, m_next) per trial."""
    if m_next.dim() == 1:
        m_next = m_next.unsqueeze(0).expand(Jm.shape[0], -1)
    num = (Jm * m_next).sum(dim=-1)
    den = Jm.norm(dim=-1).clamp_min(eps) * m_next.norm(dim=-1).clamp_min(eps)
    return num / den


# Multi-step propagation via Jacobian-vector products

def jvp_step(model, h, x_t, v, force_no_cb=False):
    """J_t @ v per trial via forward-mode AD."""
    def f(h_i):
        out = manual_step(model, h_i.unsqueeze(0), x_t_i.unsqueeze(0), force_no_cb=force_no_cb)
        return out["h_next"].squeeze(0)

    outs = []
    for b in range(h.shape[0]):
        x_t_i = x_t[b]
        _, jv = torch.func.jvp(f, (h[b],), (v[b],))
        outs.append(jv)
    return torch.stack(outs, dim=0)


def jvp_step_vmapped(model, h, x_t, v, force_no_cb=False):
    """Vectorised version of jvp_step using vmap over the trial dimension."""
    def f(h_i, x_i, v_i):
        def g(hh):
            out = manual_step(model, hh.unsqueeze(0), x_i.unsqueeze(0), force_no_cb=force_no_cb)
            return out["h_next"].squeeze(0)
        _, jv = torch.func.jvp(g, (h_i,), (v_i,))
        return jv

    return torch.func.vmap(f)(h, x_t, v)


def propagate_direction(model, h_seq, x_seq, m_start, force_no_cb=False, eps=1e-10):
    """Propagate m_start through the given steps with JVPs.

        Returns (log_growth [B], final unit direction [B, H]).
    
    """
    B = h_seq[0].shape[0]
    device = h_seq[0].device
    if m_start.dim() == 1:
        v = m_start.unsqueeze(0).expand(B, -1).clone()
    else:
        v = m_start.clone()
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(eps)

    log_growth = torch.zeros(B, device=device, dtype=v.dtype)

    for h_t, x_t in zip(h_seq, x_seq):
        v = jvp_step_vmapped(model, h_t, x_t, v, force_no_cb=force_no_cb)
        norm = v.norm(dim=-1).clamp_min(eps)
        log_growth = log_growth + torch.log(norm)
        v = v / norm.unsqueeze(-1)

    return log_growth, v
