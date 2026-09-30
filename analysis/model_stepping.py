"""Model loading, trial generation and a one-step transition for the DMS mechanism analysis.

DMS trials: length M is shared within a batch; the sample is at t_s = M-N and the
comparison/readout at t_c = M-1.
"""
import os
import numpy as np
import torch

from analysis.rebuild_model_utils import load_run_config, load_state_dict, build_model_from_config_and_state
from analysis.cb_ablation import find_available_Ns
from tasks.task_generators import make_batch_mtstyle_dms

RNN_COLOR = "cornflowerblue"
CB_COLOR = "salmon"
HIDDEN_COLOR = "#20A0C9"
CB_SIGNAL_COLOR = "#D62E24"

BASE = "results/single_task/dms_multistyle_repeats"
RNN_RUNS = [f"{BASE}/single_dms_elman_size_202_noCB_RNNlr0.01_network_{i}" for i in range(1, 9)]
CB_RUNS = [f"{BASE}/single_dms_elman_size_64_CB_gc256_RNNlr0.01_CBlr0.01_CBinput_simult_network_{i}" for i in range(1, 9)]
HIDDEN_SIZES = {"rnn_only": 202, "cb_rnn": 64}


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


# Trial generation

def make_fixed_dms_trials(N, batch_size=64, n_batches=8, seed=0):
    """Deterministic DMS batches at level N. Returns a list of dicts (seqs, label,
        sample_bit, compare_bit, t_sample, t_compare, T).
    
    """
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)

        batches = []
        for b in range(n_batches):
            seqs, labs = make_batch_mtstyle_dms([N], batch_size)
            label = labs[-1].long()
            T = seqs.shape[0]
            t_sample = T - N  # vec[-N] in 0-indexed array of length T is index T-N
            t_compare = T - 1
            bits = seqs.squeeze(-1)  # [T, B]
            sample_bit = bits[t_sample].long()
            compare_bit = bits[t_compare].long()
            # label must equal (sample_bit == compare_bit)
            assert torch.equal(label, (sample_bit == compare_bit).long()), (
                "label does not match get_match semantics; task implementation "
                "assumption is wrong."
            )
            batches.append(dict(
                seqs=seqs.clone(), label=label, sample_bit=sample_bit,
                compare_bit=compare_bit, t_sample=int(t_sample),
                t_compare=int(t_compare), T=int(T),
            ))
        return batches
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)


# Model loading

def load_model(run_path, N, device="cpu"):
    """Load a checkpoint in eval mode with gradients disabled."""
    cfg = load_run_config(run_path)
    sd = load_state_dict(run_path, N=N)
    model = build_model_from_config_and_state(cfg=cfg, state_dict=sd, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def shared_available_Ns(run_paths):
    """Intersection of checkpoint Ns available across all given run dirs."""
    shared = None
    for rp in run_paths:
        ns = set(find_available_Ns(rp))
        shared = ns if shared is None else shared & ns
    return sorted(shared) if shared else []


# One-step transition

def zero_init_hidden(model, batch_size, device):
    return torch.zeros(batch_size, model.hidden_size, device=device)


def run_full_dynamics(model, seqs, device="cpu"):
    """Full rollout from a zero initial state. Returns (out_class, dynamics)."""
    seqs = seqs.to(device)
    B = seqs.shape[1]
    h0 = zero_init_hidden(model, B, device)
    with torch.no_grad():
        hs_out, out_class, dynamics = model(seqs, hs=h0, return_dynamics=True)
    return out_class, dynamics


def cb_input_for_step(model, x_t):
    """CB task input at one timestep, or None if CB does not take input."""
    if not model.use_cb_bias or model.cb_input_size <= 0:
        return None
    return x_t


def cb_bias_at(model, h, x_t):
    """b_cb,t = CB(h_t, x_t) via the model's own CB submodule. Returns None
    if this model has no CB pathway."""
    if not model.use_cb_bias:
        return None
    cb_in = cb_input_for_step(model, x_t)
    return model.cb(h, x=cb_in)


def manual_step(model, h, x_t, b_cb_override=None, force_no_cb=False):
    """One recurrent step using the model's own submodules.

        b_cb_override replaces the CB output; force_no_cb sets it to zero.
        Returns dict with h_next, pre, post, b_cb.
    
    """
    pre = model.inp(x_t) + model.hh(h)

    if model.use_cb_bias and not force_no_cb:
        b_cb = b_cb_override if b_cb_override is not None else cb_bias_at(model, h, x_t)
        post = pre + b_cb
    else:
        b_cb = torch.zeros_like(pre) if model.use_cb_bias else None
        post = pre

    alpha = 1.0 / model.tau
    h_next = (1.0 - alpha) * h + alpha * model.afunc(post)

    return dict(h_next=h_next, pre=pre, post=post, b_cb=b_cb)


def verify_manual_step_matches_forward(model, seqs, device="cpu", atol=1e-6):
    """Assert manual_step reproduces model.forward."""
    seqs = seqs.to(device)
    T, B, _ = seqs.shape
    h0 = zero_init_hidden(model, B, device)

    with torch.no_grad():
        _, _, dyn = model(seqs, hs=h0, return_dynamics=True)

        h = h0
        for t in range(T):
            step = manual_step(model, h, seqs[t])
            assert torch.allclose(step["pre"], dyn["pre"][t], atol=atol), f"pre mismatch at t={t}"
            assert torch.allclose(step["post"], dyn["post"][t], atol=atol), f"post mismatch at t={t}"
            assert torch.allclose(step["h_next"], dyn["hidden"][t], atol=atol), f"hidden mismatch at t={t}"
            if model.use_cb_bias:
                assert torch.allclose(step["b_cb"], dyn["cb_bias"][t], atol=atol), f"cb_bias mismatch at t={t}"
            h = step["h_next"]

    return True
