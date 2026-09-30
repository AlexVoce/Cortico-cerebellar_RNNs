"""Parity trial generation and parity bookkeeping; model utilities are shared with
model_stepping.

The label is the parity of the last N bits, so the one-step update is
p_{t+1} = p_t XOR x_{t+1} XOR x_{t-N+1}.
"""
import numpy as np
import torch

from tasks.task_generators import make_batch_mtstyle_parity

from analysis.model_stepping import (  # noqa: F401
    RNN_COLOR, CB_COLOR, HIDDEN_COLOR, CB_SIGNAL_COLOR,
    default_device, to_numpy, load_model, shared_available_Ns,
    zero_init_hidden, run_full_dynamics, cb_input_for_step, cb_bias_at,
    manual_step, verify_manual_step_matches_forward,
)

BASE = "results/single_task/parity_multistyle_repeats"
RNN_RUNS = [f"{BASE}/single_parity_elman_size_202_noCB_RNNlr0.01_network_{i}" for i in range(1, 9)]
CB_RUNS = [f"{BASE}/single_parity_elman_size_64_CB_gc256_RNNlr0.01_CBlr0.01_CBinput_simult_network_{i}" for i in range(1, 9)]
HIDDEN_SIZES = {"rnn_only": 202, "cb_rnn": 64}


def _windowed_and_running_parity(bits, N):
    """Returns (windowed_parity [T,B], nan for t < N-1; running_parity [T,B])."""
    T, B = bits.shape
    cumsum = np.cumsum(bits, axis=0)
    running_parity = np.mod(cumsum, 2)

    windowed_parity = np.full((T, B), np.nan)
    for t in range(N - 1, T):
        lo = t - N + 1
        window_sum = cumsum[t] - (cumsum[lo - 1] if lo > 0 else 0)
        windowed_parity[t] = np.mod(window_sum, 2)

    return windowed_parity, running_parity


def make_fixed_parity_trials(N, batch_size=64, n_batches=8, seed=0):
    """Deterministic Parity batches at level N. Returns a list of dicts (seqs, label,
        windowed_parity, running_parity, T).
    
    """
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)

        batches = []
        for b in range(n_batches):
            seqs, labs = make_batch_mtstyle_parity([N], batch_size)
            label = labs[-1].long()
            T = seqs.shape[0]
            bits = seqs.squeeze(-1).numpy()
            windowed_parity, running_parity = _windowed_and_running_parity(bits, N)

            assert np.array_equal(
                windowed_parity[-1].astype(int), label.numpy()
            ), "windowed_parity[-1] does not match get_parity label; task assumption is wrong."

            batches.append(dict(
                seqs=seqs.clone(), label=label,
                windowed_parity=windowed_parity, running_parity=running_parity,
                T=int(T),
            ))
        return batches
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
