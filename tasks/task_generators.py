import torch
import torch.nn.functional as F
import numpy as np

def generate_binary_sequence(M, balanced=False):
    if balanced:
        # unbalanced bits make one DMS class much more likely
        return (torch.rand(M) < 0.5) * 1.
    else:
        return (torch.rand(M) < torch.rand(1)) * 1.

# Sparse sequences (not used in the paper)
def generate_sparse_binary_sequence(M, sparsity=0.9):
    s = torch.rand(M) * 2 - 1
    s = torch.where(torch.abs(s) > sparsity, torch.sign(s), 0 * s)
    return s * 1.

# DMS task
def get_match(vec, N):
    return (vec[-N] == vec[-1]).long()

def make_batch_multihead_dms(Ns, bs):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M, balanced=True).unsqueeze(-1) for i in range(bs)]
        labels = [torch.stack([get_match(s, N) for s in sequences]).squeeze() for N in Ns]

        sequences = torch.stack(sequences)
        sequences = sequences.permute(1, 0, 2)  # [T,B,1]

    return sequences, labels

def make_batch_mtstyle_dms(Ns, bs): # matched to multitask training style
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)

    with torch.no_grad():
        sequences = [
            generate_binary_sequence(M, balanced=True).unsqueeze(-1)
            for _ in range(bs)
        ]

        labels = [
            torch.stack([get_match(seq.squeeze(-1), N) for seq in sequences]).long()
            for N in Ns
        ]

        sequences = torch.stack(sequences).permute(1, 0, 2)  # [T,B,1]

    return sequences, labels

# N-bit parity task
def get_parity(vec, N):
    return (vec[-N:].sum() % 2).long()

def get_parity_in_time(vec, N):
    labels = []
    for idx in np.arange(N, len(vec)):
        vec_t = vec[idx-N:idx]
        l = get_parity(vec_t, N)
        labels.append(l)

    labels = torch.stack(labels).long()

    return labels

def make_batch_Nbit_pair_parity(Ns, bs, duplicate=1, classify_in_time=False):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M,balanced=True).unsqueeze(-1) for i in range(bs)]
        if classify_in_time:
            if duplicate != 1:
                raise NotImplementedError
            labels = [torch.stack([get_parity_in_time(s, N) for s in sequences]) for N in Ns]
        else:
            labels = [torch.stack([get_parity(s, N) for s in sequences]) for N in Ns]
        sequences = [torch.repeat_interleave(s, duplicate, dim=0) for s in sequences]
        sequences = torch.stack(sequences)
        sequences = sequences.permute(1, 0, 2)  # [T,B,1]
    return sequences, labels

def make_batch_mtstyle_parity(Ns, bs): # matched to multitask training style
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)

    with torch.no_grad():
        sequences = [
            generate_binary_sequence(M, balanced=True).unsqueeze(-1)
            for _ in range(bs)
        ]

        labels = [
            torch.stack([get_parity(seq.squeeze(-1), N) for seq in sequences]).long()
            for N in Ns
        ]

        sequences = torch.stack(sequences).permute(1, 0, 2)  # [T,B,1]

    return sequences, labels

# Fixed-pool dataset control
# trials are random-offset slices of a fixed pre-generated pool of bit sequences
def make_fixed_pool(n_sequences=200, seq_len=4000, seed=12345):
    """Fixed pool [n_sequences, seq_len] of Bernoulli(0.5) bits."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n_sequences, seq_len, generator=g) < 0.5) * 1.


def make_fixed_dataset_batch_fn(get_label_fn, pool, seed=0):
    """batch_fn(Ns, bs) that samples trials from a fixed pool instead of fresh random bits."""
    n_sequences, seq_len = pool.shape
    rng = np.random.RandomState(seed)

    def batch_fn(Ns, bs):
        M_min = Ns[-1] + 2
        M_max = M_min + 3 * Ns[-1]
        M = np.random.randint(M_min, M_max)
        if M > seq_len:
            raise ValueError(f"Requested trial length M={M} exceeds fixed pool seq_len={seq_len}.")

        row_idx = rng.randint(0, n_sequences, size=bs)
        offsets = rng.randint(0, seq_len - M + 1, size=bs)

        with torch.no_grad():
            sequences = [
                pool[r, o:o + M].unsqueeze(-1)
                for r, o in zip(row_idx, offsets)
            ]
            labels = [
                torch.stack([get_label_fn(seq.squeeze(-1), N) for seq in sequences]).long()
                for N in Ns
            ]
            sequences = torch.stack(sequences).permute(1, 0, 2)  # [T,B,1]

        return sequences, labels

    return batch_fn


FIXED_POOL_DMS = make_fixed_pool(n_sequences=200, seq_len=4000, seed=12345)
FIXED_POOL_PARITY = make_fixed_pool(n_sequences=200, seq_len=4000, seed=67890)

make_batch_fixed_dataset_dms = make_fixed_dataset_batch_fn(get_match, FIXED_POOL_DMS, seed=1)
make_batch_fixed_dataset_parity = make_fixed_dataset_batch_fn(get_parity, FIXED_POOL_PARITY, seed=2)

# Mod-3 task
# window sum mod 3 (the K=3 analogue of get_parity)
def get_mod3(vec, N):
    return (vec[-N:].sum() % 3).long()

def make_batch_mtstyle_mod3(Ns, bs):  # matched to multitask training style
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)

    with torch.no_grad():
        sequences = [
            generate_binary_sequence(M, balanced=True).unsqueeze(-1)
            for _ in range(bs)
        ]

        labels = [
            torch.stack([get_mod3(seq.squeeze(-1), N) for seq in sequences]).long()
            for N in Ns
        ]

        sequences = torch.stack(sequences).permute(1, 0, 2)  # [T,B,1]

    return sequences, labels

# Oddball task
def _count_transitions(bits):
    """bits: [N] tensor of 0/1"""
    b = bits.long()
    return (b[1:] != b[:-1]).sum().item()

def get_volatility_oddball_label(vec, N, transition_frac_threshold=0.35):
    """Label 1 if the probe matches the volatility-based expectation (standard), 0 if deviant."""
    v = vec.squeeze().long()
    context = v[-(N+1):-1]   # [N]
    probe = v[-1]
    last_bit = context[-1]

    n_trans = _count_transitions(context)
    trans_frac = n_trans / max(1, (N - 1))

    volatile = trans_frac > transition_frac_threshold
    expected = (1 - last_bit) if volatile else last_bit

    return (probe == expected).long()

def _generate_context_with_mode(N, mode, noise=0.1):
    """Context of N bits in mode 'stable' (long runs) or 'volatile' (alternation), with noise flips."""
    c = torch.zeros(N, dtype=torch.float32)

    c[0] = 1.0 if torch.rand(1).item() < 0.5 else 0.0

    if mode == "stable":
        p_flip = 0.12
    elif mode == "volatile":
        p_flip = 0.75
    else:
        raise ValueError(mode)

    for t in range(1, N):
        if torch.rand(1).item() < p_flip:
            c[t] = 1.0 - c[t-1]
        else:
            c[t] = c[t-1]

    if noise > 0:
        flips = (torch.rand(N) < noise).float()
        c = torch.where(flips > 0, 1.0 - c, c)

    return c

def make_batch_volatility_oddball(
    Ns,
    bs,
    transition_frac_threshold=0.35,
    balance_probe=True,
    standard_prob=0.8,
    context_noise=0.05,
    random_prefix=True
):
    """Volatility oddball task. Returns sequences [T,B,1] and a list of labels [B] per N."""
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    Nmax = Ns[-1]

    with torch.no_grad():
        sequences = []

        for _ in range(bs):
            s = generate_binary_sequence(M, balanced=True).float()

            mode = "volatile" if (torch.rand(1).item() < 0.5) else "stable"
            context = _generate_context_with_mode(Nmax, mode=mode, noise=context_noise)

            # context fills the Nmax steps before the probe
            s[-(Nmax+1):-1] = context

            # expected probe from the generated context (noise can change its mode)
            last_bit = int(context[-1].item())
            n_trans = _count_transitions(context)
            trans_frac = n_trans / max(1, (Nmax - 1))
            volatile = trans_frac > transition_frac_threshold
            expected = (1 - last_bit) if volatile else last_bit
            expected = float(expected)

            if balance_probe:
                is_match = (torch.rand(1).item() < 0.5)
            else:
                is_match = (torch.rand(1).item() < standard_prob)

            s[-1] = expected if is_match else (1.0 - expected)

            if random_prefix and M > (Nmax + 1):
                prefix_len = M - (Nmax + 1)
                if prefix_len > 0 and torch.rand(1).item() < 0.3:
                    s[:prefix_len] = generate_binary_sequence(prefix_len, balanced=False).float()

            sequences.append(s.unsqueeze(-1))

        sequences = torch.stack(sequences)  # [B,M,1]
        labels = [
            torch.stack(
                [get_volatility_oddball_label(seq, N, transition_frac_threshold=transition_frac_threshold)
                 for seq in sequences]
            ).squeeze().long()
            for N in Ns
        ]
        sequences = sequences.permute(1, 0, 2)

    return sequences, labels

def make_batch_mtstyle_oddball(
    Ns,
    bs,
    transition_frac_threshold=0.35,
): # matched to multitask training style
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)

    with torch.no_grad():
        sequences = [
            generate_binary_sequence(M, balanced=True).unsqueeze(-1)
            for _ in range(bs)
        ]

        labels = [
            torch.stack([
                get_volatility_oddball_label(
                    seq.squeeze(-1),
                    N,
                    transition_frac_threshold=transition_frac_threshold,
                )
                for seq in sequences
            ]).long()
            for N in Ns
        ]

        sequences = torch.stack(sequences).permute(1, 0, 2)  # [T,B,1]

    return sequences, labels