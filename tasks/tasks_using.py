import torch
import torch.nn.functional as F
import numpy as np

def generate_binary_sequence(M, balanced=False):
    if balanced:
        # for dms if the input sequence is correlated it'll make one class very likely
        return (torch.rand(M) < 0.5) * 1.
    else:
        # doesn't seem to have the same effect for nbit-parity
        return (torch.rand(M) < torch.rand(1)) * 1.

# Experiments (not included in paper) with sparse sequences
def generate_sparse_binary_sequence(M, sparsity=0.9):
    s = torch.rand(M) * 2 - 1
    s = torch.where(torch.abs(s) > sparsity, torch.sign(s), 0 * s)
    return s * 1.

############ DMS TASKS ##################
def get_match(vec, N):
    return (vec[-N] == vec[-1]).long()

def make_batch_multihead_dms(Ns, bs): # original version - identical basically to multistyle 
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M, balanced=True).unsqueeze(-1) for i in range(bs)]
        labels = [torch.stack([get_match(s, N) for s in sequences]).squeeze() for N in Ns]

        sequences = torch.stack(sequences)
        sequences = sequences.permute(1, 0, 2) # [bs, M, 1] -> [M, bs, 1] where M is equivalent to time dimension

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

############ N_PARITY TASKS ##############

def get_parity(vec, N):
    return (vec[-N:].sum() % 2).long()

def get_parity_in_time(vec, N):
    labels = []
    for idx in np.arange(N, len(vec)):
        vec_t = vec[idx-N:idx]
        l = get_parity(vec_t, N)
        labels.append(l)
        # labels.append(((vec_t + 1) / 2)[-N:].sum() % 2)

    labels = torch.stack(labels).long()

    return labels

def make_batch_Nbit_pair_parity(Ns, bs, duplicate=1, classify_in_time=False): # original
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
        # in each sequence of length M, duplicate each bit (duplicate) times
        sequences = [torch.repeat_interleave(s, duplicate, dim=0) for s in sequences]
        sequences = torch.stack(sequences)
        sequences = sequences.permute(1, 0, 2) # [M, bs, 1] where M is equivalent to time dimension
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

############ ODDBALL TASKS ##############
def _count_transitions(bits):
    """bits: [N] tensor of 0/1"""
    b = bits.long()
    return (b[1:] != b[:-1]).sum().item()

def get_volatility_oddball_label(vec, N, transition_frac_threshold=0.35):
    """
    Use the N-bit context to infer local volatility:
      - low transitions  -> stable context -> expect repeat
      - high transitions -> volatile context -> expect alternation

    Label:
      1 = probe matches expected (standard)
      0 = violation (deviant)
    """
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
    """
    mode in {'stable','volatile'}
    Returns [N] bits.
    stable   -> long runs
    volatile -> more alternation
    noise    -> independent flips to avoid triviality
    """
    c = torch.zeros(N, dtype=torch.float32)

    # seed first bit
    c[0] = 1.0 if torch.rand(1).item() < 0.5 else 0.0

    if mode == "stable":
        p_flip = 0.12   # low transition probability
    elif mode == "volatile":
        p_flip = 0.75   # high transition probability (alternation-ish)
    else:
        raise ValueError(mode)

    for t in range(1, N):
        if torch.rand(1).item() < p_flip:
            c[t] = 1.0 - c[t-1]
        else:
            c[t] = c[t-1]

    # inject noise flips
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
): # original
    """
    Harder MMN-like task where expectation depends on inferred context volatility.

    For max N:
      - build a stable or volatile context
      - infer expected probe from volatility + last bit
      - set probe to match or violate expectation

    Returns:
      sequences [T,B,1], labels list([B]) for each N
    """
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    Nmax = Ns[-1]

    with torch.no_grad():
        sequences = []

        for _ in range(bs):
            # Start with random background
            s = generate_binary_sequence(M, balanced=True).float()

            # Create a context mode (balanced)
            mode = "volatile" if (torch.rand(1).item() < 0.5) else "stable"
            context = _generate_context_with_mode(Nmax, mode=mode, noise=context_noise)

            # Write context into the final Nmax slots before probe
            s[-(Nmax+1):-1] = context

            # Infer expected probe from the actual generated context
            last_bit = int(context[-1].item())
            n_trans = _count_transitions(context)
            trans_frac = n_trans / max(1, (Nmax - 1))
            volatile = trans_frac > transition_frac_threshold
            expected = (1 - last_bit) if volatile else last_bit
            expected = float(expected)

            # Set probe to be standard or deviant
            if balance_probe:
                is_match = (torch.rand(1).item() < 0.5)
            else:
                is_match = (torch.rand(1).item() < standard_prob)

            s[-1] = expected if is_match else (1.0 - expected)

            # Optional: randomize prefix a bit more
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