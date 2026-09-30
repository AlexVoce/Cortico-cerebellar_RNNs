"""Sliding-window parity variables for the CB-computation analysis.

next_parity = current_parity XOR incoming_bit XOR outgoing_bit, with
outgoing_bit = bits[t-N].
"""
import numpy as np
import pandas as pd

from analysis.model_stepping import to_numpy
from analysis.parity_trials import make_fixed_parity_trials


# Part 0: verify the sliding-window update rule

def verify_sliding_window_update(Ns, batch_size=128, n_batches=20, seed=13579):
    """Check the XOR update rule (and the naive rule without outgoing_bit) against
        real task trials. Returns a per-N DataFrame of accuracies.
    
    """
    rows = []
    for N in Ns:
        batches = make_fixed_parity_trials(N, batch_size=batch_size, n_batches=n_batches, seed=seed)
        n_xor3 = n_naive = n_tot = 0
        for b in batches:
            wp, T = b["windowed_parity"], b["T"]
            bits = b["seqs"].squeeze(-1).numpy()
            for t in range(N, T):
                wp_b, wp_a = wp[t - 1], wp[t]
                valid = ~(np.isnan(wp_b) | np.isnan(wp_a))
                if not valid.any():
                    continue
                inc = bits[t][valid].astype(int)
                out = bits[t - N][valid].astype(int)
                cur = wp_b[valid].astype(int)
                nxt = wp_a[valid].astype(int)
                n_xor3 += int(((cur ^ inc ^ out) == nxt).sum())
                n_naive += int(((cur ^ inc) == nxt).sum())
                n_tot += int(valid.sum())
        rows.append(dict(
            N=N, n_transitions=n_tot,
            xor3_accuracy=n_xor3 / n_tot if n_tot else np.nan,
            naive_accuracy=n_naive / n_tot if n_tot else np.nan,
        ))
    return pd.DataFrame(rows)


# Part 1: per-timestep pooling and decoding of incoming/outgoing/next parity

def pool_full_variables_by_t_before_end(per_batch, N):
    """Pool hidden, cb_bias, incoming_bit, outgoing_bit and next_parity by t_before_end."""
    T_max = max(b["T"] for b in per_batch)
    by_pos = [dict(hidden=[], cb_bias=[], incoming_bit=[], outgoing_bit=[], next_parity=[])
              for _ in range(T_max)]

    for b in per_batch:
        T = b["T"]
        wp = b["windowed_parity"]
        bits = to_numpy(b["seqs"]).squeeze(-1)  # [T, B]
        B = bits.shape[1]
        for offset in range(T):
            t = T - 1 - offset
            inc = bits[t]
            out = bits[t - N] if t - N >= 0 else np.full(B, np.nan)
            nxt = wp[t + 1] if t + 1 < T else np.full(B, np.nan)

            by_pos[offset]["hidden"].append(to_numpy(b["hidden"][t]))
            if b["cb_bias"] is not None:
                by_pos[offset]["cb_bias"].append(to_numpy(b["cb_bias"][t]))
            by_pos[offset]["incoming_bit"].append(inc)
            by_pos[offset]["outgoing_bit"].append(out)
            by_pos[offset]["next_parity"].append(nxt)

    out = []
    for entry in by_pos:
        if len(entry["hidden"]) == 0:
            out.append(dict(hidden=None, cb_bias=None, incoming_bit=None, outgoing_bit=None, next_parity=None))
            continue
        d = dict(hidden=np.concatenate(entry["hidden"], axis=0))
        d["cb_bias"] = np.concatenate(entry["cb_bias"], axis=0) if entry["cb_bias"] else None
        for k in ["incoming_bit", "outgoing_bit", "next_parity"]:
            d[k] = np.concatenate(entry[k], axis=0)
        out.append(d)
    return out


def decode_variable_over_position(by_pos, var_name, feature_key="hidden", test_frac=0.3, seed=0, max_samples=4000):
    """Held-out logistic-regression AUC decoding `var_name` at each t_before_end."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    N = len(by_pos)
    aucs = np.full(N, np.nan)
    rng = np.random.RandomState(seed)
    for t in range(N):
        x = by_pos[t][feature_key]
        y_full = by_pos[t][var_name]
        if x is None or y_full is None:
            continue
        valid = ~np.isnan(y_full)
        if valid.sum() < 20:
            continue
        x, y = x[valid], y_full[valid].astype(int)
        n = x.shape[0]
        if n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            x, y = x[idx], y[idx]
        idx_all = rng.permutation(x.shape[0])
        n_test = max(1, int(round(len(idx_all) * test_frac)))
        test_idx, train_idx = idx_all[:n_test], idx_all[n_test:]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(x[train_idx], y_tr)
        aucs[t] = roc_auc_score(y_te, clf.predict_proba(x[test_idx])[:, 1])
    return aucs


# 8-condition (current_parity, incoming_bit, outgoing_bit) bookkeeping

def compute_group_id(current_parity, incoming_bit, outgoing_bit):
    """Packs the 3 binary update variables into a single 0..7 group id."""
    return current_parity.astype(int) * 4 + incoming_bit.astype(int) * 2 + outgoing_bit.astype(int)


def required_parity_of_group(g):
    """Required next parity for group id g."""
    c, i, o = (g >> 2) & 1, (g >> 1) & 1, g & 1
    return c ^ i ^ o


GROUP_LABELS = {g: f"c{(g >> 2) & 1}_i{(g >> 1) & 1}_o{g & 1}" for g in range(8)}
