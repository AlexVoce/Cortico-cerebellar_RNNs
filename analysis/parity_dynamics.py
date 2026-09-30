"""Parity latent-state, transition and causal-intervention metrics.

seqs[t] is the input consumed at step t and dynamics['hidden'][t] the resulting state.
"""
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.model_stepping import run_full_dynamics, manual_step, zero_init_hidden, to_numpy
from analysis.predictive_analysis import _split_batches_train_test


def _make_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


# Per-batch dynamics

def collect_parity_dynamics(model, batches, device="cpu"):
    """Run the model on fixed Parity batches (zero initial state). Returns per-batch dicts."""
    per_batch = []
    for b in batches:
        _, dyn = run_full_dynamics(model, b["seqs"], device=device)
        per_batch.append(dict(
            hidden=dyn["hidden"], cb_bias=dyn.get("cb_bias", None), seqs=b["seqs"].to(device),
            windowed_parity=b["windowed_parity"], running_parity=b["running_parity"], T=b["T"],
        ))
    return per_batch


# Part 1: decoding and separation by t_before_end

def pool_by_t_before_end(per_batch, hidden_key="hidden", parity_key="windowed_parity"):
    """Pool hidden, cb_bias and parity by t_before_end (0 = last timestep)."""
    T_max = max(b["T"] for b in per_batch)
    by_pos = [dict(hidden=[], cb_bias=[], parity=[]) for _ in range(T_max)]

    for b in per_batch:
        T = b["T"]
        parity_arr = b[parity_key]
        for offset in range(T):
            t = T - 1 - offset
            p = parity_arr[t]  # [B] numpy, may be all-nan
            valid = ~np.isnan(p)
            if not valid.any():
                continue
            by_pos[offset]["hidden"].append(to_numpy(b[hidden_key][t])[valid])
            if b["cb_bias"] is not None:
                by_pos[offset]["cb_bias"].append(to_numpy(b["cb_bias"][t])[valid])
            by_pos[offset]["parity"].append(p[valid].astype(int))

    out = []
    for entry in by_pos:
        if len(entry["parity"]) == 0:
            out.append(dict(hidden=None, cb_bias=None, parity=None))
            continue
        hidden = np.concatenate(entry["hidden"], axis=0)
        cb_bias = np.concatenate(entry["cb_bias"], axis=0) if entry["cb_bias"] else None
        parity = np.concatenate(entry["parity"], axis=0)
        out.append(dict(hidden=hidden, cb_bias=cb_bias, parity=parity))
    return out


def delta_parity_over_position(by_pos, key="hidden", hidden_size=None):
    """Even-vs-odd mean difference and its norm per t_before_end, raw and per unit."""
    N = len(by_pos)
    norms, norms_pu, ns = [], [], []
    for t in range(N):
        x = by_pos[t][key]
        if x is None:
            norms.append(np.nan); norms_pu.append(np.nan); ns.append(0)
            continue
        p = by_pos[t]["parity"]
        a, b = x[p == 0], x[p == 1]
        if len(a) == 0 or len(b) == 0:
            norms.append(np.nan); norms_pu.append(np.nan); ns.append(len(p))
            continue
        d = a.mean(axis=0) - b.mean(axis=0)
        norm = float(np.linalg.norm(d))
        H = hidden_size if hidden_size is not None else x.shape[1]
        norms.append(norm)
        norms_pu.append(norm / (H ** 0.5))
        ns.append(len(p))
    return dict(norm=np.array(norms), norm_per_unit=np.array(norms_pu), n=np.array(ns))


def decode_parity_over_position(by_pos, key="hidden", test_frac=0.3, seed=0, max_samples=4000):
    """Held-out logistic-regression AUC decoding parity at each t_before_end."""
    N = len(by_pos)
    aucs = np.full(N, np.nan)
    rng = np.random.RandomState(seed)
    for t in range(N):
        x = by_pos[t][key]
        if x is None:
            continue
        y = by_pos[t]["parity"]
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
        clf = _make_clf()
        clf.fit(x[train_idx], y_tr)
        aucs[t] = roc_auc_score(y_te, clf.predict_proba(x[test_idx])[:, 1])
    return aucs


# Part 2: transitions and parity-axis stability

def extract_transitions(per_batch, device="cpu", N=None):
    """One-step transitions (t >= N) pooled across batches.

        Returns stacked h_before, x_t, h_after, cb_before, parities, t_before_end and,
        if N is given, outgoing_bit = bits[t-N].
    
    """
    rows = dict(h_before=[], x_t=[], h_after=[], cb_before=[],
                wp_before=[], wp_after=[], rp_before=[], rp_after=[],
                outgoing_bit=[], t_before_end=[], batch_idx=[], trial_idx=[])

    for bi, b in enumerate(per_batch):
        T = b["T"]
        wp, rp = b["windowed_parity"], b["running_parity"]
        bits = to_numpy(b["seqs"]).squeeze(-1) if N is not None else None  # [T, B]
        for t in range(1, T):
            wp_b, wp_a = wp[t - 1], wp[t]
            valid = ~(np.isnan(wp_b) | np.isnan(wp_a))
            if not valid.any():
                continue
            idx = np.where(valid)[0]
            rows["h_before"].append(to_numpy(b["hidden"][t - 1])[idx])
            rows["x_t"].append(to_numpy(b["seqs"][t])[idx])
            rows["h_after"].append(to_numpy(b["hidden"][t])[idx])
            if b["cb_bias"] is not None:
                rows["cb_before"].append(to_numpy(b["cb_bias"][t])[idx])
            rows["wp_before"].append(wp_b[idx].astype(int))
            rows["wp_after"].append(wp_a[idx].astype(int))
            rows["rp_before"].append(rp[t - 1][idx].astype(int))
            rows["rp_after"].append(rp[t][idx].astype(int))
            if N is not None:
                rows["outgoing_bit"].append(bits[t - N][idx].astype(int))
            rows["t_before_end"].append(np.full(len(idx), T - 1 - t))
            rows["batch_idx"].append(np.full(len(idx), bi))
            rows["trial_idx"].append(idx)

    out = {}
    for k, v in rows.items():
        if len(v) == 0:
            out[k] = None
        elif k == "cb_before" and len(v) < len(rows["h_before"]):
            out[k] = None  # no CB in this model
        else:
            out[k] = np.concatenate(v, axis=0)
    return out


def parity_direction_stability(transitions, parity_key="wp_after", key="h_after", n_time_bins=5):
    """Pairwise cosine between even-vs-odd directions in t_before_end bins."""
    t_before_end = transitions["t_before_end"]
    x = transitions[key]
    p = transitions[parity_key]

    edges = np.quantile(t_before_end, np.linspace(0, 1, n_time_bins + 1))
    edges[-1] += 1  # inclusive upper bound
    directions = []
    bin_labels = []
    for i in range(n_time_bins):
        in_bin = (t_before_end >= edges[i]) & (t_before_end < edges[i + 1])
        if in_bin.sum() < 10:
            directions.append(None)
            bin_labels.append((edges[i], edges[i + 1]))
            continue
        a, b = x[in_bin & (p == 0)], x[in_bin & (p == 1)]
        if len(a) == 0 or len(b) == 0:
            directions.append(None)
        else:
            d = a.mean(axis=0) - b.mean(axis=0)
            n = np.linalg.norm(d)
            directions.append(d / n if n > 1e-8 else None)
        bin_labels.append((edges[i], edges[i + 1]))

    n_bins = len(directions)
    cos_mat = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            if directions[i] is not None and directions[j] is not None:
                cos_mat[i, j] = float(np.dot(directions[i], directions[j]))

    return dict(cos_matrix=cos_mat, bin_edges=bin_labels, directions=directions)


def transition_condition_means(transitions, parity_key_before="wp_before", device="cpu"):
    """Per-(parity, input) condition centroids of h_before, h_after and cb_before."""
    p_before = transitions[parity_key_before]
    x_t = transitions["x_t"].squeeze(-1)
    x_bit = (x_t > 0.5).astype(int)

    conditions = {}
    for p_val, p_name in [(0, "even"), (1, "odd")]:
        for x_val in [0, 1]:
            mask = (p_before == p_val) & (x_bit == x_val)
            entry = dict(mask=mask, n=int(mask.sum()))
            if mask.sum() > 0:
                entry["h_before_mean"] = transitions["h_before"][mask].mean(axis=0)
                entry["h_after_mean"] = transitions["h_after"][mask].mean(axis=0)
                if transitions["cb_before"] is not None:
                    entry["cb_mean"] = transitions["cb_before"][mask].mean(axis=0)
            conditions[f"{p_name}_x{x_val}"] = entry
    return conditions


# Part 3: one-step causal interventions

def build_reference_geometry(transitions, parity_key_after="wp_after", train_frac=0.7, seed=0):
    """Held-out even/odd centroid axis at h_after. Returns (m_hat, midpoint, train_idx, eval_idx)."""
    n = transitions["h_after"].shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_frac))
    train_idx, eval_idx = idx[:n_train], idx[n_train:]

    p = transitions[parity_key_after][train_idx]
    h = transitions["h_after"][train_idx]
    mu_even, mu_odd = h[p == 0].mean(axis=0), h[p == 1].mean(axis=0)
    d = mu_even - mu_odd
    norm = np.linalg.norm(d)
    m_hat = d / norm if norm > 1e-8 else d
    midpoint = (mu_even + mu_odd) / 2
    return m_hat, midpoint, train_idx, eval_idx


def correct_transition_score(h, m_hat, midpoint, required_parity):
    """Projection of h onto m_hat, signed so higher = closer to the required parity."""
    raw = (h - midpoint) @ m_hat
    sign = np.where(required_parity == 0, 1.0, -1.0)  # required==0 (even): higher raw = more correct
    return sign * raw


def causal_one_step_parity_interventions(
    model, transitions, m_hat, midpoint, eval_idx,
    parity_key_before="wp_before", parity_key_after="wp_after",
    device="cpu", seed=0,
):
    """One-step replay under CB conditions A-F, scored with correct_transition_score."""
    if model.use_cb_bias is False or transitions["cb_before"] is None:
        return None

    h_before = torch.tensor(transitions["h_before"][eval_idx], dtype=torch.float32, device=device)
    x_t = torch.tensor(transitions["x_t"][eval_idx], dtype=torch.float32, device=device)
    cb_real = torch.tensor(transitions["cb_before"][eval_idx], dtype=torch.float32, device=device)
    p_before = transitions[parity_key_before][eval_idx]
    x_bit = (transitions["x_t"][eval_idx].squeeze(-1) > 0.5).astype(int)
    required = transitions[parity_key_after][eval_idx]

    n = h_before.shape[0]
    rng = np.random.RandomState(seed)

    with torch.no_grad():
        step_full = manual_step(model, h_before, x_t, b_cb_override=cb_real)
        step_zero = manual_step(model, h_before, x_t, force_no_cb=True)

        cb_mean = cb_real.mean(dim=0, keepdim=True).expand_as(cb_real)
        step_mean = manual_step(model, h_before, x_t, b_cb_override=cb_mean)

        perm = rng.permutation(n)
        step_shuffled = manual_step(model, h_before, x_t, b_cb_override=cb_real[perm])

        # E: swap CB between opposite-parity, same-input groups
        cb_opposite = cb_real.clone()
        for x_val in [0, 1]:
            grp0 = np.where((p_before == 0) & (x_bit == x_val))[0]
            grp1 = np.where((p_before == 1) & (x_bit == x_val))[0]
            if len(grp0) > 0 and len(grp1) > 0:
                cb_opposite[grp0] = cb_real[np.random.RandomState(seed + 1).choice(grp1, size=len(grp0))]
                cb_opposite[grp1] = cb_real[np.random.RandomState(seed + 2).choice(grp0, size=len(grp1))]
        step_opp_parity = manual_step(model, h_before, x_t, b_cb_override=cb_opposite)

        # F: swap CB between same-parity, opposite-input groups
        cb_wrong_input = cb_real.clone()
        for p_val in [0, 1]:
            grp0 = np.where((p_before == p_val) & (x_bit == 0))[0]
            grp1 = np.where((p_before == p_val) & (x_bit == 1))[0]
            if len(grp0) > 0 and len(grp1) > 0:
                cb_wrong_input[grp0] = cb_real[np.random.RandomState(seed + 3).choice(grp1, size=len(grp0))]
                cb_wrong_input[grp1] = cb_real[np.random.RandomState(seed + 4).choice(grp0, size=len(grp1))]
        step_wrong_input = manual_step(model, h_before, x_t, b_cb_override=cb_wrong_input)

    results = {}
    for name, step in [
        ("A_full", step_full), ("B_zero", step_zero), ("C_mean", step_mean),
        ("D_shuffled", step_shuffled), ("E_opposite_parity", step_opp_parity),
        ("F_wrong_input", step_wrong_input),
    ]:
        h_after = to_numpy(step["h_next"])
        scores = correct_transition_score(h_after, m_hat, midpoint, required)
        results[name] = dict(
            mean_score=float(scores.mean()), sem_score=float(scores.std() / max(len(scores) ** 0.5, 1)),
            n=len(scores),
        )
        # also split by maintain (x=0) vs switch (x=1)
        for x_val, tag in [(0, "maintain"), (1, "switch")]:
            m = x_bit == x_val
            if m.sum() > 0:
                results[f"{name}__{tag}"] = dict(
                    mean_score=float(scores[m].mean()), sem_score=float(scores[m].std() / max(m.sum() ** 0.5, 1)),
                    n=int(m.sum()),
                )
    return results
