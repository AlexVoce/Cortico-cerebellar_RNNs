"""CB-computation metrics on sliding-window Parity (Parts 2-6). Distances are per unit."""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from analysis.model_stepping import manual_step, to_numpy
from analysis.jacobian import batched_cb_output_jacobian
from analysis.parity_dynamics import correct_transition_score
from analysis.parity_update_rule import required_parity_of_group


# Part 2: 8-condition CB output geometry

def condition_geometry(cb_vectors, group_id, hidden_size):
    """Per-condition CB centroids, magnitudes and spreads, plus pairwise cosines/distances."""
    n_groups = 8
    H = cb_vectors.shape[1]
    centroids = np.full((n_groups, H), np.nan)
    counts = np.zeros(n_groups, dtype=int)
    within_ss = np.full(n_groups, np.nan)

    for g in range(n_groups):
        idx = np.where(group_id == g)[0]
        counts[g] = len(idx)
        if len(idx) == 0:
            continue
        centroids[g] = cb_vectors[idx].mean(axis=0)
        within_ss[g] = float(np.mean(np.sum((cb_vectors[idx] - centroids[g]) ** 2, axis=1)))

    valid_g = counts > 0
    grand_mean = np.nanmean(centroids[valid_g], axis=0)

    cos_mat = np.full((n_groups, n_groups), np.nan)
    dist_mat = np.full((n_groups, n_groups), np.nan)
    for i in range(n_groups):
        for j in range(n_groups):
            if valid_g[i] and valid_g[j]:
                a, b = centroids[i], centroids[j]
                cos_mat[i, j] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                dist_mat[i, j] = float(np.linalg.norm(a - b) / np.sqrt(hidden_size))

    magnitude_per_unit = np.array([
        np.linalg.norm(centroids[g]) / np.sqrt(hidden_size) if valid_g[g] else np.nan
        for g in range(n_groups)
    ])
    within_rms_per_unit = np.sqrt(within_ss) / np.sqrt(hidden_size)
    between_ss = float(np.mean(np.sum((centroids[valid_g] - grand_mean) ** 2, axis=1)))
    between_rms_per_unit = float(np.sqrt(between_ss) / np.sqrt(hidden_size))

    return dict(
        cos_matrix=cos_mat, dist_matrix_per_unit=dist_mat, counts=counts,
        magnitude_per_unit=magnitude_per_unit, within_rms_per_unit=within_rms_per_unit,
        between_rms_per_unit=between_rms_per_unit, centroids=centroids,
    )


# Part 3: encoding models

def encoding_model_r2(cb_vectors, current_parity, incoming_bit, outgoing_bit, next_parity,
                       batch_idx, seed=0, alpha=1.0, test_frac=0.3):
    """Held-out (by batch) ridge R^2 predicting CB output from the update variables."""
    c2 = 2.0 * current_parity.astype(float) - 1.0
    i2 = 2.0 * incoming_bit.astype(float) - 1.0
    o2 = 2.0 * outgoing_bit.astype(float) - 1.0
    n2 = 2.0 * next_parity.astype(float) - 1.0

    designs = {
        "A_parity_only": np.stack([c2], axis=1),
        "B_main_effects": np.stack([c2, i2, o2], axis=1),
        "C_pairwise": np.stack([c2, i2, o2, c2 * i2, c2 * o2, i2 * o2], axis=1),
        "D_full_interaction": np.stack([c2, i2, o2, c2 * i2, c2 * o2, i2 * o2, c2 * i2 * o2], axis=1),
        "N_next_parity_only": np.stack([n2], axis=1),
    }

    uniq_batches = np.unique(batch_idx)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq_batches))
    n_test_b = max(1, int(round(len(uniq_batches) * test_frac)))
    test_batches = set(uniq_batches[perm[:n_test_b]].tolist())
    test_mask = np.isin(batch_idx, list(test_batches))
    train_mask = ~test_mask

    results = {}
    for name, X in designs.items():
        Xtr, Xte = X[train_mask], X[test_mask]
        Ytr, Yte = cb_vectors[train_mask], cb_vectors[test_mask]
        reg = Ridge(alpha=alpha)
        reg.fit(Xtr, Ytr)
        results[name] = dict(
            r2_train=float(r2_score(Ytr, reg.predict(Xtr))),
            r2_test=float(r2_score(Yte, reg.predict(Xte))),
            n_features=int(X.shape[1]), n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        )
    return results


# Parts 4-5: targeted causal swaps and margins

def new_causal_swap_conditions(model, transitions, m_hat, midpoint, eval_idx, device="cpu", seed=0):
    """Targeted CB swaps from matched donor trials (G1-G3, H, I), scored on the
        held-out correct-transition axis.

        Returns (results, margin), or (None, None) if the model has no CB.
    
    """
    if not model.use_cb_bias or transitions["cb_before"] is None or transitions["outgoing_bit"] is None:
        return None, None

    h_before = torch.tensor(transitions["h_before"][eval_idx], dtype=torch.float32, device=device)
    x_t = torch.tensor(transitions["x_t"][eval_idx], dtype=torch.float32, device=device)
    cb_real = torch.tensor(transitions["cb_before"][eval_idx], dtype=torch.float32, device=device)
    p_before = transitions["wp_before"][eval_idx]
    x_bit = (transitions["x_t"][eval_idx].squeeze(-1) > 0.5).astype(int)
    outgoing = transitions["outgoing_bit"][eval_idx]
    required = transitions["wp_after"][eval_idx]

    group_id = p_before.astype(int) * 4 + x_bit.astype(int) * 2 + outgoing.astype(int)
    n = h_before.shape[0]

    def flip_bit(g, which):
        c, i, o = (g >> 2) & 1, (g >> 1) & 1, g & 1
        if which == "c":
            c = 1 - c
        elif which == "i":
            i = 1 - i
        else:
            o = 1 - o
        return [c * 4 + i * 2 + o]

    def same_next_diff_group(g):
        req = required_parity_of_group(g)
        return [gg for gg in range(8) if gg != g and required_parity_of_group(gg) == req]

    def diff_next_group(g):
        req = required_parity_of_group(g)
        return [gg for gg in range(8) if required_parity_of_group(gg) != req]

    def borrow(donor_group_ids_fn, rng_seed):
        rng = np.random.RandomState(rng_seed)
        cb_new = cb_real.clone()
        for g in range(8):
            recip = np.where(group_id == g)[0]
            if len(recip) == 0:
                continue
            donor_idx = np.where(np.isin(group_id, donor_group_ids_fn(g)))[0]
            if len(donor_idx) == 0:
                continue
            chosen = rng.choice(donor_idx, size=len(recip), replace=True)
            cb_new[recip] = cb_real[chosen]
        return cb_new

    cb_swaps = {
        "G1_wrong_outgoing": borrow(lambda g: flip_bit(g, "o"), seed + 10),
        "G2_wrong_incoming": borrow(lambda g: flip_bit(g, "i"), seed + 11),
        "G3_wrong_parity": borrow(lambda g: flip_bit(g, "c"), seed + 12),
        "H_same_next_diff_cond": borrow(same_next_diff_group, seed + 13),
        "I_diff_next_diff_cond": borrow(diff_next_group, seed + 14),
    }

    with torch.no_grad():
        step_full = manual_step(model, h_before, x_t, b_cb_override=cb_real)
        step_zero = manual_step(model, h_before, x_t, force_no_cb=True)
        steps = {name: manual_step(model, h_before, x_t, b_cb_override=cb) for name, cb in cb_swaps.items()}
    steps["A_full"] = step_full
    steps["B_zero"] = step_zero

    results = {}
    for name, step in steps.items():
        h_after = to_numpy(step["h_next"])
        scores = correct_transition_score(h_after, m_hat, midpoint, required)
        results[name] = dict(mean_score=float(scores.mean()), sem_score=float(scores.std() / max(len(scores) ** 0.5, 1)), n=len(scores))
        for x_val, tag in [(0, "maintain"), (1, "switch")]:
            m = x_bit == x_val
            if m.sum() > 0:
                results[f"{name}__{tag}"] = dict(
                    mean_score=float(scores[m].mean()), sem_score=float(scores[m].std() / max(m.sum() ** 0.5, 1)), n=int(m.sum()),
                )

    h_full = to_numpy(step_full["h_next"])
    h_zero = to_numpy(step_zero["h_next"])
    score_full = correct_transition_score(h_full, m_hat, midpoint, required)
    score_zero = correct_transition_score(h_zero, m_hat, midpoint, required)
    margin = []
    for g in range(8):
        idx = np.where(group_id == g)[0]
        if len(idx) == 0:
            continue
        margin.append(dict(
            group_id=g, n=len(idx),
            score_full=float(score_full[idx].mean()), score_zero=float(score_zero[idx].mean()),
            margin=float((score_full[idx] - score_zero[idx]).mean()),
        ))

    return results, margin


# Part 6: within- vs between-condition CB variation

def lookup_quantification(model, transitions, eval_idx, device="cpu", min_group_n=3):
    """Within- vs between-condition CB spread, and the within-condition spread
        predicted by the local CB Jacobian. Returns None if unavailable.
    
    """
    if not model.use_cb_bias or transitions["cb_before"] is None or transitions["outgoing_bit"] is None:
        return None

    h_before_np = transitions["h_before"][eval_idx]
    x_t_np = transitions["x_t"][eval_idx]
    cb_real_np = transitions["cb_before"][eval_idx]
    p_before = transitions["wp_before"][eval_idx]
    x_bit = (transitions["x_t"][eval_idx].squeeze(-1) > 0.5).astype(int)
    outgoing = transitions["outgoing_bit"][eval_idx]
    group_id = p_before.astype(int) * 4 + x_bit.astype(int) * 2 + outgoing.astype(int)
    H = cb_real_np.shape[1]

    h_before = torch.tensor(h_before_np, dtype=torch.float32, device=device)
    x_t = torch.tensor(x_t_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        J_cb = batched_cb_output_jacobian(model, h_before, x_t)  # [n, H, H]
    J_cb_np = to_numpy(J_cb)

    within_actual_sq, within_pred_sq = [], []
    centroids = {}
    for g in range(8):
        idx = np.where(group_id == g)[0]
        if len(idx) < min_group_n:
            continue
        centroid_cb = cb_real_np[idx].mean(axis=0)
        centroid_h = h_before_np[idx].mean(axis=0)
        centroids[g] = centroid_cb
        within_actual_sq.append(np.mean(np.sum((cb_real_np[idx] - centroid_cb) ** 2, axis=1)))
        dh = h_before_np[idx] - centroid_h
        pred = np.einsum("nij,nj->ni", J_cb_np[idx], dh)
        within_pred_sq.append(np.mean(np.sum(pred ** 2, axis=1)))

    if len(centroids) < 2:
        return None

    grand_mean = np.mean(list(centroids.values()), axis=0)
    between_sq = float(np.mean([np.sum((c - grand_mean) ** 2) for c in centroids.values()]))

    within_actual_rms_pu = float(np.sqrt(np.mean(within_actual_sq)) / np.sqrt(H))
    within_pred_rms_pu = float(np.sqrt(np.mean(within_pred_sq)) / np.sqrt(H))
    between_rms_pu = float(np.sqrt(between_sq) / np.sqrt(H))

    return dict(
        within_actual_rms_per_unit=within_actual_rms_pu,
        within_local_predicted_rms_per_unit=within_pred_rms_pu,
        between_condition_rms_per_unit=between_rms_pu,
        ratio_between_over_within_actual=between_rms_pu / max(within_actual_rms_pu, 1e-8),
        ratio_between_over_within_predicted=between_rms_pu / max(within_pred_rms_pu, 1e-8),
        n_groups_used=len(centroids),
    )
