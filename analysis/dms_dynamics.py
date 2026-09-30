"""Metrics for the DMS memory-mechanism analysis, aligned on t_rel = t - t_sample."""
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.model_stepping import run_full_dynamics, manual_step, to_numpy


# Aligned dynamics

def collect_aligned_dynamics(model, batches, device="cpu"):
    """Run the model on fixed batches and pool states by t_rel.

        Returns dict with N_rel, by_t (per-t_rel hidden, cb_bias, sample_bit, x_next)
        and per-batch trajectories.
    
    """
    per_t = None
    per_batch = []

    for b in batches:
        out_class, dyn = run_full_dynamics(model, b["seqs"], device=device)
        hidden = dyn["hidden"]  # [T,B,H]
        cb_bias = dyn.get("cb_bias", None)
        t_s, t_c = b["t_sample"], b["t_compare"]
        N_this = t_c - t_s + 1
        if per_t is None:
            per_t = [dict(hidden=[], cb_bias=[], sample_bit=[], x_next=[]) for _ in range(N_this)]

        for rel in range(N_this):
            t = t_s + rel
            per_t[rel]["hidden"].append(hidden[t])
            per_t[rel]["cb_bias"].append(cb_bias[t] if cb_bias is not None else None)
            per_t[rel]["sample_bit"].append(b["sample_bit"])
            x_next = b["seqs"][t + 1] if (t + 1) < b["seqs"].shape[0] else None
            per_t[rel]["x_next"].append(x_next)

        per_batch.append(dict(
            hidden=hidden, seqs=b["seqs"], t_sample=t_s, t_compare=t_c,
            sample_bit=b["sample_bit"], label=b["label"],
        ))

    by_t = []
    for rel in range(len(per_t)):
        hidden_cat = torch.cat(per_t[rel]["hidden"], dim=0)
        cb_list = per_t[rel]["cb_bias"]
        cb_cat = torch.cat(cb_list, dim=0) if cb_list[0] is not None else None
        sample_cat = torch.cat(per_t[rel]["sample_bit"], dim=0)
        x_next_list = [x for x in per_t[rel]["x_next"] if x is not None]
        x_next_cat = torch.cat(x_next_list, dim=0) if len(x_next_list) == len(per_t[rel]["x_next"]) else None
        by_t.append(dict(hidden=hidden_cat, cb_bias=cb_cat, sample_bit=sample_cat, x_next=x_next_cat))

    return dict(N_rel=len(by_t), by_t=by_t, per_batch=per_batch)


# Delta h / Delta b_cb

def _split_by_sample(x, sample_bit):
    x = to_numpy(x)
    s = to_numpy(sample_bit)
    return x[s == 0], x[s == 1]


def delta_over_time(by_t, key="hidden"):
    """Per-t_rel mean difference between sample==0 and sample==1 trials, and its norm."""
    N = len(by_t)
    deltas, norms, n0s, n1s = [], [], [], []
    D = None
    for t in range(N):
        x = by_t[t][key]
        if x is None:
            deltas.append(None)
            norms.append(np.nan)
            n0s.append(0)
            n1s.append(0)
            continue
        a, b = _split_by_sample(x, by_t[t]["sample_bit"])
        D = a.shape[1]
        d = a.mean(axis=0) - b.mean(axis=0)
        deltas.append(d)
        norms.append(float(np.linalg.norm(d)))
        n0s.append(a.shape[0])
        n1s.append(b.shape[0])

    delta_arr = np.full((N, D), np.nan) if D is not None else np.zeros((N, 0))
    for t, d in enumerate(deltas):
        if d is not None:
            delta_arr[t] = d

    return dict(delta=delta_arr, norm=np.array(norms), n0=np.array(n0s), n1=np.array(n1s))


def alignment_cosine(delta_h, delta_b, eps=1e-8):
    """cos(theta_t) between Delta h_t and Delta b_cb,t at each t_rel."""
    N = delta_h.shape[0]
    cos = np.full(N, np.nan)
    for t in range(N):
        h, b = delta_h[t], delta_b[t]
        if np.any(np.isnan(h)) or np.any(np.isnan(b)):
            continue
        nh, nb = np.linalg.norm(h), np.linalg.norm(b)
        if nh < eps or nb < eps:
            continue
        cos[t] = float(np.dot(h, b) / (nh * nb))
    return cos


# Decoders over time

def _make_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def decode_sample_over_time(by_t, key="hidden", extra_key=None, test_frac=0.3, seed=0, max_samples=4000):
    """Held-out logistic-regression AUC decoding sample_bit at each t_rel."""
    N = len(by_t)
    aucs = np.full(N, np.nan)
    rng = np.random.RandomState(seed)

    for t in range(N):
        x = by_t[t][key]
        if x is None:
            continue
        y = to_numpy(by_t[t]["sample_bit"])
        X = to_numpy(x)
        if extra_key is not None and by_t[t][extra_key] is not None:
            X = np.concatenate([X, to_numpy(by_t[t][extra_key])], axis=1)

        n = X.shape[0]
        if n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            X, y = X[idx], y[idx]

        idx_all = rng.permutation(X.shape[0])
        n_test = max(1, int(round(len(idx_all) * test_frac)))
        test_idx, train_idx = idx_all[:n_test], idx_all[n_test:]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        clf = _make_clf()
        clf.fit(X[train_idx], y_tr)
        aucs[t] = roc_auc_score(y_te, clf.predict_proba(X[test_idx])[:, 1])

    return aucs


# One-step CB intervention

def causal_one_step_cb_effect(model, by_t, t_rel, device="cpu", seed=0):
    """One-step h_{t+1} under CB conditions A (normal), B (zero), C (mean), D (shuffled)."""
    if not model.use_cb_bias:
        return None
    if t_rel >= len(by_t) - 1 or by_t[t_rel]["x_next"] is None:
        return None

    h_t = by_t[t_rel]["hidden"].to(device)
    x_next = by_t[t_rel]["x_next"].to(device)
    sample_bit = by_t[t_rel]["sample_bit"]

    with torch.no_grad():
        step_A = manual_step(model, h_t, x_next)
        b_cb_real = step_A["b_cb"]

        step_B = manual_step(model, h_t, x_next, force_no_cb=True)

        b_cb_mean = b_cb_real.mean(dim=0, keepdim=True).expand_as(b_cb_real)
        step_C = manual_step(model, h_t, x_next, b_cb_override=b_cb_mean)

        rng = np.random.RandomState(seed)
        perm = rng.permutation(h_t.shape[0])
        step_D = manual_step(model, h_t, x_next, b_cb_override=b_cb_real[perm])

    results = {}
    delta_h_t_arr = None
    if by_t[t_rel]["hidden"] is not None:
        d0 = delta_over_time([by_t[t_rel]], key="hidden")
        delta_h_t_arr = d0["delta"][0]

    m_t = None
    if delta_h_t_arr is not None and not np.any(np.isnan(delta_h_t_arr)) and np.linalg.norm(delta_h_t_arr) > 1e-8:
        m_t = torch.tensor(delta_h_t_arr / np.linalg.norm(delta_h_t_arr), dtype=h_t.dtype, device=device)

    for name, step in [("A_full", step_A), ("B_zero", step_B), ("C_mean", step_C), ("D_shuffled", step_D)]:
        h_next = step["h_next"]
        a_np, b_np = _split_by_sample(h_next, sample_bit)
        d = a_np.mean(axis=0) - b_np.mean(axis=0)
        entry = {"delta_next_norm": float(np.linalg.norm(d))}
        if m_t is not None:
            proj = float(torch.dot(torch.tensor(d, dtype=h_t.dtype), m_t.cpu()))
            entry["proj_on_mt"] = proj
        results[name] = entry

    results["cb_memory_effect"] = results["A_full"]["delta_next_norm"] - results["B_zero"]["delta_next_norm"]
    return results


# Full-rollout CB ablation

def rollout_with_cb_condition(model, seqs, condition, device="cpu", seed=0):
    """Full rollout under a CB condition ('full', 'zero', 'mean', 'shuffled'). Returns h_T [B,H]."""
    from analysis.model_stepping import zero_init_hidden

    seqs = seqs.to(device)
    T, B, _ = seqs.shape
    h = zero_init_hidden(model, B, device)

    if condition == "mean":
        # per-timestep CB output averaged across trials
        with torch.no_grad():
            h0 = h.clone()
            b_cb_means = []
            for t in range(T):
                step = manual_step(model, h0, seqs[t])
                b_cb_means.append(step["b_cb"].mean(dim=0, keepdim=True))
                h0 = step["h_next"]

    if condition == "shuffled":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(B)

    with torch.no_grad():
        for t in range(T):
            if condition == "full":
                step = manual_step(model, h, seqs[t])
            elif condition == "zero":
                step = manual_step(model, h, seqs[t], force_no_cb=True)
            elif condition == "mean":
                override = b_cb_means[t].expand(B, -1)
                step = manual_step(model, h, seqs[t], b_cb_override=override)
            elif condition == "shuffled":
                real_step = manual_step(model, h, seqs[t])
                override = real_step["b_cb"][perm]
                step = manual_step(model, h, seqs[t], b_cb_override=override)
            else:
                raise ValueError(condition)
            h = step["h_next"]

    return h


def rollout_accuracy_with_cb_condition(model, batches, condition, head_idx=0, device="cpu", seed=0):
    """Accuracy of the model's head_idx readout under one CB condition,
    pooled over all fixed batches."""
    correct, total = 0, 0
    for b in batches:
        h_final = rollout_with_cb_condition(model, b["seqs"], condition, device=device, seed=seed)
        with torch.no_grad():
            logits = model.heads[head_idx](h_final)
        preds = logits.argmax(dim=-1).cpu()
        correct += int((preds == b["label"]).sum().item())
        total += b["label"].numel()
    return correct / total if total > 0 else float("nan")
