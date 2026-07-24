# analysis/sparsity_analysis.py
from analysis.ablation import find_available_Ns
from analysis.activity_analysis import collect_activity_for_n, summarize_multi_runs_pca_for_multi_keys
import numpy as np

from analysis.rebuild_model_utils import load_run_config

def population_sparseness(r, eps=1e-8):
    """Treves-Rolls sparseness for one sample's activations across units.
    r: (n_units,) non-negative activation vector. Returns 0 (dense) to 1 (sparse)."""
    n = r.shape[0]
    r = np.clip(r, 0, None)
    mean_r, mean_r2 = r.mean(), (r ** 2).mean()
    if mean_r2 < eps:
        return np.nan
    return (1 - mean_r**2 / mean_r2) / (1 - 1/n)

def lifetime_sparseness(unit_trace, eps=1e-8):
    """Same formula, but across samples/time for one unit instead of across units."""
    return population_sparseness(unit_trace, eps)

def sparsity_over_checkpoint(activity):
    """activity: (T, B, D) GC activity from one checkpoint.
    Returns (mean population sparseness, mean lifetime sparseness)."""
    flat = activity.reshape(-1, activity.shape[-1])       # (T*B, D)
    pop_s = np.nanmean([population_sparseness(r) for r in flat])
    life_s = np.nanmean([lifetime_sparseness(flat[:, u]) for u in range(flat.shape[1])])
    return pop_s, life_s

# collect activity for Ns and compute sparsity metrics
def collect_sparsity_for_n(run_path, N):
    activity = collect_activity_for_n(run_path, N)
    pop_s, life_s = sparsity_over_checkpoint(activity)
    return {
        "N": N,
        "population_sparseness": pop_s,
        "lifetime_sparseness": life_s,
    }
# do for all Ns in a run and summarize
def summarize_multi_runs_sparsity_for_multi_keys(run_paths, keys):
    rows = []
    for run_path in run_paths:
        run_config = load_run_config(run_path)
        available_Ns = find_available_Ns(run_path)
        for N in available_Ns:
            r = collect_sparsity_for_n(run_path, N)
            rows.append({
                **{k: run_config[k] for k in keys},
                "N": r["N"],
                "population_sparseness": r["population_sparseness"],
                "lifetime_sparseness": r["lifetime_sparseness"],
            })
    return rows