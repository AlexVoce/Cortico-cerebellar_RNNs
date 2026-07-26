# analysis/sparsity_analysis.py
import os
import numpy as np
import pandas as pd
import torch

from analysis.ablation import find_available_Ns
from analysis.activity_analysis import collect_activity_for_n, flatten_activity_from_output, summarize_multi_runs_pca_for_multi_keys
from analysis.rebuild_model_utils import load_run_config

def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

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

from analysis.rebuild_model_utils import load_state_dict, build_model_from_config_and_state
# collect activity for Ns and compute sparsity metrics
def collect_sparsity_for_n(run_path, N, batch_fn, key="gc", device=None):
    device = device or default_device()

    cfg = load_run_config(run_path)
    sd = load_state_dict(run_path,N)

    model = build_model_from_config_and_state(
        cfg=cfg,
        state_dict=sd,
        device=device,
    )
    model.eval()

    output = collect_activity_for_n(
    model=model,
    batch_fn=batch_fn,
    eval_n=N,
    batch_size=64,
    n_batches=10,
    head_idx=0,
    device=device,
    )
    activity = flatten_activity_from_output(output, key=key).detach().cpu().numpy()
    print(activity.shape)
    pop_s, life_s = sparsity_over_checkpoint(activity)
    return {
        "N": N,
        "population_sparseness": pop_s,
        "lifetime_sparseness": life_s,
    }
# do for all Ns in a run, for one or more activity populations (keys)
def summarize_run_sparsity_for_multiple_keys(
    run_path,
    batch_fn,
    keys=("hidden", "gc", "cb_bias"),
    device=None,
    run_id=None,
):
    device = device or default_device()
    if run_id is None:
        run_id = os.path.basename(str(run_path).rstrip("/"))

    available_Ns = find_available_Ns(run_path)
    rows = []
    for key in keys:
        for N in available_Ns:
            r = collect_sparsity_for_n(run_path, N, batch_fn, key=key, device=device)
            rows.append({
                "run_id": run_id,
                "run_path": str(run_path),
                "population": key,
                "N": r["N"],
                "population_sparseness": r["population_sparseness"],
                "lifetime_sparseness": r["lifetime_sparseness"],
            })
    return pd.DataFrame(rows)


# do the above across multiple runs and concatenate
def summarize_multi_runs_sparsity_for_multi_keys(
    run_paths,
    batch_fn,
    keys=("hidden", "gc", "cb_bias"),
    device=None,
):
    device = device or default_device()
    dfs = []
    for i, run_path in enumerate(run_paths):
        run_id = os.path.basename(str(run_path).rstrip("/"))
        print(f"Processing run {i+1}/{len(run_paths)}: {run_id}")

        dfs.append(summarize_run_sparsity_for_multiple_keys(
            run_path=run_path,
            batch_fn=batch_fn,
            keys=keys,
            device=device,
            run_id=run_id,
        ))
    return pd.concat(dfs, ignore_index=True)
