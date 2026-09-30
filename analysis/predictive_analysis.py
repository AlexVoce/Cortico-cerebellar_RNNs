import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

from analysis.cb_ablation import find_available_Ns
from analysis.pca_dimensionality import collect_activity_for_n
from analysis.rebuild_model_utils import load_run_config, load_state_dict, build_model_from_config_and_state

DEFAULT_ALPHAS = np.logspace(-1, 4, 11)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


# Dynamics collection

def collect_dynamics_for_n(
    run_path,
    N,
    batch_fn,
    batch_size=64,
    n_batches=20,
    head_idx=0,
    device=None,
):
    """Collect time-resolved dynamics for one run and N."""
    device = device or default_device()

    cfg = load_run_config(run_path)
    sd = load_state_dict(run_path, N)

    model = build_model_from_config_and_state(cfg=cfg, state_dict=sd, device=device)
    model.eval()

    return collect_activity_for_n(
        model=model,
        batch_fn=batch_fn,
        eval_n=N,
        batch_size=batch_size,
        n_batches=n_batches,
        head_idx=head_idx,
        device=device,
    )


# Lagged design matrices and forward-prediction scoring

def _flatten_lagged_batches(dyn_list, x_key, z_key, lag):
    """Pool (t, b) pairs for one lag. Returns X (state at t), Z (CB at t, or None), Y (state at t + lag)."""
    X_chunks, Z_chunks, Y_chunks = [], [], []

    for dyn in dyn_list:
        h = _to_numpy(dyn[x_key])
        T = h.shape[0]
        if T <= lag:
            continue

        h_t = h[:T - lag]
        h_tk = h[lag:]
        Th, B, Dh = h_t.shape
        X_chunks.append(h_t.reshape(Th * B, Dh))
        Y_chunks.append(h_tk.reshape(Th * B, Dh))

        if z_key is not None:
            z = _to_numpy(dyn[z_key])
            z_t = z[:T - lag]
            Z_chunks.append(z_t.reshape(Th * B, z_t.shape[-1]))

    if len(X_chunks) == 0:
        return None

    X = np.concatenate(X_chunks, axis=0)
    Y = np.concatenate(Y_chunks, axis=0)
    Z = np.concatenate(Z_chunks, axis=0) if z_key is not None else None
    return X, Z, Y


def _split_batches_train_test(n_batches, test_frac=0.3, random_state=0):
    """Split batch indices into train/test."""
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(n_batches)
    n_test = max(1, int(round(n_batches * test_frac)))
    test_idx = sorted(idx[:n_test].tolist())
    train_idx = [i for i in range(n_batches) if i not in set(test_idx)]
    return train_idx, test_idx


def forward_prediction_scores_for_lag(
    dyn_train,
    dyn_test,
    x_key="hidden",
    z_key="cb_bias",
    lag=1,
    alphas=DEFAULT_ALPHAS,
    cv=5,
    n_permutations=0,
    random_state=0,
):
    """Held-out ridge R^2 predicting x[t+lag] from x[t] alone vs x[t] plus z[t].

        Optionally adds a permutation null for delta_r2.
    
    """
    train = _flatten_lagged_batches(dyn_train, x_key, z_key, lag)
    test = _flatten_lagged_batches(dyn_test, x_key, z_key, lag)
    if train is None or test is None:
        return None

    X_tr, Z_tr, Y_tr = train
    X_te, Z_te, Y_te = test

    model_A = RidgeCV(alphas=alphas, cv=cv)
    model_A.fit(X_tr, Y_tr)
    r2_A = r2_score(Y_te, model_A.predict(X_te), multioutput="variance_weighted")

    XZ_tr = np.concatenate([X_tr, Z_tr], axis=1)
    XZ_te = np.concatenate([X_te, Z_te], axis=1)
    model_B = RidgeCV(alphas=alphas, cv=cv)
    model_B.fit(XZ_tr, Y_tr)
    r2_B = r2_score(Y_te, model_B.predict(XZ_te), multioutput="variance_weighted")

    alpha_min = float(np.min(alphas))

    result = {
        "lag": lag,
        "r2_state_only": float(r2_A),
        "r2_state_plus_cb": float(r2_B),
        "delta_r2": float(r2_B - r2_A),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "alpha_A": float(model_A.alpha_),
        "alpha_B": float(model_B.alpha_),
        # True if the selected alpha is at the low edge of the grid
        "alpha_at_grid_floor": bool(model_A.alpha_ <= alpha_min or model_B.alpha_ <= alpha_min),
    }

    if n_permutations > 0:
        rng = np.random.RandomState(random_state)
        null_deltas = np.empty(n_permutations)
        # reuse model B's alpha for the permutation fits
        perm_alpha = float(model_B.alpha_)

        for i in range(n_permutations):
            perm_tr = rng.permutation(Z_tr.shape[0])
            perm_te = rng.permutation(Z_te.shape[0])

            model_perm = Ridge(alpha=perm_alpha)
            model_perm.fit(np.concatenate([X_tr, Z_tr[perm_tr]], axis=1), Y_tr)
            r2_perm = r2_score(
                Y_te,
                model_perm.predict(np.concatenate([X_te, Z_te[perm_te]], axis=1)),
                multioutput="variance_weighted",
            )
            null_deltas[i] = r2_perm - r2_A

        result["null_delta_r2_mean"] = float(null_deltas.mean())
        result["null_delta_r2_std"] = float(null_deltas.std(ddof=1)) if n_permutations > 1 else 0.0
        result["p_value"] = float((np.sum(null_deltas >= result["delta_r2"]) + 1) / (n_permutations + 1))

    return result


def forward_prediction_for_n(
    run_path,
    N,
    batch_fn,
    x_key="hidden",
    z_key="cb_bias",
    lags=(1, 2, 4, 8),
    batch_size=64,
    n_batches=20,
    test_frac=0.3,
    alphas=DEFAULT_ALPHAS,
    cv=5,
    n_permutations=0,
    head_idx=0,
    device=None,
    random_state=0,
):
    """Forward-prediction results over lags for one run and N."""
    device = device or default_device()
    output = collect_dynamics_for_n(
        run_path=run_path,
        N=N,
        batch_fn=batch_fn,
        batch_size=batch_size,
        n_batches=n_batches,
        head_idx=head_idx,
        device=device,
    )

    dyn_list = output["dynamics"]
    train_idx, test_idx = _split_batches_train_test(
        len(dyn_list), test_frac=test_frac, random_state=random_state,
    )
    dyn_train = [dyn_list[i] for i in train_idx]
    dyn_test = [dyn_list[i] for i in test_idx]

    rows = []
    for lag in lags:
        res = forward_prediction_scores_for_lag(
            dyn_train=dyn_train,
            dyn_test=dyn_test,
            x_key=x_key,
            z_key=z_key,
            lag=lag,
            alphas=alphas,
            cv=cv,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        if res is None:
            continue
        res["N"] = N
        rows.append(res)

    return rows


# Across Ns / runs

def summarize_run_forward_prediction(
    run_path,
    batch_fn,
    x_key="hidden",
    z_key="cb_bias",
    lags=(1, 2, 4, 8),
    batch_size=64,
    n_batches=20,
    test_frac=0.3,
    alphas=DEFAULT_ALPHAS,
    cv=5,
    n_permutations=0,
    head_idx=0,
    device=None,
    run_id=None,
    random_state=0,
):
    device = device or default_device()
    if run_id is None:
        run_id = os.path.basename(str(run_path).rstrip("/"))

    available_Ns = find_available_Ns(run_path)
    rows = []

    for N in available_Ns:
        n_rows = forward_prediction_for_n(
            run_path=run_path,
            N=N,
            batch_fn=batch_fn,
            x_key=x_key,
            z_key=z_key,
            lags=lags,
            batch_size=batch_size,
            n_batches=n_batches,
            test_frac=test_frac,
            alphas=alphas,
            cv=cv,
            n_permutations=n_permutations,
            head_idx=head_idx,
            device=device,
            random_state=random_state,
        )
        for r in n_rows:
            r["run_id"] = run_id
            r["run_path"] = str(run_path)
            r["x_key"] = x_key
            r["z_key"] = z_key
            rows.append(r)

    return pd.DataFrame(rows)


def summarize_multi_runs_forward_prediction(
    run_paths,
    batch_fn,
    x_key="hidden",
    z_key="cb_bias",
    lags=(1, 2, 4, 8),
    batch_size=64,
    n_batches=20,
    test_frac=0.3,
    alphas=DEFAULT_ALPHAS,
    cv=5,
    n_permutations=0,
    head_idx=0,
    device=None,
    random_state=0,
):
    device = device or default_device()
    dfs = []

    for i, run_path in enumerate(run_paths):
        run_id = os.path.basename(str(run_path).rstrip("/"))
        print(f"Processing run {i+1}/{len(run_paths)}: {run_id}")

        dfs.append(summarize_run_forward_prediction(
            run_path=run_path,
            batch_fn=batch_fn,
            x_key=x_key,
            z_key=z_key,
            lags=lags,
            batch_size=batch_size,
            n_batches=n_batches,
            test_frac=test_frac,
            alphas=alphas,
            cv=cv,
            n_permutations=n_permutations,
            head_idx=head_idx,
            device=device,
            run_id=run_id,
            random_state=random_state,
        ))

    return pd.concat(dfs, ignore_index=True)


# Plotting

def plot_delta_r2_vs_lag(
    df,
    group_col=None,
    ylabel="delta R2 (state+CB vs state only)",
    title=None,
    figsize=(4, 3),
    save_path=None,
):
    """Mean delta_r2 vs lag, optionally split by group_col, with SEM bands."""
    fig, ax = plt.subplots(figsize=figsize)

    if group_col is None:
        grouped = df.groupby("lag", as_index=False).agg(
            mean_val=("delta_r2", "mean"),
            sem_val=("delta_r2", "sem"),
        )
        x = grouped["lag"].values.astype(float)
        y = grouped["mean_val"].values.astype(float)
        yerr = grouped["sem_val"].fillna(0).values.astype(float)
        ax.plot(x, y, marker="o", linewidth=1.4, markersize=3, color="#0553CF")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.18, color="#0553CF")
    else:
        for group_val, sub in df.groupby(group_col):
            grouped = sub.groupby("lag", as_index=False).agg(
                mean_val=("delta_r2", "mean"),
                sem_val=("delta_r2", "sem"),
            )
            x = grouped["lag"].values.astype(float)
            y = grouped["mean_val"].values.astype(float)
            yerr = grouped["sem_val"].fillna(0).values.astype(float)
            ax.plot(x, y, marker="o", linewidth=1.2, markersize=2.5, label=str(group_val))
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)
        ax.legend(fontsize=6, frameon=False, title=group_col)

    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Lag (timesteps)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()

    return fig, ax


def _select_Ns(available_Ns, max_Ns=None):
    """Optionally subsample curriculum levels to at most max_Ns."""
    if max_Ns is None or len(available_Ns) <= max_Ns:
        return available_Ns
    idx = np.linspace(0, len(available_Ns) - 1, max_Ns).round().astype(int)
    return [available_Ns[i] for i in sorted(set(idx))]
