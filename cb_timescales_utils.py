from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit


# ============================================================
# Threading
# ============================================================

def set_cpu_threads(
    omp_threads: str = "2",
    blas_threads: str = "2",
    torch_threads: int = 10,
) -> None:
    os.environ["OMP_NUM_THREADS"] = omp_threads
    os.environ["OPENBLAS_NUM_THREADS"] = blas_threads
    os.environ["MKL_NUM_THREADS"] = blas_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = blas_threads
    os.environ["NUMEXPR_NUM_THREADS"] = blas_threads
    torch.set_num_threads(torch_threads)


# ============================================================
# Config / checkpoint loading
# ============================================================

def load_run_config(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {run_dir}")
    with open(config_path, "r") as f:
        return json.load(f)


def activation_from_string(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "sigmoid":
        return nn.Sigmoid
    else:
        raise ValueError(f"Unrecognized activation function name: {name!r}")


def load_state_dict_from_checkpoint(ckpt_path: str | Path, device: str = "cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def load_state_dict_legacy(run_dir: str | Path, N: int, device: str = "cpu") -> dict:
    """
    Supports legacy checkpoint names like:
      rnn_N5_N5
      rnn_N5_N5.pt
    """
    run_dir = Path(run_dir)
    candidates = [
        run_dir / f"rnn_N{N}_N{N}",
        run_dir / f"rnn_N{N}_N{N}.pt",
    ]
    for p in candidates:
        if p.exists():
            return load_state_dict_from_checkpoint(p, device=device)
    raise FileNotFoundError(f"No legacy checkpoint found for N={N} in {run_dir}")


def find_available_Ns_legacy(run_dir: str | Path) -> List[int]:
    run_dir = Path(run_dir)
    Ns = []
    pat = re.compile(r"rnn_N(\d+)_N\d+(?:\.pt)?$")
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.fullmatch(p.name)
        if m is not None:
            Ns.append(int(m.group(1)))
    return sorted(set(Ns))


def find_multitask_checkpoints(run_dir: str | Path) -> List[dict]:
    """
    Finds multitask .pt checkpoints like:
      shared_multitask_ep207_N10.pt
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ directory found in {run_dir}")

    out = []
    pat = re.compile(r".*?_ep(\d+)_N(\d+)\.pt$")
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            out.append({
                "path": str(p),
                "epoch": int(m.group(1)),
                "N": int(m.group(2)),
                "name": p.name,
            })

    if not out:
        raise RuntimeError(f"No multitask .pt checkpoints found in {ckpt_dir}")

    return sorted(out, key=lambda x: x["N"])


# ============================================================
# Model build
# ============================================================

def build_model_from_config_and_state(cfg: dict, state_dict: dict, device: str = "cpu"):
    """
    Builds current ElmanRNNMultiHead from config.json + state_dict.
    """
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    alex_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'alex_crap'))

    sys.path.insert(0, parent_dir)
    sys.path.insert(0, os.path.join(parent_dir, 'src'))
    sys.path.insert(0, alex_dir)
    from models_cb import ElmanRNNMultiHead

    model_cfg = cfg["model_config"]["model"]
    cli_args = cfg.get("cli_args", {})
    afunc = activation_from_string(model_cfg["activation"])

    if model_cfg["model_type"] != "ElmanRNNMultiHead":
        raise ValueError(f"Unsupported model_type in config: {model_cfg['model_type']}")

    hidden_size = model_cfg["hidden_size"]
    input_size = model_cfg["input_size"]
    use_cb_bias = model_cfg["use_cb_bias"]

    if use_cb_bias:
        if "cb.gc.weight" in state_dict:
            cb_gc_dim = state_dict["cb.gc.weight"].shape[0]
            cb_pc_dim = state_dict["cb.pc.weight"].shape[0]
            cb_dcn_dim = state_dict["cb.dcn.weight"].shape[0]
            cb_input_size = state_dict["cb.gc.weight"].shape[1] - hidden_size
        else:
            cb_gc_dim = cli_args.get("gc_dim", 256)
            cb_pc_dim = cli_args.get("pc_dim", 64)
            cb_dcn_dim = cli_args.get("dcn_dim", 64)
            cb_sees_input = cli_args.get("cb_sees_input", False)
            cb_input_size = input_size if cb_sees_input else 0
    else:
        cb_gc_dim = 0
        cb_pc_dim = 64
        cb_dcn_dim = 64
        cb_input_size = 0

    model = ElmanRNNMultiHead(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=model_cfg["num_classes"],
        num_readout_heads=model_cfg["num_readout_heads"],
        tau=model_cfg["tau"],
        scramble=model_cfg["scramble"],
        afunc=afunc,
        bias=model_cfg["bias"],
        use_cb_bias=use_cb_bias,
        cb_gc_dim=cb_gc_dim,
        cb_pc_dim=cb_pc_dim,
        cb_dcn_dim=cb_dcn_dim,
        cb_input_size=cb_input_size,
        multiply=model_cfg["multiply"],
        rnn_eat=model_cfg["rnn_eat"],
        rnn_eat_lambda=model_cfg["rnn_eat_lambda"] if model_cfg["rnn_eat_lambda"] is not None else 0.1,
        debug_stats=model_cfg.get("debug_stats", False),
        train_tau=model_cfg["train_tau"],
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# Input builders
# ============================================================

def make_random_binary_input(T: int, batch_size: int, input_size: int = 1, device: str = "cpu") -> torch.Tensor:
    """
    Generic stationary binary input for single-task models with input_size=1.
    """
    if input_size != 1:
        raise ValueError(
            f"Default random input builder only supports input_size=1, got {input_size}. "
            "Pass a custom input_builder instead."
        )
    return (torch.rand(T, batch_size, 1, device=device) < 0.5).float()


def make_random_multitask_input(T: int, batch_size: int, device: str = "cpu") -> torch.Tensor:
    """
    For multitask models trained with shared binary sequences and no task cue.
    Shape: [T, B, 1]
    """
    return (torch.rand(T, batch_size, 1, device=device) < 0.5).float()


# ============================================================
# Long dynamics collection
# ============================================================

def collect_long_dynamics(
    model,
    T: int = 20000,
    batch_size: int = 8,
    burn_T: int = 500,
    device: str = "cpu",
    input_builder: Optional[Callable[[int, int, str], torch.Tensor]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Simulate long activity and return post-burn-in dynamics.

    Assumes the model forward supports:
        model(x, return_timewise=False, return_dynamics=True)
    and returns (_, _, dyn)

    dyn should contain some subset of:
        hidden, gc, pc, dcn, cb_bias

    Returns arrays of shape [T_eff, B, D].
    """
    model.eval()

    if input_builder is None:
        input_builder = lambda T, B, device: make_random_binary_input(
            T=T,
            batch_size=B,
            input_size=model.input_size,
            device=device,
        )

    x = input_builder(T, batch_size, device)

    with torch.no_grad():
        _, _, dyn = model(
            x,
            return_timewise=False,
            return_dynamics=True,
        )

    out = {}
    for key in ["hidden", "gc", "pc", "dcn", "cb_bias"]:
        arr = dyn.get(key, None)
        out[key] = None if arr is None else arr[burn_T:].detach().cpu().numpy()

    return out


# ============================================================
# AC computation + fitting
# ============================================================

def comp_ac_fft(data: np.ndarray) -> np.ndarray:
    """
    data: [n_trials, T]
    Returns average non-normalized autocorrelation across trials.
    """
    n = data.shape[1]
    xp = data - data.mean(axis=1, keepdims=True)
    xp = np.concatenate((xp, np.zeros_like(xp)), axis=1)
    f = np.fft.fft(xp, axis=1)
    p = np.abs(f) ** 2
    pi = np.fft.ifft(p, axis=1)
    ac_all = np.real(pi)[:, : n - 1] / np.arange(1, n)[::-1]
    return np.mean(ac_all, axis=0)


def single_exp(time, a, tau):
    return a * np.exp(-time / tau)


def double_exp(time, a, tau1, tau2, coeff):
    return a * coeff * np.exp(-time / tau1) + a * (1 - coeff) * np.exp(-time / tau2)


def model_comp(ac: np.ndarray, lags: np.ndarray, min_lag: int, max_lag: int):
    """
    Fit single vs double exponential to normalized AC.
    Returns:
        selected_model: 1 or 2 or nan
        selected_tau: scalar or sorted [fast, slow]
    """
    xdata = lags[min_lag:max_lag + 1]
    ydata = ac[min_lag:max_lag + 1] / ac[0]

    AIC_1 = 1e5
    AIC_2 = 1e5
    popt_1 = None
    popt_2 = None

    try:
        popt_1, _ = curve_fit(
            single_exp,
            xdata,
            ydata,
            maxfev=3000,
            bounds=((0, 0), (1.0, 500.0)),
        )
        yfit = single_exp(xdata, *popt_1)
        RSS = ((yfit - ydata) ** 2).sum()
        n = len(xdata)
        k = 2
        AIC_1 = 2 * k + n * np.log(RSS / n)
    except Exception:
        pass

    try:
        popt_2, _ = curve_fit(
            double_exp,
            xdata,
            ydata,
            maxfev=5000,
            bounds=((0, 0, 0, 0), (1.0, 500.0, 500.0, 1.0)),
        )
        yfit = double_exp(xdata, *popt_2)
        RSS = ((yfit - ydata) ** 2).sum()
        n = len(xdata)
        k = 4
        AIC_2 = 2 * k + n * np.log(RSS / n)
    except Exception:
        pass

    if AIC_1 < AIC_2 and popt_1 is not None:
        return 1, float(popt_1[1])
    elif AIC_2 < AIC_1 and popt_2 is not None:
        taus = np.sort([popt_2[1], popt_2[2]])
        return 2, taus
    else:
        return np.nan, np.nan


def compute_module_ac_and_taus(
    data_all: Optional[np.ndarray],
    max_lag: int = 200,
    fit_lag: int = 30,
) -> Optional[dict]:
    """
    data_all: [T, B, D]
    Returns:
      {
        ac_pop,
        ac_all,
        taus_net,
        selected_models,
        mean_activity,
        std_activity,
      }
    """
    if data_all is None:
        return None

    T_eff, B, D = data_all.shape
    lags = np.arange(0, max_lag)
    min_lag = 0

    # population AC
    pop_activity = np.transpose(np.sum(data_all, axis=2))   # [B, T]
    ac_pop = comp_ac_fft(pop_activity)[:max_lag]

    # single-unit AC + tau fit
    ac_all = np.zeros((D, max_lag), dtype=float)
    taus_net = np.zeros(D, dtype=float)
    selected_models = np.zeros(D, dtype=float)

    for j in range(D):
        unit_data = np.transpose(data_all[:, :, j])  # [B, T]
        ac = comp_ac_fft(unit_data)[:max_lag]
        ac_all[j, :] = ac

        model_id, tau = model_comp(ac, lags, min_lag, fit_lag)
        selected_models[j] = model_id

        if model_id == 1:
            taus_net[j] = tau
        elif model_id == 2:
            taus_net[j] = tau[1]  # slow timescale
        else:
            taus_net[j] = np.nan

    return {
        "ac_pop": ac_pop,
        "ac_all": ac_all,
        "taus_net": taus_net,
        "selected_models": selected_models,
        "mean_activity": data_all.mean(axis=(0, 1)),
        "std_activity": data_all.std(axis=(0, 1)),
    }


# ============================================================
# One-checkpoint analysis
# ============================================================

def analyze_checkpoint_timescales(
    run_dir: str | Path,
    state_dict: dict,
    T: int = 20000,
    batch_size: int = 8,
    burn_T: int = 500,
    max_lag: int = 200,
    fit_lag: int = 30,
    device: str = "cpu",
    input_builder: Optional[Callable[[int, int, str], torch.Tensor]] = None,
) -> dict:
    cfg = load_run_config(run_dir)
    model = build_model_from_config_and_state(cfg, state_dict, device=device)

    dyn = collect_long_dynamics(
        model=model,
        T=T,
        batch_size=batch_size,
        burn_T=burn_T,
        device=device,
        input_builder=input_builder,
    )

    module_results = {}
    for module_name, arr in dyn.items():
        module_results[module_name] = compute_module_ac_and_taus(
            arr,
            max_lag=max_lag,
            fit_lag=fit_lag,
        )

    tau_value = getattr(model, "tau_param", None)
    shared_tau = float(tau_value.detach().cpu().item()) if tau_value is not None else np.nan

    return {
        "modules": module_results,
        "shared_tau": shared_tau,
        "duration": T - burn_T,
        "trials": batch_size,
        "max_fit_lag": fit_lag,
        "max_lag": max_lag,
        "T_total": T,
        "burn_T": burn_T,
    }


# ============================================================
# Saving helpers
# ============================================================

def save_timescale_result(result: dict, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


def make_default_save_name(run_dir: str | Path, N: int, tag: str = "timescales") -> str:
    run_dir = Path(run_dir)
    return str(run_dir / f"{tag}_N{N}.pkl")