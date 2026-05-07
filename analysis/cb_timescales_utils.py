from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
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
    name = str(name).lower()

    if name == "relu":
        return nn.ReLU
    if name == "leakyrelu":
        return nn.LeakyReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid

    raise ValueError(f"Unrecognized activation function name: {name!r}")


def load_state_dict_from_checkpoint(ckpt_path: str | Path, device: str = "cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    return ckpt


def load_state_dict_legacy(run_dir: str | Path, N: int, device: str = "cpu") -> dict:
    """
    Supports checkpoint names:
        rnn_N5_N5
        rnn_N5_N5.pt
    """
    run_dir = Path(run_dir)

    candidates = [
        run_dir / f"rnn_N{N}_N{N}",
        run_dir / f"rnn_N{N}_N{N}.pt",
    ]

    for path in candidates:
        if path.exists():
            return load_state_dict_from_checkpoint(path, device=device)

    raise FileNotFoundError(f"No checkpoint found for N={N} in {run_dir}")


def find_available_Ns_legacy(run_dir: str | Path) -> List[int]:
    run_dir = Path(run_dir)
    Ns = []

    pattern = re.compile(r"rnn_N(\d+)_N\d+(?:\.pt)?$")

    for path in run_dir.iterdir():
        if not path.is_file():
            continue

        match = pattern.fullmatch(path.name)

        if match is not None:
            Ns.append(int(match.group(1)))

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
    pattern = re.compile(r".*?_ep(\d+)_N(\d+)\.pt$")

    for path in ckpt_dir.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(path.name)

        if match:
            out.append(
                {
                    "path": str(path),
                    "epoch": int(match.group(1)),
                    "N": int(match.group(2)),
                    "name": path.name,
                }
            )

    if not out:
        raise RuntimeError(f"No multitask .pt checkpoints found in {ckpt_dir}")

    return sorted(out, key=lambda x: x["N"])


# ============================================================
# Model build
# ============================================================

def _infer_model_type_from_state_dict(state_dict: dict) -> str:
    """
    Infer cleaned model type from state_dict keys.
    """
    keys = set(state_dict.keys())

    if "inp.weight" in keys and "hh.weight" in keys:
        return "elman"

    if "gru_cell.x_z.weight" in keys:
        return "gru"

    raise KeyError("Could not infer model type from state_dict keys.")


def _normalise_model_type(model_type) -> Optional[str]:
    if model_type is None:
        return None

    model_type = str(model_type).lower()

    if model_type in {"elman", "elmanrnnmultihead"}:
        return "elman"

    if model_type in {"gru", "grumultiheadwithcb"}:
        return "gru"

    return model_type


def _get_model_config(cfg: dict) -> dict:
    """
    Supports both:
        cfg["model_config"]["model"]
    and:
        cfg["model"]
    """
    if "model_config" in cfg and isinstance(cfg["model_config"], dict):
        if "model" in cfg["model_config"]:
            return cfg["model_config"]["model"]

    if "model" in cfg and isinstance(cfg["model"], dict):
        return cfg["model"]

    return {}


def _get_head_keys(state_dict: dict) -> List[str]:
    return sorted(
        [k for k in state_dict.keys() if re.fullmatch(r"heads\.\d+\.weight", k)],
        key=lambda x: int(x.split(".")[1]),
    )


def _infer_core_dims(model_type: str, state_dict: dict) -> tuple[int, int]:
    """
    Returns hidden_size, input_size.
    """
    if model_type == "elman":
        return state_dict["inp.weight"].shape

    if model_type == "gru":
        return state_dict["gru_cell.x_z.weight"].shape

    raise ValueError(f"Unsupported model_type={model_type}")


def _infer_cb_dims(
    *,
    state_dict: dict,
    hidden_size: int,
    input_size: int,
    model_cfg: dict,
    cli_args: dict,
    use_cb_bias: bool,
) -> tuple[int, int, int, bool]:
    """
    Returns:
        cb_gc_dim, cb_pc_dim, cb_input_size, cb_no_hidden
    """
    if not use_cb_bias:
        return 128, 64, 0, False

    if "cb.gc.weight" in state_dict:
        cb_gc_dim = int(state_dict["cb.gc.weight"].shape[0])
        cb_gc_in = int(state_dict["cb.gc.weight"].shape[1])

        if "cb.pc.weight" in state_dict:
            cb_pc_dim = int(state_dict["cb.pc.weight"].shape[0])
        else:
            cb_pc_dim = int(model_cfg.get("cb_pc_dim", cli_args.get("pc_dim", 64)))

        if cb_gc_in == hidden_size:
            cb_input_size = 0
            cb_no_hidden = False

        elif cb_gc_in > hidden_size:
            cb_input_size = cb_gc_in - hidden_size
            cb_no_hidden = False

        else:
            cb_input_size = cb_gc_in
            cb_no_hidden = True

        return cb_gc_dim, cb_pc_dim, cb_input_size, cb_no_hidden

    cb_gc_dim = int(model_cfg.get("cb_gc_dim", cli_args.get("gc_dim", 256)))
    cb_pc_dim = int(model_cfg.get("cb_pc_dim", cli_args.get("pc_dim", 64)))

    cb_no_hidden = bool(
        model_cfg.get("cb_no_hidden", cli_args.get("cb_no_hidden", False))
    )
    cb_sees_input = bool(cli_args.get("cb_sees_input", False))

    cb_input_size = int(
        model_cfg.get(
            "cb_input_size",
            input_size if (cb_sees_input or cb_no_hidden) else 0,
        )
    )

    return cb_gc_dim, cb_pc_dim, cb_input_size, cb_no_hidden


def build_model_from_config_and_state(cfg: dict, state_dict: dict, device: str = "cpu"):
    """
    Build a cleaned ElmanRNNMultiHead or GRUMultiHeadWithCB from config.json
    and a saved state_dict.
    """
    from model.models_cb import ElmanRNNMultiHead
    from model.models_gru import GRUMultiHeadWithCB

    model_cfg = _get_model_config(cfg)
    cli_args = cfg.get("cli_args", {})

    inferred_model_type = _infer_model_type_from_state_dict(state_dict)
    saved_model_type = _normalise_model_type(model_cfg.get("model_type", None))
    model_type = saved_model_type if saved_model_type is not None else inferred_model_type

    if model_type not in {"elman", "gru"}:
        raise ValueError(f"Unsupported model_type in config/state_dict: {model_type}")

    hidden_size_from_sd, input_size_from_sd = _infer_core_dims(model_type, state_dict)

    head_keys = _get_head_keys(state_dict)

    if len(head_keys) == 0:
        raise KeyError("No readout head weights found in state_dict.")

    num_readout_heads_from_sd = len(head_keys)
    num_classes_from_sd = state_dict[head_keys[0]].shape[0]

    input_size = int(model_cfg.get("input_size", input_size_from_sd))
    hidden_size = int(model_cfg.get("hidden_size", hidden_size_from_sd))
    num_classes = int(model_cfg.get("num_classes", num_classes_from_sd))
    num_readout_heads = int(
        model_cfg.get("num_readout_heads", num_readout_heads_from_sd)
    )

    tau = float(model_cfg.get("tau", 1.5))
    bias = bool(model_cfg.get("bias", True))

    activation_name = model_cfg.get(
        "activation",
        model_cfg.get(
            "candidate_afunc",
            cli_args.get("afunc", "leakyrelu" if model_type == "elman" else "tanh"),
        ),
    )
    afunc = activation_from_string(activation_name)

    use_cb_bias = bool(
        model_cfg.get(
            "use_cb_bias",
            any(k.startswith("cb.") for k in state_dict.keys()),
        )
    )

    cb_gc_dim, cb_pc_dim, cb_input_size, cb_no_hidden = _infer_cb_dims(
        state_dict=state_dict,
        hidden_size=hidden_size,
        input_size=input_size,
        model_cfg=model_cfg,
        cli_args=cli_args,
        use_cb_bias=use_cb_bias,
    )

    cb_max_ratio = float(
        model_cfg.get(
            "cb_max_ratio",
            cli_args.get("cb_max_ratio", 1.0),
        )
    )

    if model_type == "elman":
        model = ElmanRNNMultiHead(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_readout_heads=num_readout_heads,
            tau=tau,
            afunc=afunc,
            bias=bias,
            use_cb_bias=use_cb_bias,
            cb_gc_dim=cb_gc_dim,
            cb_pc_dim=cb_pc_dim,
            cb_input_size=cb_input_size,
            cb_no_hidden=cb_no_hidden,
            cb_max_ratio=cb_max_ratio,
        ).to(device)

    else:
        model = GRUMultiHeadWithCB(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_readout_heads=num_readout_heads,
            candidate_afunc=afunc,
            bias=bias,
            use_cb_bias=use_cb_bias,
            cb_gc_dim=cb_gc_dim,
            cb_pc_dim=cb_pc_dim,
            cb_input_size=cb_input_size,
            cb_no_hidden=cb_no_hidden,
            cb_max_ratio=cb_max_ratio,
        ).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)

    except RuntimeError as err:
        print(
            "[warning] strict=True checkpoint loading failed. "
            "Retrying with strict=False. This is expected only for older checkpoints.",
            flush=True,
        )
        print(err, flush=True)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


# ============================================================
# Input builders
# ============================================================

def make_random_binary_input(
    T: int,
    batch_size: int,
    input_size: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generic stationary binary input for single-task models.
    """
    if input_size != 1:
        raise ValueError(
            f"Default random input builder only supports input_size=1, got {input_size}. "
            "Pass a custom input_builder instead."
        )

    return (torch.rand(T, batch_size, 1, device=device) < 0.5).float()


def make_random_multitask_input(
    T: int,
    batch_size: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    For multitask models trained with shared binary sequences and no task cue.
    Shape: [T, B, 1].
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

    for key in [
        "hidden",
        "gc",
        "pc",
        "dcn",
        "cb_bias",
        "z_gate",
        "r_gate",
        "candidate_pre",
        "candidate",
    ]:
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
    Fit single vs double exponential to normalized autocorrelation.

    Returns:
        selected_model: 1, 2, or nan
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

    if AIC_2 < AIC_1 and popt_2 is not None:
        taus = np.sort([popt_2[1], popt_2[2]])
        return 2, taus

    return np.nan, np.nan


def compute_module_ac_and_taus(
    data_all: Optional[np.ndarray],
    max_lag: int = 200,
    fit_lag: int = 30,
) -> Optional[dict]:
    """
    data_all: [T, B, D]
    """
    if data_all is None:
        return None

    T_eff, B, D = data_all.shape
    lags = np.arange(0, max_lag)
    min_lag = 0

    pop_activity = np.transpose(np.sum(data_all, axis=2))  # [B, T]
    ac_pop = comp_ac_fft(pop_activity)[:max_lag]

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
            taus_net[j] = tau[1]

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

    if hasattr(model, "tau"):
        shared_tau = float(model.tau)
    elif hasattr(model, "tau_param"):
        tau_value = getattr(model, "tau_param")
        shared_tau = float(tau_value.detach().cpu().item())
    else:
        shared_tau = np.nan

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