import os
import re
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch


def find_available_Ns(run_path):
    """
    Find all checkpoint Ns in a run directory for files named like:
      rnn_N12_N12
    """
    run_path = Path(run_path)
    Ns = []

    for p in run_path.iterdir():
        if not p.is_file():
            continue
        m = re.fullmatch(r"rnn_N(\d+)_N\d+", p.name)
        if m is not None:
            Ns.append(int(m.group(1)))

    return sorted(set(Ns))

def load_state_dict(run_path, N):
    """
    Loads a checkpoint file rnn_N{N}_N{N}. Handles either raw state_dict
    or dict with 'state_dict'.
    """
    p = os.path.join(run_path, f"rnn_N{N}_N{N}")
    sd = torch.load(p, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def find_available_Ns(run_path):
    """
    Find all checkpoint Ns in a run directory for files named like:
      rnn_N12_N12
    """
    run_path = Path(run_path)
    Ns = []

    for p in run_path.iterdir():
        if not p.is_file():
            continue
        m = re.fullmatch(r"rnn_N(\d+)_N\d+", p.name)
        if m is not None:
            Ns.append(int(m.group(1)))

    return sorted(set(Ns))


def get_weight_array(sd, key):
    """
    Extract one parameter from a loaded state_dict as a flat numpy array.
    Returns None if key is missing.
    """
    if key not in sd:
        return None
    w = sd[key]
    if isinstance(w, torch.Tensor):
        w = w.detach().cpu().numpy()
    return np.asarray(w, dtype=float).ravel()
def infer_model_dims_from_state_dict(sd):
    dims = {}

    # Input / hidden dims
    if "inp.weight" in sd:
        dims["hidden_size"], dims["input_size"] = sd["inp.weight"].shape
    elif "input.weight" in sd:
        dims["hidden_size"], dims["input_size"] = sd["input.weight"].shape
    else:
        raise KeyError("Could not infer input_size / hidden_size from state_dict")

    # Output / number of heads
    head_keys = sorted(
        [k for k in sd.keys() if re.fullmatch(r"heads\.\d+\.weight", k)],
        key=lambda x: int(x.split(".")[1])
    )
    dims["num_readout_heads"] = len(head_keys)

    if len(head_keys) == 0:
        raise KeyError("No head weights found in state_dict")

    first_head = head_keys[0]
    dims["num_classes"] = sd[first_head].shape[0]

    # CB dims if present
    dims["use_cb_bias"] = any(k.startswith("cb.") for k in sd.keys())

    if dims["use_cb_bias"]:
        dims["cb_gc_dim"] = sd["cb.gc.weight"].shape[0]
        dims["cb_pc_dim"] = sd["cb.pc.weight"].shape[0]
        dims["cb_dcn_dim"] = sd["cb.dcn.weight"].shape[0]
        dims["cb_input_size"] = sd["cb.gc.weight"].shape[1] - dims["hidden_size"]

    return dims
import torch.nn as nn
from models_cb import ElmanRNNMultiHead

def load_run_config(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in run dir: {run_dir}")
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

def build_model_from_config_and_state(cfg, state_dict, device):
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
            cb_input_size = state_dict["cb.gc.weight"].shape[1] - hidden_size
        else:
            cb_gc_dim = cli_args.get("gc_dim", 128)
            cb_sees_input = cli_args.get("cb_sees_input", False)
            cb_input_size = input_size if cb_sees_input else 0
    else:
        cb_gc_dim = 0
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
        cb_input_size=cb_input_size,
        multiply=model_cfg["multiply"],
        rnn_eat=model_cfg["rnn_eat"],
        rnn_eat_lambda=model_cfg["rnn_eat_lambda"] if model_cfg["rnn_eat_lambda"] is not None else 0.1,
        debug_stats=model_cfg.get("debug_stats", False),
        train_tau=model_cfg["train_tau"],
    ).to(device)

    return model