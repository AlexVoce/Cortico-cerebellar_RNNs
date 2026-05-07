import os
import re
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.models_cb import ElmanRNNMultiHead
from model.GRU_test import GRUMultiHeadWithCB

def _safe_int(value, default):
    if value is None:
        return default
    return int(value)

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

        match = re.fullmatch(r"rnn_N(\d+)_N\d+", p.name)

        if match is not None:
            Ns.append(int(match.group(1)))

    return sorted(set(Ns))


def load_state_dict(run_path, N):
    """
    Load checkpoint file rnn_N{N}_N{N}.

    Handles either a raw state_dict or a dict containing 'state_dict'.
    """
    path = os.path.join(run_path, f"rnn_N{N}_N{N}")
    state_dict = torch.load(path, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    return state_dict


def get_weight_array(state_dict, key):
    """
    Extract one parameter from a loaded state_dict as a flat numpy array.

    Returns None if key is missing.
    """
    if key not in state_dict:
        return None

    weight = state_dict[key]

    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()

    return np.asarray(weight, dtype=float).ravel()


def activation_from_string(name: str):
    """
    Convert saved activation-function name to torch module class.
    """
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


def load_run_config(run_dir):
    """
    Load config.json from a saved run directory.
    """
    config_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in run dir: {run_dir}")

    with open(config_path, "r") as f:
        return json.load(f)


def _head_keys(state_dict):
    return sorted(
        [k for k in state_dict.keys() if re.fullmatch(r"heads\.\d+\.weight", k)],
        key=lambda x: int(x.split(".")[1]),
    )


def _infer_model_type_from_state_dict(state_dict):
    """
    Infer whether the state_dict belongs to the cleaned Elman or GRU model.
    """
    keys = set(state_dict.keys())

    if "inp.weight" in keys and "hh.weight" in keys:
        return "elman"

    if "gru_cell.x_z.weight" in keys:
        return "gru"

    raise KeyError("Could not infer model type from state_dict keys.")


def infer_model_dims_from_state_dict(state_dict):
    """
    Infer model dimensions from a cleaned Elman/GRU state_dict.
    """
    dims = {}
    model_type = _infer_model_type_from_state_dict(state_dict)
    dims["model_type"] = model_type

    if model_type == "elman":
        dims["hidden_size"], dims["input_size"] = state_dict["inp.weight"].shape

    elif model_type == "gru":
        dims["hidden_size"], dims["input_size"] = state_dict["gru_cell.x_z.weight"].shape

    head_keys = _head_keys(state_dict)

    if len(head_keys) == 0:
        raise KeyError("No head weights found in state_dict.")

    dims["num_readout_heads"] = len(head_keys)
    dims["num_classes"] = state_dict[head_keys[0]].shape[0]

    dims["use_cb_bias"] = any(k.startswith("cb.") for k in state_dict.keys())

    if dims["use_cb_bias"]:
        dims["cb_gc_dim"] = state_dict["cb.gc.weight"].shape[0]
        dims["cb_pc_dim"] = state_dict["cb.pc.weight"].shape[0]

        cb_gc_in = state_dict["cb.gc.weight"].shape[1]

        if cb_gc_in == dims["hidden_size"]:
            dims["cb_input_size"] = 0
            dims["cb_no_hidden"] = False

        elif cb_gc_in > dims["hidden_size"]:
            dims["cb_input_size"] = cb_gc_in - dims["hidden_size"]
            dims["cb_no_hidden"] = False

        else:
            # Input-only CB case.
            dims["cb_input_size"] = cb_gc_in
            dims["cb_no_hidden"] = True

    else:
        dims["cb_gc_dim"] = None
        dims["cb_pc_dim"] = None
        dims["cb_input_size"] = 0
        dims["cb_no_hidden"] = False

    return dims


def _get_nested(dct, keys, default=None):
    """
    Safely get nested config values.
    """
    current = dct

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def _model_config_from_run_config(cfg):
    """
    Return model config dict from supported config layouts.
    """
    model_cfg = _get_nested(cfg, ["model_config", "model"], default=None)

    if model_cfg is not None:
        return model_cfg

    model_cfg = cfg.get("model", None)

    if model_cfg is not None:
        return model_cfg

    return {}

def _cli_args_from_run_config(cfg):
    return cfg.get("cli_args", {})


def clean_state_dict_for_current_model(state_dict):
    """
    Remove keys from older checkpoints that are no longer present in the cleaned model.
    """
    cleaned = dict(state_dict)
    cleaned.pop("tau_param", None)
    return cleaned


def build_model_from_config_and_state(cfg, state_dict, device="cpu", load_weights=True, strict=False):
    dims = infer_model_dims_from_state_dict(state_dict)

    model_cfg = _model_config_from_run_config(cfg)
    cli_args = _cli_args_from_run_config(cfg)

    inferred_type = dims["model_type"]
    saved_model_type = model_cfg.get("model_type", None)

    if saved_model_type in {"ElmanRNNMultiHead", "elman"}:
        model_type = "elman"
    elif saved_model_type in {"GRUMultiHeadWithCB", "gru"}:
        model_type = "gru"
    else:
        model_type = inferred_type

    activation_name = model_cfg.get(
        "activation",
        model_cfg.get(
            "candidate_afunc",
            cli_args.get("afunc", "leakyrelu" if model_type == "elman" else "tanh"),
        ),
    )
    afunc = activation_from_string(activation_name)

    input_size = int(model_cfg.get("input_size", dims["input_size"]))
    hidden_size = int(model_cfg.get("hidden_size", dims["hidden_size"]))
    num_classes = int(model_cfg.get("num_classes", dims["num_classes"]))
    num_readout_heads = int(model_cfg.get("num_readout_heads", dims["num_readout_heads"]))

    use_cb_bias = bool(
        model_cfg.get(
            "use_cb_bias",
            any(k.startswith("cb.") for k in state_dict.keys()),
        )
    )

    bias = bool(model_cfg.get("bias", True))
    tau = float(model_cfg.get("tau", 1.5))

    if use_cb_bias:
        if "cb.gc.weight" in state_dict:
            cb_gc_dim = int(state_dict["cb.gc.weight"].shape[0])
            cb_gc_in = int(state_dict["cb.gc.weight"].shape[1])

            if "cb.pc.weight" in state_dict:
                cb_pc_dim = int(state_dict["cb.pc.weight"].shape[0])
            else:
                cb_pc_dim = _safe_int(
                    model_cfg.get("cb_pc_dim", cli_args.get("pc_dim", None)),
                    64,
                )

            if cb_gc_in == hidden_size:
                cb_input_size = 0
                cb_no_hidden = False
            elif cb_gc_in > hidden_size:
                cb_input_size = cb_gc_in - hidden_size
                cb_no_hidden = False
            else:
                cb_input_size = cb_gc_in
                cb_no_hidden = True

        else:
            cb_gc_dim = _safe_int(
                model_cfg.get("cb_gc_dim", cli_args.get("gc_dim", None)),
                128,
            )
            cb_pc_dim = _safe_int(
                model_cfg.get("cb_pc_dim", cli_args.get("pc_dim", None)),
                64,
            )

            cb_no_hidden = bool(
                model_cfg.get("cb_no_hidden", cli_args.get("cb_no_hidden", False))
            )
            cb_sees_input = bool(cli_args.get("cb_sees_input", False))

            cb_input_size = _safe_int(
                model_cfg.get("cb_input_size", None),
                input_size if (cb_sees_input or cb_no_hidden) else 0,
            )

    else:
        cb_gc_dim = 128
        cb_pc_dim = 64
        cb_input_size = 0
        cb_no_hidden = False

    cb_max_ratio = float(
        model_cfg.get("cb_max_ratio")
        if model_cfg.get("cb_max_ratio") is not None
        else cli_args.get("cb_max_ratio", 1.0)
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

    elif model_type == "gru":
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

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if load_weights:
        cleaned_sd = clean_state_dict_for_current_model(state_dict)
        model.load_state_dict(cleaned_sd, strict=strict)

    model.eval()
    return model


def load_model_from_run(run_dir, N, device="cpu", strict=False):
    """
    Convenience helper: load config + checkpoint, rebuild model, and load weights.
    """
    cfg = load_run_config(run_dir)
    state_dict = load_state_dict(run_dir, N)

    model = build_model_from_config_and_state(
        cfg=cfg,
        state_dict=state_dict,
        device=device,
        load_weights=True,
        strict=strict,
    )

    model.eval()
    return model