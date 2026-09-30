import torch
import os
import json, hashlib, platform, datetime,sys, time
import numpy as np


def _count_from_params(params):
    params = list(params)
    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def _module_param_ids(module):
    return {id(p) for p in module.parameters()}

def save_model(
    model,
    curriculum_type: str,
    n_heads: int,
    n_forget: int,
    task: str,
    network_number: int,
    N_max: int,
    N_min: int = 2,
    base_path="../trained_models",
    affixes=None,
    init=False,
    args=None,
    Ns_init=None,
    device=None,
    subdir_override=None,
):
    if affixes is None:
        affixes = []

    # Run folder
    if subdir_override is not None:
        rnn_subdir = subdir_override
        os.makedirs(rnn_subdir, exist_ok=True)
    else:
        affix_str = "_"
        if len(affixes) > 0:
            affix_str += "_".join(affixes) + "_"

        if curriculum_type == "sliding":
            rnn_subdir = os.path.join(
                base_path,
                f"{curriculum_type}_{n_heads}_{n_forget}_{task}{affix_str}network_{network_number}",
            )
        else:
            rnn_subdir = os.path.join(
                base_path,
                f"{curriculum_type}_{task}{affix_str}network_{network_number}",
            )

        # reuse an existing run dir after init; never create a new one mid-run
        if init:
            os.makedirs(rnn_subdir, exist_ok=False)
        else:
            os.makedirs(rnn_subdir, exist_ok=True)

    # Checkpoint filename
    if init:
        rnn_name = "rnn_init"
    else:
        rnn_name = f"rnn_N{N_min:d}_N{N_max:d}"

    # Config (written once)
    config_path = os.path.join(rnn_subdir, "config.json")
    if not os.path.exists(config_path):
        save_run_config(
            subdir=rnn_subdir,
            args=args,
            affixes=affixes,
            curriculum_type=curriculum_type,
            task=task,
            Ns_init=Ns_init,
            device=device,
            model=model,
            filename="config.json",
        )

    # Checkpoint
    filename = os.path.join(rnn_subdir, rnn_name)
    torch.save({"state_dict": model.state_dict()}, filename)

    return rnn_subdir

def find_next_free_network_number(
        base_path,
        curriculum_type,
        task,
        affixes,
        n_heads,
        n_forget,
        start_number=1
):
    """Finds the next available network number to avoid overwriting."""
    
    affix_str = '_'
    if len(affixes) > 0:
        affix_str += '_'.join(affixes) + '_'

    # same folder naming as save_model
    if curriculum_type == 'sliding':
        folder_pattern = f'{curriculum_type}_{n_heads}_{n_forget}_{task}{affix_str}network_{{}}'
    else:
        folder_pattern = f'{curriculum_type}_{task}{affix_str}network_{{}}'

    current_number = start_number
    while True:
        folder_name = folder_pattern.format(current_number)
        print('Base path:', base_path)
        print("Checking for folder:", folder_name)
        full_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(full_path):
            return current_number
        
        current_number += 1

def count_params(module):
    return _count_from_params(module.parameters())


def count_model_params(model):
    """Parameter counts ('total', 'trainable', plus per-module breakdown) for Elman/GRU models."""
    out = count_params(model)

    hparams = getattr(model, "_hparams", {}) or {}
    out["model_type"] = hparams.get("model_type", model.__class__.__name__)

    readout_ids = set()
    if hasattr(model, "heads") and model.heads is not None:
        out["readout"] = count_params(model.heads)
        readout_ids = _module_param_ids(model.heads)

    cb_ids = set()
    if hasattr(model, "cb") and model.cb is not None:
        out["cb"] = count_params(model.cb)
        cb_ids = _module_param_ids(model.cb)

    # core recurrent params = all except readout and CB
    excluded = readout_ids | cb_ids
    core_params = [p for p in model.parameters() if id(p) not in excluded]
    out["core"] = _count_from_params(core_params)

    out["rnn"] = dict(out["core"])
    out["non_cb_trainable"] = int(out["trainable"] - out.get("cb", {"trainable": 0})["trainable"])

    return out

def safe_serialize(obj):
    """Convert argparse Namespace / non-JSON objects into JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: safe_serialize(v) for k, v in vars(obj).items()}
    return str(obj)

def save_run_config(
    subdir,
    args=None,
    affixes=None,
    curriculum_type=None,
    task=None,
    Ns_init=None,
    device=None,
    model=None,
    extra=None,
    filename="config.json",
):
    os.makedirs(subdir, exist_ok=True)

    cfg = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "run": {
            "curriculum_type": curriculum_type,
            "task": task,
            "Ns_init": list(Ns_init) if Ns_init is not None else None,
            "affixes": list(affixes) if affixes is not None else None,
        },
        "cli_args": safe_serialize(args),
        "model_config": extract_model_config(model) if model is not None else None,
        "params": count_model_params(model) if model is not None else None,
    }
    cfg["argv"] = sys.argv

    if model is not None:
        cfg["params"] = count_model_params(model)

    if extra is not None:
        cfg["extra"] = safe_serialize(extra)

    path = os.path.join(subdir, filename)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    return path

def extract_model_config(model):
    cfg = {}

    if hasattr(model, "_hparams"):
        cfg["model"] = model._hparams

    if hasattr(model, "cb") and model.cb is not None:
        if hasattr(model.cb, "_hparams"):
            cfg["cerebellum"] = model.cb._hparams

    cfg["params"] = count_model_params(model)

    return cfg

def log_and_save(row, subdir, stats, global_epoch_box,
                 save_every=5, force=False):
    """Log a row and save every save_every calls. global_epoch_box is a dict like {'v': 0}."""
    global_epoch_box["v"] += 1
    ep = global_epoch_box["v"]

    stats["n_task"].append(row["N"])
    stats["phase"].append(row["phase"])
    stats["loss"].append(row["loss"])
    stats["accuracy"].append(row["acc"])
    stats["grad_rnn"].append(row["gRNN"])
    stats["grad_cb"].append(row["gCB"])
    stats["epoch"].append(ep)

    final_path = os.path.join(subdir, "stats.npy")
    tmp_path = os.path.join(subdir, "stats_temp.npy")
    n_rows = len(stats["epoch"])
    if force or (save_every is not None and n_rows % save_every == 0):
        np.save(tmp_path, stats)
        os.replace(tmp_path, final_path)
        print("SAVED", row["phase"], row["N"], "epoch", ep, "rows", n_rows, flush=True)

def make_unique_dir(path: str) -> str:
    """Append _v2, _v3, ... to `path` until unique, then create it."""
    base = path
    k = 1
    while os.path.exists(path):
        k += 1
        path = f"{base}_v{k}"
    os.makedirs(path, exist_ok=True)
    return path

def match_aux_hidden_size(target_params, input_size, main_hidden_size,
                           search_range=range(2, 400), tau=1.5, bias=True):
    best_H2, best_diff, best_count = None, None, None
    from model.cb_rnn import RecurrentAuxModule
    for H2 in search_range:
        aux = RecurrentAuxModule(
            input_size=input_size,
            aux_hidden_size=H2,
            main_hidden_size=main_hidden_size,
            tau=tau,
            bias=bias,
        )
        n = count_params(aux)["trainable"]
        diff = abs(n - target_params)
        if best_diff is None or diff < best_diff:
            best_H2, best_diff, best_count = H2, diff, n
    return best_H2, best_count, best_diff

