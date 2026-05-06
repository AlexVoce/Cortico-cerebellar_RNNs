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
    subdir_override=None,   # NEW
):
    if affixes is None:
        affixes = []

    # ---- choose run folder ----
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

        # IMPORTANT: do NOT silently create a new run dir mid-run
        # If it already exists and this isn't init, just reuse it.
        if init:
            os.makedirs(rnn_subdir, exist_ok=False)  # fail fast if collision
        else:
            os.makedirs(rnn_subdir, exist_ok=True)   # reuse existing run dir

    # ---- checkpoint filename (inside the same run folder) ----
    if init:
        rnn_name = "rnn_init"
    else:
        rnn_name = f"rnn_N{N_min:d}_N{N_max:d}"

    # ---- write config once ----
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

    # ---- save checkpoint ----
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

    # Replicate the folder naming logic from save_model
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
    """
    Model-aware parameter counting that works for both Elman and GRU multi-head models.

    Returns a dict with stable top-level keys used downstream (`total`, `trainable`)
    plus optional breakdown keys when those submodules exist.
    """
    out = count_params(model)

    hparams = getattr(model, "_hparams", {}) or {}
    out["model_type"] = hparams.get("model_type", model.__class__.__name__)

    # Readout heads (shared across Elman/GRU in this repo)
    readout_ids = set()
    if hasattr(model, "heads") and model.heads is not None:
        out["readout"] = count_params(model.heads)
        readout_ids = _module_param_ids(model.heads)

    # Optional cerebellar module
    cb_ids = set()
    if hasattr(model, "cb") and model.cb is not None:
        out["cb"] = count_params(model.cb)
        cb_ids = _module_param_ids(model.cb)

    # Core recurrent params = all params excluding readout and CB params.
    excluded = readout_ids | cb_ids
    core_params = [p for p in model.parameters() if id(p) not in excluded]
    out["core"] = _count_from_params(core_params)

    # Convenience aliases for compatibility/readability.
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
    # argparse.Namespace
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

    # Optional CB block
    if hasattr(model, "cb") and model.cb is not None:
        if hasattr(model.cb, "_hparams"):
            cfg["cerebellum"] = model.cb._hparams

    # Include model-aware parameter counts (Elman/GRU compatible).
    cfg["params"] = count_model_params(model)

    return cfg

def log_and_save(row, subdir, stats, global_epoch_box,
                 save_every=5, force=False):
    """
    global_epoch_box: dict like {"v": 0}
    save_every: save to disk every N log calls
    """
    # increment epoch (persists)
    global_epoch_box["v"] += 1
    ep = global_epoch_box["v"]

    # append always
    stats["n_task"].append(row["N"])
    stats["phase"].append(row["phase"])
    stats["loss"].append(row["loss"])
    stats["accuracy"].append(row["acc"])
    stats["grad_rnn"].append(row["gRNN"])
    stats["grad_cb"].append(row["gCB"])
    stats["epoch"].append(ep)

    final_path = os.path.join(subdir, "stats.npy")
    tmp_path = os.path.join(subdir, "stats_temp.npy")
    # only save sometimes
    n_rows = len(stats["epoch"])
    if force or (save_every is not None and n_rows % save_every == 0):
        np.save(tmp_path, stats)
        os.replace(tmp_path, final_path)
        print("SAVED", row["phase"], row["N"], "epoch", ep, "rows", n_rows, flush=True)

def make_unique_dir(path: str) -> str:
    """
    If `path` exists, append _v2, _v3, ... until unique.
    Returns the final directory path (created).
    """
    base = path
    k = 1
    while os.path.exists(path):
        k += 1
        path = f"{base}_v{k}"
    os.makedirs(path, exist_ok=True)
    return path
