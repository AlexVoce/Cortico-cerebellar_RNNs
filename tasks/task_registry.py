import numpy as np
import torch
import torch.nn as nn

import os
import sys

try:
    import src.tasks as tasks
except ImportError:
    # Fallback: compute path relative to this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.dirname(current_dir)  # Go up from tasks/ → src/
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    import tasks


# ---------- Metrics ----------
def metric_classification(outputs, labels, ignore_index=-100):
    """
    outputs: list of head outputs, each [B,C] or [T,B,C] for timewise
    labels:  list of head labels, each [B] or [T,B] for timewise
    """
    per_head = []
    for h in range(len(labels)):
        out = outputs[h]
        lbl = labels[h]

        # Flatten timewise
        if out.dim() == 3:
            out = out.reshape(-1, out.size(-1))
            lbl = lbl.reshape(-1)

        # Mask ignored labels
        if ignore_index is not None:
            mask = (lbl != ignore_index)
            if mask.any():
                pred = torch.argmax(out[mask], dim=-1)
                acc = (pred == lbl[mask]).float().mean().item() * 100.0
            else:
                acc = float("nan")  # no scored positions
        else:
            pred = torch.argmax(out, dim=-1)
            acc = (pred == lbl).float().mean().item() * 100.0

        per_head.append(acc)

    # score = mean over heads (ignoring NaNs if a head has no scored positions)
    score = float(np.nanmean(per_head)) if len(per_head) else float("nan")

    return {
        "score": score,
        "per_head": per_head,
        "name": "accuracy"
    }

IGNORE = -100
def copy_criterion_ctor():
    return nn.CrossEntropyLoss(ignore_index=IGNORE)

def metric_copy(outputs, labels, blank_class=None, ignore_index=IGNORE):
    """
    outputs: list of [T,B,C] logits (one per head)
    labels:  list of [T,B] longs (one per head)
    """
    outs = outputs if isinstance(outputs, list) else [outputs]
    labs = labels if isinstance(labels, list) else [labels]

    per_head = []
    for out, y in zip(outs, labs):
        # out: [T,B,C], y: [T,B]
        pred = out.argmax(dim=-1)  # [T,B]
        mask = (y != ignore_index)
        if mask.sum() == 0:
            per_head.append(0.0)
            continue
        acc = (pred[mask] == y[mask]).float().mean().item() * 100.0
        per_head.append(acc)

    return {"name": "acc", "score": float(np.mean(per_head)), "per_head": per_head}

def metric_dco(outputs, labels):
    """
    outputs: list of head outputs, each [T,B,2]
    labels:  list of head targets, each [T,B,2]
    Returns score + endpoint error.
    """
    per_head_endpoint_err = []
    per_head_score = []

    for h in range(len(labels)):
        pred = outputs[h]              # [T,B,2]
        tgt = labels[h].float()        # [T,B,2]

        pred_end = pred[-1]            # [B,2]
        tgt_end = tgt[-1]              # [B,2]
        endpoint_err = torch.norm(pred_end - tgt_end, dim=1).mean().item()
        per_head_endpoint_err.append(endpoint_err)

        # "higher is better" convenience score for generic curriculum logging
        score = max(0.0, 100.0 * (1.0 - endpoint_err))
        per_head_score.append(score)

    return {
        "score": float(np.mean(per_head_score)),
        "per_head": per_head_score,
        "endpoint_error": float(np.mean(per_head_endpoint_err)),
        "per_head_endpoint_error": per_head_endpoint_err,
        "name": "endpoint_score"
    }


def metric_regression_mse(outputs, labels):
    """
    outputs: list of head outputs, each [B,D] or [T,B,D]
    labels:  list of head targets, each [B,D] or [T,B,D]
    Returns score + mse (lower is better for mse, higher is better for score).
    """
    per_head_mse = []
    per_head_score = []

    for h in range(len(labels)):
        out = outputs[h]
        tgt = labels[h].float()

        # Flatten batch/time dims but preserve feature dim for MSE.
        if out.dim() == 3:
            out = out.reshape(-1, out.size(-1))
            tgt = tgt.reshape(-1, tgt.size(-1))

        mse = torch.mean((out - tgt) ** 2).item()
        per_head_mse.append(mse)

        # Smooth bounded score in [0, 100], useful for generic curriculum logging.
        score = 100.0 / (1.0 + mse)
        per_head_score.append(score)

    return {
        "score": float(np.mean(per_head_score)),
        "per_head": per_head_score,
        "mse": float(np.mean(per_head_mse)),
        "per_head_mse": per_head_mse,
        "name": "mse_score"
    }


# ---------- Loss helper ----------

def compute_loss(outputs, labels, target_type, criterion):
    """
    outputs: list of head outputs
    labels:  list of head labels
    Handles both per-timestep (timewise) and final-only outputs.
    """
    loss = 0.0
    for h in range(len(labels)):
        out = outputs[h]
        lbl = labels[h]

        if target_type == "class":
            # Handle timewise classification: [T,B,C] + [T,B] -> [T*B,C] + [T*B]
            if out.dim() == 3:
                out = out.reshape(-1, out.size(-1))
                lbl = lbl.reshape(-1)
            loss = loss + criterion(out, lbl)
        elif target_type == "class_timewise":
            # Handle timewise classification: [T,B,C] + [T,B] -> [T*B,C] + [T*B]
            if out.dim() == 3:
                out = out.reshape(-1, out.size(-1))
                lbl = lbl.reshape(-1)
            loss = loss + criterion(out, lbl)
        elif target_type == "regression":
            # Handle timewise regression: [T,B,D] + [T,B,D] -> [T*B,D] + [T*B,D]
            if out.dim() == 3:
                out = out.reshape(-1, out.size(-1))
                lbl = lbl.reshape(-1, lbl.size(-1))
            loss = loss + criterion(out, lbl.float())
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
    return loss


# ---------- Curriculum advance rules ----------

def advance_by_accuracy(metric_dict, threshold=98.0,loss_threshold=None):
    return metric_dict["score"] >= threshold


def advance_by_endpoint_error(metric_dict, threshold=0.15):
    # Smaller is better
    return metric_dict.get("endpoint_error", 1e9) <= threshold


def advance_by_mse(metric_dict, threshold=0.02):
    # Smaller is better
    return metric_dict.get("mse", 1e9) <= threshold


# ---------- Task specs ----------

TASK_SPECS = {
    "parity": {
        "batch_fn": tasks.make_batch_mtstyle_parity,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "dms": {
        "batch_fn": tasks.make_batch_mtstyle_dms,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
        "oddball": {
        "batch_fn": tasks.make_batch_mtstyle_oddball,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "dms_distractor": {
        "batch_fn": tasks.make_batch_multihead_dms_distractor,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "majority": {
        "batch_fn": tasks.make_batch_majority_matches_probe,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "evidence": {
        "batch_fn": tasks.make_batch_latent_evidence_accumulation,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "change_point": {
        "batch_fn": tasks.make_batch_change_point_detection,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 4,
    },
    "violation": {
        "batch_fn": tasks.make_batch_context_rule_violation,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "hazard": {
        "batch_fn": tasks.make_batch_hazard_regime_violation,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "dco": {
        "batch_fn": tasks.make_batch_delayed_center_out,
        "input_size": 4,      # target_x, target_y, go, hold
        "output_size": 2,     # x,y trajectory
        "target_type": "regression",
        "timewise_output": True,
        "criterion_ctor": nn.MSELoss,
        "loss_name": "mse",
        "metric_fn": metric_dco,
        "advance_fn": lambda m: advance_by_endpoint_error(m, threshold=0.20),
        "start_n": 2,         # N means target set size level in your curriculum
    },
    "rule_switch": {
        "batch_fn": lambda Ns, bs: tasks.make_batch_rule_switch(
            Ns, bs, D=2, K=7, M=10,
        ),
        "input_size": 4,       # 2 * D
        "output_size": 4,     # 2^D
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=85.0),
        "start_n": 2,
    },
    "copy_task": {
        "batch_fn": tasks.make_batch_copy_task,
        "input_size": 1,         # 6 symbols + delimiter
        "output_size": 10,        # 6 symbols + blank token (still fine)
        "target_type": "class_timewise",
        "timewise_output": True,
        "criterion_ctor": copy_criterion_ctor,
        "loss_name": "cross_entropy",
        "metric_fn": metric_copy,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 3,
    },
    "adding_task": {
        "batch_fn": tasks.make_batch_adding_task,
        "input_size": 2,       # [value, marker]
        "output_size": 1,      # scalar target
        "target_type": "regression",
        "timewise_output": False,
        "criterion_ctor": nn.MSELoss,
        "loss_name": "mse",
        "metric_fn": metric_regression_mse,
        "advance_fn": lambda m: advance_by_mse(m, threshold=0.005),
        "start_n": 3,
    },
    "associative_recall": {
        "batch_fn": tasks.make_batch_associative_recall,
        "input_size": 7,            # num_symbols + query_flag; e.g. 6+1
        "output_size": 6,           # classes = values in [0, num_symbols-1]
        "target_type": "class_timewise",
        "timewise_output": True,
        "criterion_ctor": lambda: nn.CrossEntropyLoss(ignore_index=-100),
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,               # assumes it can handle IGNORE=-100
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "context_integration": {
        "batch_fn": tasks.make_batch_context_integration,
        "input_size": 3,        # context scalar + stream A + stream B
        "output_size": 2,       # binary classification
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
    "rapid_reversal": {
        "batch_fn": tasks.make_batch_rapid_reversal,
        "input_size": 2,
        "output_size": 2,
        "target_type": "class_timewise",
        "timewise_output": True,
        "criterion_ctor": lambda: nn.CrossEntropyLoss(ignore_index=-100),
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 4,
    },
    "few_shot": {
        "batch_fn": lambda Ns, bs: tasks.make_batch_few_shot_classification(
            Ns, bs, K=3, input_dim=20, max_N=20
        ),
        "input_size": 41,   # input_dim(20) + max_N(20) + is_query(1)
        "output_size": 20,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=98.0),
        "start_n": 2,
    },
}