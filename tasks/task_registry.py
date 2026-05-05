import numpy as np
import torch
import torch.nn as nn

import tasks.tasks_using as tasks


def metric_classification(outputs, labels, ignore_index=None):
    """
    Mean classification accuracy across readout heads.
    Supports outputs of shape [B, C] or [T, B, C].
    """
    if not isinstance(outputs, list):
        outputs = [outputs]
    if not isinstance(labels, list):
        labels = [labels]

    per_head = []

    for out, lbl in zip(outputs, labels):
        if out.dim() == 3:
            out = out.reshape(-1, out.size(-1))
            lbl = lbl.reshape(-1)

        if ignore_index is not None:
            mask = lbl != ignore_index
            if not mask.any():
                per_head.append(float("nan"))
                continue
            out = out[mask]
            lbl = lbl[mask]

        pred = torch.argmax(out, dim=-1)
        acc = (pred == lbl).float().mean().item() * 100.0
        per_head.append(acc)

    return {
        "score": float(np.nanmean(per_head)) if per_head else float("nan"),
        "per_head": per_head,
        "name": "accuracy",
    }


def compute_loss(outputs, labels, target_type, criterion):
    """
    Cross-entropy loss across readout heads.
    """
    if target_type != "class":
        raise ValueError(f"Expected target_type='class', got {target_type!r}")

    if not isinstance(outputs, list):
        outputs = [outputs]
    if not isinstance(labels, list):
        labels = [labels]

    loss = 0.0

    for out, lbl in zip(outputs, labels):
        if out.dim() == 3:
            out = out.reshape(-1, out.size(-1))
            lbl = lbl.reshape(-1)

        loss = loss + criterion(out, lbl)

    return loss


def advance_by_accuracy(metric_dict, threshold=98.0):
    return metric_dict["score"] >= threshold


def make_binary_classification_spec(batch_fn, start_n=2, threshold=98.0):
    return {
        "batch_fn": batch_fn,
        "input_size": 1,
        "output_size": 2,
        "target_type": "class",
        "timewise_output": False,
        "criterion_ctor": nn.CrossEntropyLoss,
        "loss_name": "cross_entropy",
        "metric_fn": metric_classification,
        "advance_fn": lambda m: advance_by_accuracy(m, threshold=threshold),
        "start_n": start_n,
    }


TASK_SPECS = {
    "dms": make_binary_classification_spec(tasks.make_batch_mtstyle_dms),
    "parity": make_binary_classification_spec(tasks.make_batch_mtstyle_parity),
}