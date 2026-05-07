# training/base.py

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from training.train_utils import step_optimizer
from tasks.task_registry import compute_loss


def head_idx_factory(Ns_init, num_heads):
    def head_idx(n):
        idx = n - Ns_init[0]

        if idx < 0 or idx >= num_heads:
            raise IndexError(
                f"N={n} -> head {idx}, but only {num_heads} heads exist"
            )

        return idx

    return head_idx


def compute_active_set(active_Ns, next_N, readout_head_dyn, n_heads, n_forget):
    """
    Compute which N-levels are actively trained at the current curriculum stage.
    """
    if readout_head_dyn == "single":
        return [next_N]

    temp = active_Ns + [next_N]

    if readout_head_dyn == "cumulative":
        return temp

    if readout_head_dyn == "sliding":
        if len(temp) > n_heads:
            return temp[n_forget:]
        return temp

    raise ValueError(
        f"Unknown readout_head_dyn='{readout_head_dyn}'. "
        "Expected one of: 'single', 'cumulative', 'sliding'."
    )


@torch.no_grad()
def evaluate(
    model,
    task_fn,
    active_Ns,
    batch_size,
    test_steps,
    device,
    criterion,
    head_idx,
    spec,
):
    """
    Evaluate model performance over test_steps batches.
    """
    model.eval()
    metrics = []

    for _ in range(test_steps):
        seq, labels = task_fn(active_Ns, batch_size)
        labels = labels if isinstance(labels, list) else [labels]

        seq = seq.to(device)
        labels = [label.to(device) for label in labels]

        _, out_heads = model(
            seq,
            return_timewise=spec["timewise_output"],
        )
        out_heads = out_heads if isinstance(out_heads, list) else [out_heads]

        selected_outputs = [out_heads[head_idx(n)] for n in active_Ns]
        metric = spec["metric_fn"](selected_outputs, labels)
        metrics.append(metric)

    excluded = {
        "per_head",
        "name",
        "per_head_endpoint_error",
        "per_head_mse",
    }

    aggregated = {}

    for key in metrics[0].keys():
        if key in excluded:
            continue

        vals = [m[key] for m in metrics if key in m]

        if len(vals) > 0:
            aggregated[key] = float(np.mean(vals))

    return aggregated


def _zero_optimizer(optimizer, shared_optimiser=True):
    if shared_optimiser:
        optimizer.zero_grad(set_to_none=True)
    else:
        for opt in optimizer.values():
            if opt is not None:
                opt.zero_grad(set_to_none=True)


def _split_model_params(model):
    """
    Split model parameters into base RNN/readout parameters and CB parameters.

    Uses model.cb directly rather than relying on parameter-name strings.
    """
    cb_params = (
        list(model.cb.parameters())
        if getattr(model, "cb", None) is not None
        else []
    )

    cb_param_ids = {id(p) for p in cb_params}
    rnn_params = [
        p for p in model.parameters()
        if id(p) not in cb_param_ids
    ]

    return rnn_params, cb_params


def _clip_gradients(model, shared_optimiser=True, max_norm=7.5):
    if shared_optimiser:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        return

    rnn_params, cb_params = _split_model_params(model)

    if len(rnn_params) > 0:
        nn.utils.clip_grad_norm_(rnn_params, max_norm=max_norm)

    if len(cb_params) > 0:
        nn.utils.clip_grad_norm_(cb_params, max_norm=max_norm)


def train_steps(
    model,
    task_fn,
    active_Ns,
    batch_size,
    phase,
    device,
    criterion,
    head_idx,
    optimizer,
    get_grad_norms,
    training_steps,
    shared_optimiser=True,
    spec=None,
):
    """
    Run training_steps minibatches for one reservoir curriculum stage.
    """
    if spec is None:
        raise ValueError("train_steps requires spec=...")

    model.train()

    losses_step = []
    grad_rnn_step = []
    grad_cb_step = []
    grad_rnn_preclip_step = []
    grad_cb_preclip_step = []
    task_loss_step = []

    pbar = tqdm(
        range(training_steps),
        desc=f"N={active_Ns}",
        leave=False,
    )

    for _ in pbar:
        seq, labels = task_fn(active_Ns, batch_size)
        labels = labels if isinstance(labels, list) else [labels]

        seq = seq.to(device)
        labels = [label.to(device) for label in labels]

        _zero_optimizer(
            optimizer,
            shared_optimiser=shared_optimiser,
        )

        _, out_heads = model(
            seq,
            return_timewise=spec["timewise_output"],
        )
        out_heads = out_heads if isinstance(out_heads, list) else [out_heads]

        selected_outputs = [out_heads[head_idx(n)] for n in active_Ns]

        loss = compute_loss(
            selected_outputs,
            labels,
            spec["target_type"],
            criterion,
        )

        task_loss_step.append(float(loss.item()))
        loss.backward()

        gRNN_preclip, gCB_preclip, _ = get_grad_norms(model)

        _clip_gradients(
            model,
            shared_optimiser=shared_optimiser,
            max_norm=7.5,
        )

        gRNN, gCB, _ = get_grad_norms(model)

        step_optimizer(
            optimizer,
            phase,
            shared_optimiser=shared_optimiser,
        )

        losses_step.append(float(loss.item()))
        grad_rnn_step.append(float(gRNN))
        grad_cb_step.append(float(gCB))
        grad_rnn_preclip_step.append(float(gRNN_preclip))
        grad_cb_preclip_step.append(float(gCB_preclip))

        pbar.set_postfix(
            {
                "Loss": f"{np.mean(losses_step):.4f}",
                "gRNN": f"{np.mean(grad_rnn_step):.4f}",
                "gCB": f"{np.mean(grad_cb_step):.4f}",
            }
        )

    return {
        "loss": float(np.mean(losses_step)),
        "gRNN": float(np.mean(grad_rnn_step)),
        "gCB": float(np.mean(grad_cb_step)),
        "gRNN_preclip": float(np.mean(grad_rnn_preclip_step)),
        "gCB_preclip": float(np.mean(grad_cb_preclip_step)),
        "task_loss": float(np.mean(task_loss_step)),
    }