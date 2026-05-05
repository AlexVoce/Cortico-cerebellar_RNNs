# training/base.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from training.train_utils import step_optimizer
from tasks.task_registry import compute_loss

def get_eat_lambda(base_lambda, current_N, decay_start=20, decay_end=35):
    if current_N <= decay_start:
        return base_lambda
    if current_N >= decay_end:
        return 0.0

    # Linear decay between decay_start and decay_end.
    progress = (current_N - decay_start) / (decay_end - decay_start)
    return base_lambda * (1 - progress)

def head_idx_factory(Ns_init, num_heads):
    def head_idx(n):
        idx = n - Ns_init[0]
        if idx < 0 or idx >= num_heads:
            raise IndexError(f"N={n} -> head {idx}, but only {num_heads} heads exist")
        return idx
    return head_idx

def compute_active_set(active_Ns, next_N, readout_head_dyn, n_heads, n_forget):
    if readout_head_dyn == "single":
        return [next_N]  # only the new head is active
    temp = active_Ns + [next_N]
    if readout_head_dyn == "cumulative":
        return temp
    # sliding
    if len(temp) > n_heads:
        return temp[n_forget:]
    return temp

@torch.no_grad()
def evaluate(model, task_fn, Ns, batch_size, test_steps, device, criterion, head_idx, spec):
    model.eval()
    metrics = []
    for _ in range(test_steps):
        seq, labels = task_fn(Ns, batch_size)
        labels = labels if isinstance(labels, list) else [labels]
        seq = seq.to(device)
        labels = [l.to(device) for l in labels]
        _, out_heads = model(seq, return_timewise=spec["timewise_output"])
        out_heads = out_heads if isinstance(out_heads, list) else [out_heads]
        selected_outputs = [out_heads[head_idx(n)] for n in Ns]
        metric = spec["metric_fn"](selected_outputs, labels)
        metrics.append(metric)

    # Aggregate all numeric keys across test steps
    excluded = {"per_head", "name", "per_head_endpoint_error", "per_head_mse"}
    aggregated = {}
    for k in metrics[0].keys():
        if k in excluded:
            continue
        vals = [m[k] for m in metrics if k in m]
        if len(vals) > 0:
            aggregated[k] = float(np.mean(vals))
    return aggregated

def train_steps(
    model, task_fn,
    Ns, batch_size, phase,
    device, criterion, head_idx, optimizer,
    get_grad_norms, training_steps,
    shared_optimiser=True,
    spec=None,
    cb_sees_input=True,

):
    """
    Runs `training_steps` minibatches.
    Returns (mean_loss, mean_gRNN, mean_gCB).

    - BPTT: gRNN/gCB are true gradient norms from get_grad_norms.
    """
    model.train()
    losses_step = []
    grad_rnn_step = []
    grad_cb_step = []
    # below are for debugging loss spikes with RNN eat
    grad_cb_preclip_step = []
    grad_rnn_preclip_step = []
    task_loss_step = []

    pbar = tqdm(range(training_steps), desc=f"N={Ns}", leave=False)

    for _ in pbar:
        seq, labels = task_fn(Ns, batch_size)
        labels = labels if isinstance(labels, list) else [labels]
        seq = seq.to(device)
        labels = [l.to(device) for l in labels]

        if shared_optimiser:
            optimizer.zero_grad(set_to_none=True)
        else:
            for opt in optimizer.values():
                if opt:
                    opt.zero_grad(set_to_none=True)

        _, out_heads = model(seq, return_timewise=spec["timewise_output"])
        out_heads = out_heads if isinstance(out_heads, list) else [out_heads]

        selected_outputs = [out_heads[head_idx(n)] for n in Ns]
        loss = compute_loss(selected_outputs, labels, spec["target_type"], criterion)
        task_loss_step.append(float(loss.item()))

        loss.backward()

        gRNN_preclip, gCB_preclip, _ = get_grad_norms(model)

        if not shared_optimiser:
            rnn_params = [p for n, p in model.named_parameters() if "cb" not in n]
            nn.utils.clip_grad_norm_(rnn_params, max_norm=7.5)
            cb_params = [p for n, p in model.named_parameters() if "cb" in n]
            if cb_params:
                nn.utils.clip_grad_norm_(cb_params, max_norm=7.5)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=7.5)

        gRNN, gCB, _ = get_grad_norms(model)
        step_optimizer(optimizer, phase, shared_optimiser=shared_optimiser)

        losses_step.append(float(loss.item()))
        grad_rnn_step.append(float(gRNN))
        grad_cb_step.append(float(gCB))
        grad_rnn_preclip_step.append(float(gRNN_preclip))
        grad_cb_preclip_step.append(float(gCB_preclip))

        pbar.set_postfix({
            "Loss": f"{np.mean(losses_step):.4f}",
            "gRNN": f"{np.mean(grad_rnn_step):.4f}",
            "gCB":  f"{np.mean(grad_cb_step):.4f}",
        })

    return {
        "loss": float(np.mean(losses_step)),
        "gRNN": float(np.mean(grad_rnn_step)),
        "gCB": float(np.mean(grad_cb_step)),
        "gRNN_preclip": float(np.mean(grad_rnn_preclip_step)),
        "gCB_preclip": float(np.mean(grad_cb_preclip_step)),
        "task_loss": float(np.mean(task_loss_step)),
    }