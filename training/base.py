# training/alternating/base.py
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from ..alex_utils import step_optimizer
except ImportError:
    from alex_utils import step_optimizer

try:
    from tasks.task_registry import compute_loss
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(current_dir, "..", "..", "rnn_timescale_public-main", "src"))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
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
    cb_l2=0.0,
    rnn_eat=False, 
    rnn_eat_lambda=0.1,
    rnn_eat_loss_type='hidden',
    cb_sees_input=True,
    learning_alg="BPTT",
):
    """
    Runs `training_steps` minibatches.
    Returns (mean_loss, mean_gRNN, mean_gCB).

    - BPTT: gRNN/gCB are true gradient norms from get_grad_norms.
    - RFLO: gRNN/gCB are *update magnitudes* (proxy for learning signal):
        gRNN ~ ||ΔW_inp|| + ||ΔW_hh||
        gCB  ~ ||ΔW_cb|| (sum over cb layers)
    """
    model.train()
    losses_step = []
    grad_rnn_step = []
    grad_cb_step = []
    # below are for debugging loss spikes with RNN eat
    grad_cb_preclip_step = []
    grad_rnn_preclip_step = []
    task_loss_step = []
    eat_loss_step = []
    # ht_m, ht_M = [], []
    # cb_m, cb_M = [], []
    # pre_m, pre_M = [], []
    # post_m, post_M = [], []
    # nf_ht, nf_cb, nf_pre, nf_post = [], [], [], []
    # m_a_l = []  # max abs logit
    
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
        # dbg = getattr(model, "_dbg", None) or {}
        # ht_m.append(float(dbg.get("ht_norm_mean", np.nan)))
        # ht_M.append(float(dbg.get("ht_norm_max", np.nan)))
        # cb_m.append(float(dbg.get("cb_norm_mean", np.nan)))
        # cb_M.append(float(dbg.get("cb_norm_max", np.nan)))
        # pre_m.append(float(dbg.get("pre_norm_mean", np.nan)))
        # pre_M.append(float(dbg.get("pre_norm_max", np.nan)))
        # post_m.append(float(dbg.get("post_norm_mean", np.nan)))
        # post_M.append(float(dbg.get("post_norm_max", np.nan)))
        # nf_ht.append(int(dbg.get("nonfinite_ht", 0)))
        # nf_cb.append(int(dbg.get("nonfinite_cb", 0)))
        # nf_pre.append(int(dbg.get("nonfinite_pre", 0)))
        # nf_post.append(int(dbg.get("nonfinite_post", 0)))
        # m_a_l.append(float(dbg.get("max_abs_logit", np.nan)))

        selected_outputs = [out_heads[head_idx(n)] for n in Ns]
        loss = compute_loss(selected_outputs, labels, spec["target_type"], criterion)
        task_loss_step.append(float(loss.item()))

        if cb_l2 > 0.0 and hasattr(model, "cb"):
            cb_params = [p for n, p in model.named_parameters() if "cb" in n]
            cb_l2_loss = sum(p.norm(2) ** 2 for p in cb_params)
            loss = loss + cb_l2 * cb_l2_loss
        elif rnn_eat:
            effective_eat_lambda = rnn_eat_lambda
            if rnn_eat_loss_type == 'hidden' and getattr(model, "_last_eat_loss", None) is not None:
                eat_loss_val = float(model._last_eat_loss.item())
                eat_loss_step.append(eat_loss_val)
                if eat_loss_val <= 5.0:
                    # loss = loss + effective_eat_lambda * model._last_eat_loss
                    loss = ((1 - effective_eat_lambda) * loss) + (effective_eat_lambda * model._last_eat_loss)
            elif rnn_eat_loss_type == 'task' and getattr(model, "_last_h_student", None) is not None:
                h_stu = model._last_h_student
                student_out = [model.heads[head_idx(n)](h_stu) for n in Ns]
                student_task_loss = compute_loss(student_out, labels, spec["target_type"], criterion)
                eat_loss_val = float(student_task_loss.item())
                eat_loss_step.append(eat_loss_val)
                if eat_loss_val <= 5.0:  # fixed condition
                    loss = ((1 - effective_eat_lambda) * loss) + (effective_eat_lambda * model._last_eat_loss)

        loss.backward()

        if rnn_eat and rnn_eat_loss_type == 'task':
            for head in model.heads:
                for p in head.parameters():
                    p.grad = None

        gRNN_preclip, gCB_preclip, _ = get_grad_norms(model)

        if not shared_optimiser:
            rnn_params = [p for n, p in model.named_parameters() if "cb" not in n]
            nn.utils.clip_grad_norm_(rnn_params, max_norm=7.5)
            cb_params = [p for n, p in model.named_parameters() if "cb" in n]
            if cb_params:
                nn.utils.clip_grad_norm_(cb_params, max_norm=5.0)
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
        "eat_loss": float(np.mean(eat_loss_step)),
        # "dbg": {
        #     "ht_norm_mean": float(np.mean(ht_m)),
        #     "ht_norm_max": float(np.nanmean(ht_M)),
        #     "cb_norm_mean": float(np.nanmean(cb_m)),
        #     "cb_norm_max": float(np.nanmax(cb_M)),
        #     "pre_norm_mean": float(np.nanmean(pre_m)),
        #     "pre_norm_max": float(np.nanmax(pre_M)),
        #     "post_norm_mean": float(np.nanmean(post_m)),
        #     "post_norm_max": float(np.nanmax(post_M)),
        #     "nonfinite_ht": int(np.sum(nf_ht)),
        #     "nonfinite_cb": int(np.sum(nf_cb)),
        #     "nonfinite_pre": int(np.sum(nf_pre)),
        #     "nonfinite_post": int(np.sum(nf_post)),
        #     "max_abs_logit": float(np.nanmax(m_a_l)),
        # }
    }