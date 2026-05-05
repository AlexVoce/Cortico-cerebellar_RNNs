# alex_training/alternating/variants.py
from training.base import compute_active_set, train_steps, evaluate
import os
import sys

from tasks.task_registry import advance_by_accuracy

def variant_cb_only(*, model, task_fn, active_Ns, current_N, Ns_init,
                        batch_size,training_steps_n, test_steps, device,
                        criterion, head_idx, optimizer, get_grad_norms, global_epoch, 
                        set_active_module, readout_head_dyn, n_heads, n_forget,
                        THRESHOLD_FINAL, MAX_STAGE_EPOCHS, MAX_GLOBAL_EPOCHS, shared_optimiser,
                        learning_alg,spec, cb_l2=0.0,rnn_eat=False,rnn_eat_lambda=0.1,rnn_eat_loss_type='hidden', cb_sees_input=True, **kwargs):
    """
    Returns: (success, new_active_Ns, new_current_N, logs, updated_global_epoch)
    """
    logs = []
    next_N = int(kwargs.get("next_N_override", current_N + 1))
    active_set = sorted(compute_active_set(active_Ns, next_N, readout_head_dyn, n_heads, n_forget))

    # check if advance_by_accuracy function has loss_threshold argument not None
    advance_by_acc_sig = advance_by_accuracy.__code__.co_varnames
    advance_by_acc_has_loss_threshold = 'loss_threshold' in advance_by_acc_sig
    if advance_by_acc_has_loss_threshold:
        print("advance_by_accuracy has loss_threshold argument; using it to determine success")
    
    # Phase A: CB
    phase = "CB_ONLY"
    use_eat_here=False
    set_active_module(model, phase)
    solved_cb = False
    model.cb_scale = 1.0  # reset CB scale at start of CB phase
    for ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Reached MAX_GLOBAL_EPOCHS during CB phase")
            break
        summ = train_steps(model, task_fn, active_set, batch_size, phase,device, criterion, head_idx,
                                   optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser, 
                                   learning_alg=learning_alg,spec=spec, cb_l2=cb_l2, rnn_eat=use_eat_here, rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type,cb_sees_input=cb_sees_input)
        metrics = evaluate(model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx,spec=spec)
        acc = metrics['score'] if 'score' in metrics else 0.0
        row = {
            "phase": phase,
            "N": next_N,
            "loss": summ['loss'],
            "acc": acc,
            "gRNN": summ['gRNN'],
            "gCB": summ['gCB'],
            "epoch": global_epoch,
            # **{k: v for k, v in summ.items() if k != "dbg"},
            # **summ.get("dbg", {}),
        }
        logs.append(row)
        if kwargs.get("on_log"):
            kwargs["on_log"](row)
        print(f"Phase {phase} | N={next_N} | Global Epoch={global_epoch} | Loss={row['loss']:.4f} | Acc={row['acc']:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}", flush=True)
        if spec["advance_fn"](metrics):
            solved_cb = True
            print("CB solved N=", next_N,metrics)
            break
        
    if not solved_cb:
        # Return global_epoch so we don't lose the count of wasted epochs
        return False, active_Ns, current_N, logs, global_epoch 
    return True, active_set, next_N, logs, global_epoch 


def _is_reservoir_refresh_n(next_n: int, anchor_n: int, interval: int) -> bool:
    if interval <= 0:
        raise ValueError("reservoir_interval must be > 0")
    if next_n <= anchor_n:
        return False
    return (next_n - anchor_n) % interval == 0


def variant_cb_only_reservoir(*, model, task_fn, active_Ns, current_N, Ns_init,
                        batch_size,training_steps_n, test_steps, device,
                        criterion, head_idx, optimizer, get_grad_norms, global_epoch, 
                        set_active_module, readout_head_dyn, n_heads, n_forget,
                        THRESHOLD_FINAL, MAX_STAGE_EPOCHS, MAX_GLOBAL_EPOCHS, shared_optimiser,
                        learning_alg,spec, cb_l2=0.0,rnn_eat=False,rnn_eat_lambda=0.1,rnn_eat_loss_type='hidden', cb_sees_input=True, **kwargs):
    """
    Reservoir schedule for the normal alternating setup.

    Behavior:
      - Base phase still solves the first N with RNN_ONLY.
      - Between reservoir points, train CB_ONLY.
      - Every `reservoir_interval` Ns, train BOTH modules for that N once it is
        reached, then return to CB_ONLY until the next reservoir point.

    Example with `reservoir_interval=10` and Ns_init[0]=2:
      3..11 -> CB_ONLY
      12    -> BOTH
      13..21 -> CB_ONLY
      22    -> BOTH
    """
    logs = []
    next_N = int(kwargs.get("next_N_override", current_N + 1))
    reservoir_interval = int(kwargs.get("reservoir_interval", 10))
    reservoir_anchor_n = int(kwargs.get("reservoir_anchor_n", Ns_init[0]))
    active_set = sorted(compute_active_set(active_Ns, next_N, readout_head_dyn, n_heads, n_forget))

    if _is_reservoir_refresh_n(next_N, reservoir_anchor_n, reservoir_interval):
        phase = "BOTH"
        print(
            f"[reservoir] refresh N={next_N} (anchor={reservoir_anchor_n}, interval={reservoir_interval}) -> BOTH",
            flush=True,
        )
    else:
        phase = "CB_ONLY"

    advance_by_acc_sig = advance_by_accuracy.__code__.co_varnames
    advance_by_acc_has_loss_threshold = 'loss_threshold' in advance_by_acc_sig
    if advance_by_acc_has_loss_threshold:
        print("advance_by_accuracy has loss_threshold argument; using it to determine success")

    use_eat_here = False
    set_active_module(model, phase)
    solved = False
    if hasattr(model, "cb_scale"):
        model.cb_scale = 1.0

    for ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print(f"Reached MAX_GLOBAL_EPOCHS during {phase} phase")
            break

        summ = train_steps(
            model, task_fn, active_set, batch_size, phase, device, criterion, head_idx,
            optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser,
            learning_alg=learning_alg, spec=spec, cb_l2=cb_l2, rnn_eat=use_eat_here,
            rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type, cb_sees_input=cb_sees_input,
        )
        metrics = evaluate(model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx, spec=spec)
        acc = metrics['score'] if 'score' in metrics else 0.0
        row = {
            "phase": phase,
            "N": next_N,
            "loss": summ['loss'],
            "acc": acc,
            "gRNN": summ['gRNN'],
            "gCB": summ['gCB'],
            "epoch": global_epoch,
        }
        logs.append(row)
        if kwargs.get("on_log"):
            kwargs["on_log"](row)
        print(
            f"Phase {phase} | N={next_N} | Global Epoch={global_epoch} | Loss={row['loss']:.4f} | "
            f"Acc={row['acc']:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}",
            flush=True,
        )
        if spec["advance_fn"](metrics):
            solved = True
            print(f"{phase} solved N={next_N}", metrics)
            break

    if not solved:
        return False, active_Ns, current_N, logs, global_epoch

    return True, active_set, next_N, logs, global_epoch