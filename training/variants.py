# alex_training/alternating/variants.py
from .base import compute_active_set, train_steps, evaluate
import os
import sys

try:
    from src.tasks.task_registry import advance_by_accuracy
except ImportError:
    # Fallback: compute path relative to this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.dirname(current_dir)  # Go up from tasks/ → src/
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from tasks.task_registry import advance_by_accuracy, advance_by_mse
    
def variant_cb_then_rnn(*, model, task_fn, active_Ns, current_N, Ns_init,
                        batch_size,training_steps_n, test_steps, device,
                        criterion, head_idx, optimizer, get_grad_norms, global_epoch, 
                        set_active_module, readout_head_dyn, n_heads, n_forget,
                        THRESHOLD_FINAL, MAX_STAGE_EPOCHS, MAX_GLOBAL_EPOCHS, shared_optimiser,
                        learning_alg,spec, cb_l2=0.0,rnn_eat=False,rnn_eat_lambda=0.1,rnn_eat_loss_type='hidden', **kwargs):
    """
    Returns: (success, new_active_Ns, new_current_N, logs, updated_global_epoch)
    """
    logs = []
    next_N = int(kwargs.get("next_N_override", current_N + 1))
    active_set = sorted(compute_active_set(active_Ns, next_N, readout_head_dyn, n_heads, n_forget))

    # # check if advance_by_accuracy function has loss_threshold argument not None
    # advance_by_acc_sig = advance_by_accuracy.__code__.co_varnames
    # advance_by_acc_has_loss_threshold = 'loss_threshold' in advance_by_acc_sig
    # if advance_by_acc_has_loss_threshold:
    #     print("advance_by_accuracy has loss_threshold argument; using it to determine success")
    
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
                                   learning_alg=learning_alg,spec=spec, cb_l2=cb_l2, rnn_eat=use_eat_here, rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type)
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
            print("CB solved N=", next_N, metrics)
            break
        
    if not solved_cb:
        # Return global_epoch so we don't lose the count of wasted epochs
        return False, active_Ns, current_N, logs, global_epoch 

    # Phase B: RNN (decaying CB if applicable)
    phase="RNN_ONLY"
    use_eat_here=rnn_eat
    set_active_module(model, phase)
    solved_rnn = False
    for ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Reached MAX_GLOBAL_EPOCHS during RNN phase")
            break
        summ = train_steps(model, task_fn, active_set, batch_size, phase,device, criterion, head_idx,
                                   optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser, 
                                   learning_alg=learning_alg,spec=spec, cb_l2=cb_l2, rnn_eat=use_eat_here, rnn_eat_lambda=rnn_eat_lambda)
        metrics = evaluate(model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx, spec=spec)
        acc = metrics['score'] if 'score' in metrics else 0.0
        row = {
            "phase": phase,
            "N": next_N,
            "loss": summ["loss"],
            "acc": acc,
            "gRNN": summ["gRNN"],
            "gCB": summ["gCB"],
            "epoch": global_epoch,
            # **{k: v for k, v in summ.items() if k != "dbg"},
            # **summ.get("dbg", {}),
        }
        logs.append(row)
        if kwargs.get("on_log"):
            kwargs["on_log"](row)
        print(f"Phase {phase} | N={next_N} | Global Epoch={global_epoch} | Loss={row['loss']:.4f} | Acc={row['acc']:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}", flush=True)
        if spec["advance_fn"](metrics):
            solved_rnn = True
            print("RNN solved N=", next_N, metrics)
            break

    if not solved_rnn:
        return False, active_Ns, current_N, logs, global_epoch 

    return True, active_set, next_N, logs, global_epoch 

def variant_train_simultaneous(*, model, task_fn, active_Ns, current_N, Ns_init,
                        batch_size,training_steps_n, test_steps, device,
                        criterion, head_idx, optimizer, get_grad_norms, global_epoch, 
                        set_active_module, readout_head_dyn, n_heads, n_forget,
                        THRESHOLD_FINAL, MAX_STAGE_EPOCHS, MAX_GLOBAL_EPOCHS, shared_optimiser,
                        learning_alg,spec, cb_l2=0.0,rnn_eat=False,rnn_eat_lambda=0.1,rnn_eat_loss_type='hidden', **kwargs):
    """
    Returns: (success, new_active_Ns, new_current_N, logs, updated_global_epoch)
    """
    logs = []
    next_N = int(kwargs.get("next_N_override", current_N + 1))
    active_set = sorted(compute_active_set(active_Ns, next_N, readout_head_dyn, n_heads, n_forget))

    # # check if advance_by_accuracy function has loss_threshold argument not None
    # advance_by_acc_sig = advance_by_accuracy.__code__.co_varnames
    # advance_by_acc_has_loss_threshold = 'loss_threshold' in advance_by_acc_sig
    # if advance_by_acc_has_loss_threshold:
    #     print("advance_by_accuracy has loss_threshold argument; using it to determine success")
    
    # Phase A: CB
    phase = "BOTH"
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
                                   learning_alg=learning_alg,spec=spec, cb_l2=cb_l2, rnn_eat=use_eat_here, rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type)
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

def variant_rnn_then_cb_finetune(*, model, task_fn, active_Ns, current_N, Ns_init,
                                 batch_size,training_steps_n, test_steps, device,
                                 criterion, head_idx, optimizer, get_grad_norms, global_epoch,
                                 set_active_module, readout_head_dyn, n_heads, n_forget, shared_optimiser,
                                 THRESHOLD_RNN_LEAD, THRESHOLD_FINAL, MAX_STAGE_EPOCHS, MAX_GLOBAL_EPOCHS,learning_alg,
                                 spec, cb_l2=0.0, rnn_eat=False, rnn_eat_lambda=0.1, rnn_eat_loss_type='hidden',
                                  **kwargs):
    
    logs = []
    next_N = int(kwargs.get("next_N_override", current_N + 1))
    active_set = sorted(compute_active_set(active_Ns, next_N, readout_head_dyn, n_heads, n_forget))

    # Phase A: RNN lead
    phase = "RNN_ONLY"
    rnn_eat_here=rnn_eat
    set_active_module(model, phase)
    solved_rnn = False
    for ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Reached MAX_GLOBAL_EPOCHS during RNN lead phase")
            break
        summ = train_steps(model, task_fn, active_set, batch_size, phase,device, criterion, head_idx,
                           optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser,
                           learning_alg=learning_alg,spec=spec, cb_l2=cb_l2, rnn_eat=rnn_eat_here,
                           rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type)
        acc = evaluate(model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx,spec=spec)
        row = {
            "phase": "RNN_lead",
            "N": next_N,
            "loss": summ["loss"],
            "acc": acc,
            "gRNN": summ["gRNN"],
            "gCB": summ["gCB"],
            "epoch": global_epoch,
            **{k: v for k, v in summ.items() if k != "dbg"},
            **summ.get("dbg", {}),
        }
        logs.append(row)
        if kwargs.get("on_log"):
            kwargs["on_log"](row)
        print(f"Phase {phase} | N={next_N} | Global Epoch={global_epoch} | Loss={row['loss']:.4f} | Acc={acc:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}", flush=True)
        if acc >= THRESHOLD_RNN_LEAD:
            solved_rnn = True
            print(f"RNN lead solved N={next_N} Acc={acc:.2f}%")
            break
            
    if not solved_rnn:
        return False, active_Ns, current_N, logs, global_epoch 

    # Phase B: CB finetune
    phase="CB_ONLY"
    rnn_eat_here=False
    set_active_module(model, phase)
    solved_cb = False
    for ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Reached MAX_GLOBAL_EPOCHS during CB finetune phase")
            break
        summ = train_steps(model, task_fn, active_set, batch_size, phase,device, criterion, head_idx,
                           optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser, learning_alg=learning_alg,
                           spec=spec, cb_l2=cb_l2, rnn_eat=rnn_eat_here, rnn_eat_lambda=rnn_eat_lambda,
                           rnn_eat_loss_type=rnn_eat_loss_type)
        acc = evaluate(model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx,spec=spec)
        row = {
            "phase": "CB_finetune",
            "N": next_N,
            "loss": summ["loss"],
            "acc": acc,
            "gRNN": summ["gRNN"],
            "gCB": summ["gCB"],
            "epoch": global_epoch,
            **{k: v for k, v in summ.items() if k != "dbg"},
            **summ.get("dbg", {}),
        }
        logs.append(row)
        if kwargs.get("on_log"):
            kwargs["on_log"](row)
        print(f"Phase CB finetune | N={next_N} | Global Epoch={global_epoch} | Loss={row['loss']:.4f} | Acc={acc:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}", flush=True)
        if acc >= THRESHOLD_FINAL:
            solved_cb = True
            print(f"CB finetuned N={next_N} Acc={acc:.2f}%")
            break
            
    if not solved_cb:
        return False, active_Ns, current_N, logs, global_epoch 

    return True, active_set, next_N, logs, global_epoch 

def variant_interleaved_finetune(*, model, task_fn, active_Ns, current_N, Ns_init,
                                 batch_size, training_steps_n, test_steps, device,
                                 criterion, head_idx, optimizer, get_grad_norms, global_epoch,
                                 set_active_module, readout_head_dyn, n_heads, n_forget,shared_optimiser,
                                 THRESHOLD_CB_LEAD, THRESHOLD_RNN_LEAD, THRESHOLD_FINAL, 
                                 MAX_STAGE_EPOCHS, MAX_GLOBAL_EPOCHS,learning_alg,spec,cb_l2=0.0,
                                 rnn_eat=False, rnn_eat_lambda=0.1, rnn_eat_loss_type='hidden', **kwargs):
    
    logs = []
    next_N = int(kwargs.get("next_N_override", current_N + 1))
    active_set = sorted(compute_active_set(active_Ns, next_N, readout_head_dyn, n_heads, n_forget))

    # Phase A: CB Trains to THRESHOLD_CB_LEAD
    phase = "CB_ONLY"
    rnn_eat_here=False
    set_active_module(model, phase)
    solved_cb = False
    for ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Reached MAX_GLOBAL_EPOCHS during CB lead phase")
            break
        summ = train_steps(model, task_fn, active_set, batch_size, phase,device, criterion, head_idx,
                           optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser,
                           learning_alg=learning_alg, spec=spec, cb_l2=cb_l2, rnn_eat=rnn_eat_here,
                           rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type)
        acc = evaluate(model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx,spec=spec)
        row = {
            "phase": "CB_lead",
            "N": next_N,
            "loss": summ["loss"],
            "acc": acc,
            "gRNN": summ["gRNN"],
            "gCB": summ["gCB"],
            "epoch": global_epoch,
            **{k: v for k, v in summ.items() if k != "dbg"},
            **summ.get("dbg", {}),
        }
        logs.append(row)
        if kwargs.get("on_log"):
            kwargs["on_log"](row)
        print(f"Phase CB lead | N={next_N} | Global Epoch={global_epoch} | Loss={row['loss']:.4f} | Acc={acc:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}", flush=True)
        if acc >= THRESHOLD_CB_LEAD:
            solved_cb = True
            print(f"CB lead solved N={next_N} Acc={acc:.2f}%")
            break
            
    if not solved_cb:
        return False, active_Ns, current_N, logs, global_epoch 

    # Phase B: train RNN to THRESHOLD_RNN_LEAD
    phase = "RNN_ONLY"
    rnn_eat_here=rnn_eat
    set_active_module(model, phase)
    solved_rnn = False
    for ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Reached MAX_GLOBAL_EPOCHS during RNN lead phase")
            break
        summ = train_steps(model, task_fn, active_set, batch_size, phase,device, criterion, head_idx,
                           optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser,
                           learning_alg=learning_alg,spec=spec, cb_l2=cb_l2, rnn_eat=rnn_eat_here,
                           rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type)
        acc = evaluate(model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx,spec=spec)
        row = {
            "phase": "RNN",
            "N": next_N,
            "loss": summ["loss"],
            "acc": acc,
            "gRNN": summ["gRNN"],
            "gCB": summ["gCB"],
            "epoch": global_epoch,
            **{k: v for k, v in summ.items() if k != "dbg"},
            **summ.get("dbg", {}),
        }
        logs.append(row)
        if kwargs.get("on_log"):
            kwargs["on_log"](row)
        print(f"Phase RNN | N={next_N} | Global Epoch={global_epoch} | Loss={row['loss']:.4f} | Acc={acc:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}", flush=True)
        if acc >= THRESHOLD_RNN_LEAD:
            solved_rnn = True
            print(f"RNN lead solved N={next_N} Acc={acc:.2f}%")
            break
            
    if not solved_rnn:
        return False, active_Ns, current_N, logs, global_epoch 

    # Phase C: CB finetunes to final threshold
    phase = "CB_ONLY"
    rnn_eat_here=False
    set_active_module(model, "CB_ONLY")
    solved_cb = False
    for ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1
        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print("Reached MAX_GLOBAL_EPOCHS during CB finetune phase")
            break
        summ = train_steps(model, task_fn, active_set, batch_size, phase,device, criterion, head_idx,
                           optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser,
                           learning_alg=learning_alg, spec=spec, cb_l2=cb_l2, rnn_eat=rnn_eat_here,
                           rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type)
        acc = evaluate(model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx,spec=spec)
        row = {
            "phase": "CB_finetune",
            "N": next_N,
            "loss": summ["loss"],
            "acc": acc,
            "gRNN": summ["gRNN"],
            "gCB": summ["gCB"],
            "epoch": global_epoch,
            **{k: v for k, v in summ.items() if k != "dbg"},
            **summ.get("dbg", {}),
        }
        logs.append(row)
        if kwargs.get("on_log"):
            kwargs["on_log"](row)
        print(f"Phase CB finetune | N={next_N} | Global Epoch={global_epoch} | Loss={row['loss']:.4f} | Acc={acc:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}", flush=True)
        if acc >= THRESHOLD_FINAL:
            solved_cb = True
            print(f"CB finetuned N={next_N} Acc={acc:.2f}%")
            break
            
    if not solved_cb:
        return False, active_Ns, current_N, logs, global_epoch 

    return True, active_set, next_N, logs, global_epoch 
def variant_interleaved_curr(*, model, task_fn, active_Ns, current_N, Ns_init,
                             batch_size, training_steps_n, test_steps, device,
                             criterion, head_idx, optimizer, get_grad_norms, global_epoch,
                             set_active_module, readout_head_dyn, n_heads, n_forget,shared_optimiser,
                             THRESHOLD_FINAL, MAX_STAGE_EPOCHS, MAX_GLOBAL_EPOCHS, learning_alg,
                             spec,cb_l2=0.0, rnn_eat=False, rnn_eat_lambda=0.1, rnn_eat_loss_type='hidden', **kwargs):
    """
    Interleaved schedule:
      - Primary: CB tries N+1, then RNN tries N+2 (if CB succeeded).
      - Fallback: If primary module fails on a target N, the other module gets one try on the SAME N.
      - Neither module may attempt the same N more than once. If both fail -> return False.

    Returns: (success, new_active_Ns, new_current_N, logs, updated_global_epoch)
    """
    logs = []

    # Always keep active_Ns a list with stable order
    if not isinstance(active_Ns, list):
        active_Ns = list(active_Ns)

    def attempt(mode: str, target_N: int):
        """
        One module attempts one target N for up to MAX_STAGE_EPOCHS.
        Returns (solved: bool, global_epoch: int).
        """
        nonlocal global_epoch, logs

        active_set = [target_N]
        set_active_module(model, "CB_ONLY" if mode == "CB" else "RNN_ONLY")

        solved = False
        for _ in range(MAX_STAGE_EPOCHS):
            global_epoch += 1
            if global_epoch >= MAX_GLOBAL_EPOCHS:
                print(f"Reached MAX_GLOBAL_EPOCHS during {mode} phase")
                break
            phase=mode
            if mode=="RNN":
                rnn_eat_here=rnn_eat
            else:
                rnn_eat_here=False

            summ = train_steps(
                model, task_fn, active_set, batch_size, phase,device, criterion, head_idx,
                optimizer, get_grad_norms, training_steps=training_steps_n, shared_optimiser=shared_optimiser, 
                learning_alg=learning_alg, spec=spec, cb_l2=cb_l2, rnn_eat=rnn_eat_here,
                rnn_eat_lambda=rnn_eat_lambda, rnn_eat_loss_type=rnn_eat_loss_type
            )
            acc = evaluate(
                model, task_fn, active_set, batch_size, test_steps, device, criterion, head_idx,spec=spec
            )

            row = {
                "phase": mode,
                "N": target_N,
                "loss": summ["loss"],
                "acc": acc,
                "gRNN": summ["gRNN"],
                "gCB": summ["gCB"],
                "epoch": global_epoch,
                "cb_scale": getattr(model, "cb_scale", None),
                **{k: v for k, v in summ.items() if k != "dbg"},
                **summ.get("dbg", {}),
            }
            logs.append(row)
            if kwargs.get("on_log"):
                kwargs["on_log"](row)

            print(
                f"Phase {mode} | N={target_N} | Global Epoch={global_epoch} "
                f"| Loss={row['loss']:.4f} | Acc={acc:.2f}% | gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}"
                + (f" | CB Scale={model.cb_scale:.4f}" if hasattr(model, "cb_scale") else ""),
                flush=True
            )

            if acc >= THRESHOLD_FINAL:
                solved = True
                print(f"{mode} solved N={target_N} (acc={acc:.2f}%)", flush=True)
                break

        # reset cb_scale after attempt
        if hasattr(model, "cb_scale"):
            model.cb_scale = 1.0

        return solved

    def add_active(N):
        nonlocal active_Ns
        if N not in active_Ns:
            active_Ns.append(N)

    # ---- Step 1: target for CB (primary) ----
    target_cb = current_N + 1

    # CB tries target_cb once
    solved = attempt("CB", target_cb)
    if not solved:
        # fallback: RNN tries SAME target once
        print(f"CB failed on N={target_cb}; RNN fallback tries same N once...", flush=True)
        solved = attempt("RNN", target_cb)

        if not solved:
            # both used their one attempt -> quit
            return False, active_Ns, current_N, logs, global_epoch

    # If either CB or fallback RNN solved target_cb, we advance curriculum state
    add_active(target_cb)
    current_N = target_cb

    # ---- Step 2: target for RNN (primary next) ----
    target_rnn = current_N + 1

    solved = attempt("RNN", target_rnn)
    if not solved:
        # fallback: CB tries SAME target once
        print(f"RNN failed on N={target_rnn}; CB fallback tries same N once...", flush=True)
        solved = attempt("CB", target_rnn)

        if not solved:
            return False, active_Ns, current_N, logs, global_epoch

    add_active(target_rnn)
    current_N = target_rnn

    return True, active_Ns, current_N, logs, global_epoch
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