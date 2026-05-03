import sys
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
alex_utils_path = os.path.dirname(__file__)
if alex_utils_path not in sys.path:
    sys.path.insert(0, alex_utils_path)
from src.utils import save_model, find_next_free_network_number
from alex_utils import get_grad_norms, set_active_module, ensure_list

def train_alternating(model, curriculum_type, task_function, num_epochs, Ns_init, run_number, 
                            batch_size, training_steps, test_steps, 
                            device, base_path, affixes, 
                            n_heads=1, n_forget=1,task_name='dms',
                            scramble=False, rnn_lr = 0.05, cb_lr=0.05,
                            readout_head_dyn='sliding', cb_store=False):
    """
    Custom curriculum: 
    1. Train RNN on N (Base).
    2. Freeze RNN, Train CB on N+1 (Bias).
    3. Freeze CB, Train RNN on N+1 (Consolidate).
    4. Repeat.
    """
    # Configuration
    THRESHOLD = 98.0 # accuracy threshold to consider task "solved"
    MAX_STAGE_EPOCHS = num_epochs 
    MAX_GLOBAL_EPOCHS = num_epochs 
    TARGET_END_N = 150 # final N to reach
    criterion = nn.CrossEntropyLoss() # standard cross-entropy loss
    
    # Helper to map N to head index
    if not hasattr(model, "heads"):
        raise AttributeError("Model has no attribute 'heads'.")
    def head_idx(n): # function to map task N to head index
        idx = n - Ns_init[0] 
        if idx < 0 or idx >= len(model.heads):
            raise IndexError(f"Task N={n} maps to head index {idx}, but only {len(model.heads)} heads available")
        return idx

    current_N = Ns_init[0] # Start at the easiest N
    active_Ns = [current_N] # Tracks list of active tasks

    global_epoch = 0 
    stats = {'n_task': [], 'phase': [], 'loss': [], 'accuracy': [], 'grad_rnn': [], 'grad_cb': [], 'epoch': []} 

    # initialise save directory
    run_number = find_next_free_network_number(
        base_path=base_path, curriculum_type=curriculum_type, task=task_name,
        affixes=affixes, n_heads=n_heads, n_forget=n_forget      
    )
    print(f"=== Saving to network number: {run_number} ===", flush=True)

    subdir = save_model(model, curriculum_type, n_heads, n_forget, task_name, 
                        run_number, current_N, current_N, base_path=base_path, affixes=affixes)
    
    cb_params = list(model.cb.parameters()) if (hasattr(model, "cb") and model.cb) else []
    cb_param_ids = {id(p) for p in cb_params}

    rnn_params = [p for p in model.parameters() if id(p) not in cb_param_ids]

    optimizer = torch.optim.SGD(
        [
            {"params": rnn_params, "lr": rnn_lr},
            {"params": cb_params, "lr": cb_lr},
        ],
        momentum=0.1,
        nesterov=True
    )

    try:
        # BEGIN! Get RNN to learn base task first (Phase 0)
        print(f"\n=== PHASE 0: Establish Base at N={active_Ns} ===", flush=True)
        set_active_module(model, 'RNN_ONLY') 
        
        for epoch in range(num_epochs): 
            global_epoch += 1
            if global_epoch > MAX_GLOBAL_EPOCHS:
                print(f"Reached max global epochs ({MAX_GLOBAL_EPOCHS}). Stopping.", flush=True)
                np.save(os.path.join(subdir, 'stats.npy'), stats)
                save_model(model, curriculum_type, n_heads, n_forget, task_name, run_number, 
                        current_N, current_N, base_path=base_path, affixes=affixes)
                return stats

            losses_step = []
            grad_rnn_step = []; grad_cb_step = []

            model.train() 
            pbar = tqdm(range(training_steps), desc=f"Init N={active_Ns}", leave=False) 
            for _ in pbar:
                optimizer.zero_grad()
                sequences, labels = task_function(active_Ns, batch_size) 
                labels = ensure_list(labels)
                sequences, labels = sequences.to(device), [l.to(device) for l in labels] 

                out, out_class = model(sequences)
                out_class = ensure_list(out_class)
                
                loss = 0.0
                for li, n in enumerate(active_Ns):
                    loss += criterion(out_class[head_idx(n)], labels[li])            
                loss.backward()

                losses_step.append(loss.item()) 
                # clip grads to avoid explosions
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                # store post-clip grad norms (efficiently!)
                grad_rnn, grad_cb, _ = get_grad_norms(model)
                grad_rnn_step.append(grad_rnn)
                grad_cb_step.append(grad_cb)

                optimizer.step() 
                pbar.set_postfix({'Loss': f'{np.mean(losses_step):.4f}', 'gRNN': f'{np.mean(grad_rnn_step):.4f}'})
            
            # test step
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for _ in range(test_steps):
                    sequences, labels = task_function(active_Ns, batch_size)
                    labels = ensure_list(labels)
                    sequences, labels = sequences.to(device), [l.to(device) for l in labels]
                    _, out_class = model(sequences)
                    for li, n in enumerate(active_Ns):
                        predicted = out_class[head_idx(n)].argmax(dim=1)
                        correct += (predicted == labels[li]).sum().item()
                        total += labels[li].size(0) 
            
            acc = 100 * correct / total if total > 0 else 0.0
            
            # Update Stats
            stats['n_task'].append(current_N)
            stats['phase'].append('init')
            stats['loss'].append(np.mean(losses_step))
            stats['accuracy'].append(acc)
            stats['grad_rnn'].append(np.mean(grad_rnn_step))
            stats['grad_cb'].append(np.mean(grad_cb_step))
            stats['epoch'].append(global_epoch)

            np.save(os.path.join(subdir, 'stats.npy'), stats)
            print(f'Init N={current_N} | Ep {epoch+1}| Global Ep {global_epoch} | Acc: {acc:.2f}% | gRNN: {np.mean(grad_rnn_step):.4f} | gCB: {np.mean(grad_cb_step):.4f}', flush=True)
            
            if acc > THRESHOLD:
                print(f"Base N={current_N} Solved!", flush=True)
                save_model(model, curriculum_type, n_heads, n_forget, task_name, run_number, current_N, current_N, base_path=base_path, affixes=affixes)
                break

        # After initial base training, begin alternating phases !!! exciting !!!
        while current_N < TARGET_END_N:
            next_N = current_N + 1 # define next N to learn
            
            temp_active_set = active_Ns + [next_N] 
            
            # figure out current active set based on readout head dynamics
            if readout_head_dyn == 'cumulative':
                current_active_set = temp_active_set 
            else: # sliding
                if len(temp_active_set) > n_heads:
                    current_active_set = temp_active_set[n_forget:] # slide window
                else:
                    current_active_set = temp_active_set

            # CB Training - Scouting
            print(f"\n=== Phase A: CB Training on {current_active_set} (Store={cb_store}) ===", flush=True)
            set_active_module(model, 'CB_ONLY')
            
            solved_cb = False 
            for epoch in range(MAX_STAGE_EPOCHS):
                losses_step = []
                grad_rnn_step = []; grad_cb_step = []
                global_epoch += 1
                if global_epoch > MAX_GLOBAL_EPOCHS:
                    print(f"Reached max global epochs ({MAX_GLOBAL_EPOCHS}). Stopping.", flush=True)
                    np.save(os.path.join(subdir, 'stats.npy'), stats)
                    save_model(model, curriculum_type, n_heads, n_forget, task_name, run_number, 
                            current_N, current_N, base_path=base_path, affixes=affixes)
                    return stats
                
                model.train()
                pbar = tqdm(range(training_steps), desc=f"CB Trying N={next_N}", leave=False)
                for _ in pbar:
                    optimizer.zero_grad()
                    sequences, labels = task_function(current_active_set, batch_size)
                    labels = ensure_list(labels)
                    sequences, labels = sequences.to(device), [l.to(device) for l in labels]
                    
                    _, out_class = model(sequences)
                    out_class = ensure_list(out_class)

                    # custom loss calculation for CB training set
                    loss = 0.0
                    for li, n in enumerate(current_active_set):
                        loss += criterion(out_class[head_idx(n)], labels[li])
                    loss.backward()
                    losses_step.append(loss.item())

                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    grad_rnn, grad_cb, _ = get_grad_norms(model)
                    grad_rnn_step.append(grad_rnn)
                    grad_cb_step.append(grad_cb)
                    
                    optimizer.step()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'gCB': f"{grad_cb:.4f}"})
                    
                # Test (CB stage)
                model.eval()
                correct = 0; total = 0
                with torch.no_grad():
                    for _ in range(test_steps):
                        sequences, labels = task_function(current_active_set, batch_size)
                        labels = ensure_list(labels)
                        sequences, labels = sequences.to(device), [l.to(device) for l in labels]

                        _, out_class = model(sequences)
                        out_class = ensure_list(out_class)
                        for li, n in enumerate(current_active_set):
                            predicted = out_class[head_idx(n)].argmax(dim=1) 
                            correct += (predicted == labels[li]).sum().item()
                            total += labels[li].size(0)                
                acc = 100 * correct / total if total > 0 else 0.0
                
                stats['n_task'].append(next_N)
                stats['phase'].append('CB')
                stats['loss'].append(np.mean(losses_step))
                stats['accuracy'].append(acc)
                stats['grad_rnn'].append(np.mean(grad_rnn_step))
                stats['grad_cb'].append(np.mean(grad_cb_step))
                stats['epoch'].append(global_epoch)

                np.save(os.path.join(subdir, 'stats.npy'), stats)
                print(f"CB N={next_N} | Ep {epoch+1} | Global Ep {global_epoch} | Acc: {acc:.2f}% | gRNN: {np.mean(grad_rnn_step):.4f} | gCB: {np.mean(grad_cb_step):.4f}", flush=True) 
                
                if acc > THRESHOLD:
                    print(f">> CB successfully biased N={next_N}!", flush=True)
                    solved_cb = True
                    break
            
            if not solved_cb: return stats 
            
            # RNN CONSOLIDATION TIME!
            print(f"\n=== Phase B: RNN Consolidating {current_active_set} ===", flush=True)
            set_active_module(model, 'RNN_ONLY')
            
            solved_rnn = False
            for epoch in range(MAX_STAGE_EPOCHS):
                losses_step = []
                grad_rnn_step = []; grad_cb_step = []
                global_epoch += 1
                if global_epoch > MAX_GLOBAL_EPOCHS:
                    print(f"Reached max global epochs ({MAX_GLOBAL_EPOCHS}). Stopping.", flush=True)
                    np.save(os.path.join(subdir, 'stats.npy'), stats)
                    save_model(model, curriculum_type, n_heads, n_forget, task_name, run_number, 
                            current_N, current_N, base_path=base_path, affixes=affixes)
                    return stats
                                
                model.train()
                pbar = tqdm(range(training_steps), desc=f"RNN Consolidating N={next_N}", leave=False)
                for _ in pbar:
                    optimizer.zero_grad()
                    sequences, labels = task_function(current_active_set, batch_size)
                    labels = ensure_list(labels)
                    sequences, labels = sequences.to(device), [l.to(device) for l in labels]
                    
                    _, out_class = model(sequences)
                    out_class = ensure_list(out_class)
                    
                    # RNN always sees full active set
                    loss = 0.0
                    for li, n in enumerate(current_active_set):
                        loss += criterion(out_class[head_idx(n)], labels[li])
                    loss.backward()
                    losses_step.append(loss.item())

                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    grad_rnn, grad_cb, _ = get_grad_norms(model)
                    grad_rnn_step.append(grad_rnn)
                    grad_cb_step.append(grad_cb)
                    
                    optimizer.step()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'gRNN': f"{grad_rnn:.4f}"})
                    
                # Test
                model.eval()
                correct = 0; total = 0
                with torch.no_grad():
                    for _ in range(test_steps):
                        sequences, labels = task_function(current_active_set, batch_size)
                        labels = ensure_list(labels)
                        sequences, labels = sequences.to(device), [l.to(device) for l in labels]
                        _, out_class = model(sequences)
                        for li, n in enumerate(current_active_set):
                            predicted = out_class[head_idx(n)].argmax(dim=1) 
                            correct += (predicted == labels[li]).sum().item()
                            total += labels[li].size(0)     
                
                acc = 100 * correct / total if total > 0 else 0.0

                stats['n_task'].append(next_N)
                stats['phase'].append('RNN')
                stats['loss'].append(np.mean(losses_step))
                stats['accuracy'].append(acc)
                stats['grad_rnn'].append(np.mean(grad_rnn_step))
                stats['grad_cb'].append(np.mean(grad_cb_step))
                stats['epoch'].append(global_epoch)
                
                np.save(os.path.join(subdir, 'stats.npy'), stats)
                print(f"RNN N={next_N} | Ep {epoch+1} | Global Ep {global_epoch} | Acc: {acc:.2f}% | gRNN: {np.mean(grad_rnn_step):.4f} | gCB: {np.mean(grad_cb_step):.4f}", flush=True)

                if acc > THRESHOLD:
                    print(f">> RNN consolidated N={next_N}!", flush=True)
                    solved_rnn = True
                    current_N = next_N 
                    active_Ns = current_active_set
                    save_model(model, curriculum_type, n_heads, n_forget, task_name, run_number, current_N, current_N, base_path=base_path, affixes=affixes)
                    break
            
            if not solved_rnn: return stats
            
    except KeyboardInterrupt:
        print("Training interrupted by user. Returning current stats.", flush=True)
        np.save(os.path.join(subdir, 'stats.npy'), stats)
        return stats

    # Final Return
    print(f"✓ Training completed! Reached N={current_N}", flush=True)
    np.save(os.path.join(subdir, 'stats.npy'), stats)
    save_model(model, curriculum_type, n_heads, n_forget, task_name, run_number, 
            current_N, current_N, base_path=base_path, affixes=affixes)
    return stats