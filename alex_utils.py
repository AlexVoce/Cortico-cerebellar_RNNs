# alex_utils.py - utility functions for my curriculum learning experiments
import torch

def get_grad_norms(model): 
    """
    Calculates L2 norm of gradients for:
    1. RNN Core (non-CB, non-head parameters)
    2. Cerebellum (CB parameters)
    3. Readout Heads (head parameters)
    """
    device = next(model.parameters()).device
    
    # Get parameter sets
    cb_params = set(model.cb.parameters()) if hasattr(model, 'cb') and model.cb else set()
    head_params = set()
    if hasattr(model, 'heads'):
        for head in model.heads:
            head_params.update(head.parameters())
    
    # Initialize accumulators
    rnn_norm_sq = torch.tensor(0.0, device=device)
    cb_norm_sq = torch.tensor(0.0, device=device)
    head_norm_sq = torch.tensor(0.0, device=device)
    
    # Accumulate squared norms
    for param in model.parameters():
        if param.grad is None:
            continue
            
        grad_sq = param.grad.norm(2) ** 2
        
        if param in cb_params:
            cb_norm_sq += grad_sq
        elif param in head_params:
            head_norm_sq += grad_sq
        else:
            rnn_norm_sq += grad_sq
    
    # Return sqrt of accumulated squares
    return (rnn_norm_sq.sqrt().item(), 
            cb_norm_sq.sqrt().item(), 
            head_norm_sq.sqrt().item())

def set_active_module(model, mode):
    # freeze all
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    cb_params = list(model.cb.parameters()) if getattr(model, "cb", None) is not None else []

    head_params = []
    if hasattr(model, "heads"):
        for head in model.heads:
            head_params.extend(list(head.parameters()))

    # heads always trainable
    for p in head_params:
        p.requires_grad = True

    cb_param_ids = {id(p) for p in cb_params}
    head_param_ids = {id(p) for p in head_params}

    if mode == "RNN_ONLY" or mode == "RNN":
        # train everything except CB and heads
        for p in model.parameters():
            if id(p) not in cb_param_ids and id(p) not in head_param_ids:
                p.requires_grad = True
        print(">> MODE: RNN core + Heads (CB frozen)", flush=True)

    elif mode == "CB_ONLY" or mode == "CB":
        for p in cb_params:
            p.requires_grad = True
        print(">> MODE: Cerebellum + Heads (RNN core frozen)", flush=True)

    elif mode == "BOTH":
        for p in model.parameters():
            if id(p) not in head_param_ids:
                p.requires_grad = True
        print(">> MODE: RNN core + Cerebellum + Heads", flush=True)

    else:
        raise ValueError(f"Unknown mode: {mode}")
    
def ensure_list(x):
    return x if isinstance(x, list) else [x]

def step_optimizer(optimizer, phase, shared_optimiser=True):
    if shared_optimiser:
        optimizer.step()
    else:
        if phase in ("RNN_ONLY", "RNN"):
            optimizer["RNN"].step()
        elif phase in ("CB_ONLY", "CB"):
            if optimizer["CB"]:
                optimizer["CB"].step()
        else:  # both active
            optimizer["RNN"].step()
            if optimizer["CB"]:
                optimizer["CB"].step()
