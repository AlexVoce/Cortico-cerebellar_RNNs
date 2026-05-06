import torch
import torch.nn as nn
import torch.nn.functional as F

def rescale_to_reference_norm(vec, ref_vec, max_ratio=1.0, eps=1e-8):
    """
    vec:     [B, D] tensor to be rescaled
    ref_vec: [B, D] reference tensor
    max_ratio: max allowed ||vec|| / ||ref_vec||
    """
    vec_norm = vec.norm(dim=-1, keepdim=True).clamp_min(eps)
    ref_norm = ref_vec.norm(dim=-1, keepdim=True).clamp_min(eps)

    max_allowed = max_ratio * ref_norm
    scale = torch.minimum(torch.ones_like(vec_norm), max_allowed / vec_norm)
    return vec * scale

class CB_bias(nn.Module):
    def __init__(
            self, 
            hidden_size:int, # input size to CB = RNN hidden size (for now...)
            gc_dim: int = 128, # dimension of granule cell layer, adjustable hyperparameter but set to 512 as default
            pc_dim: int = 64,
            dcn_dim: int = 64,
            input_size: int = 0, # if above 0, cb will also receive task input
            use_hidden: bool = True,
            sparsity: float = 0.0, # not using rn - sets GC layer sparsity (top-k)
            scramble: bool = False, # also not using rn - whether to add noise to the bias
            device: str = "cuda" if torch.cuda.is_available() else "cpu", 
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gc_dim = gc_dim
        self.pc_dim = pc_dim
        self.dcn_dim = dcn_dim
        self.sparsity = sparsity
        self.use_input = input_size > 0
        self.use_hidden = use_hidden
        self.k = int(self.gc_dim * self.sparsity)
        self.device = device

        if not self.use_hidden and not self.use_input:
            raise ValueError("CB_bias requires input_size > 0 when use_hidden is False")

        if self.use_hidden:
            gc_input_size = self.hidden_size + input_size if self.use_input else self.hidden_size
        else:
            gc_input_size = input_size
        self.gc = nn.Linear(gc_input_size, self.gc_dim, bias=True)
        self.pc = nn.Linear(self.gc_dim, self.pc_dim, bias=True)
        self.dcn = nn.Linear(self.pc_dim, self.dcn_dim, bias=True)
        if self.dcn_dim != self.hidden_size:
            self.dcn_proj = nn.Linear(self.dcn_dim, self.hidden_size, bias=True)

        # intiialise all of them at 0 bias and small weights so that CB bias starts off small and then can grow if useful
        nn.init.zeros_(self.gc.bias)
        nn.init.zeros_(self.pc.bias)
        nn.init.zeros_(self.dcn.bias)
        nn.init.normal_(self.gc.weight, std=0.01)
        nn.init.normal_(self.pc.weight, std=0.1)
        nn.init.normal_(self.dcn.weight, std=0.1)
        self.cb_parameters = list(self.gc.parameters()) + list(self.pc.parameters()) + list(self.dcn.parameters())

        self._hparams = dict(
            input_size=hidden_size,
            gc_dim=gc_dim,
            use_hidden=use_hidden,
            sparsity=sparsity,
            scramble=scramble,
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor = None,
        return_gc: bool = False,
        return_all: bool = False,
        scramble: bool = False,
    ):
        if not self.use_hidden:
            if not self.use_input:
                raise ValueError("CB_bias is configured without hidden input but no task input was provided")
            if x is None:
                raise ValueError("CB_bias is configured to use input only, but x is None")
            gc_input = x
        elif self.use_input and x is not None:
            gc_input = torch.cat([h, x], dim=-1)
        else:
            gc_input = h
        g_t = F.relu(self.gc(gc_input), inplace=False)   # [B, gc_dim]
        self._last_gc = g_t
        p_t = F.relu(self.pc(g_t), inplace=False)        # [B, pc_dim]
        if self.dcn_dim != self.hidden_size:
            d_t = self.dcn(p_t)                          # [B, dcn_dim]
            bias = self.dcn_proj(d_t)                    # [B, hidden_size]
        else:
            d_t = self.dcn(p_t)                          # [B, hidden_size]
            bias = d_t
        if return_all:
            return {"gc_input": gc_input, "gc": g_t, "pc": p_t, "dcn": d_t, "cb_bias": bias}
        if return_gc:
            return bias, g_t
        
        return bias
    
class ElmanRNNMultiHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2, # output classes, e.g. 2 for dms
        num_readout_heads: int = 1, # number of readout heads - readout heads are shaped [hidden_size, num_classes]
        tau: float = 1.5, # time constant - hyperparameter
        afunc=nn.LeakyReLU(), # activation function for RNN - default ReLU - this aint workimg rn 
        bias: bool = True, # should RNN use bias (in addition to CB bias if implemented)
        scramble: bool = False, # dont use 
        train_tau: bool = False, # not training tau rn 
        use_cb_bias: bool = True, # whether to use cerebellar bias 
        cb_gc_dim: int = 128, # dimension of granule cell layer in CB bias module
        cb_pc_dim: int = 64, # dimension of Purkinje cell layer in CB bias module
        cb_dcn_dim: int = 64, # dimension of DCN layer in CB bias module
        cb_sparsity: float = 0.0, # sparsity of granule cell layer in CB bias module - not using rn 
        multiply: bool = False, # whether CB bias will be applied as multiplactive 
        cb_no_hidden: bool = False, # if True, CB consumes task input only and not RNN hidden state
        rnn_eat: bool = False, # whether to use CB bias as loss term to encourage the RNN to "eat its own tail" and internalize the CB bias (inspired by Hwang et al. 2023) - only relevant if use_cb_bias is True
        rnn_eat_lambda: float = 0.1, # strength of RNN eat own tail loss term if rnn_eat is True
        rnn_eat_loss_type: str = 'hidden', # whether the RNN eat own tail loss should be applied to the hidden states ('hidden') or the CB bias itself ('cb')
        debug_stats: bool = False, # if True, collect expensive per-time-step debug stats
        cb_input_size: int = 0, # if > 0, CB bias module will also receive the task input (in addition to the RNN hidden state)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes 
        self.num_readout_heads = num_readout_heads
        self.afunc = afunc()
        self.tau = tau
        self.train_tau = train_tau
        self.scramble = scramble
        if self.train_tau == False:
            with torch.no_grad():
                self.register_buffer("tau_param", torch.tensor(self.tau, dtype=torch.float32)) # no training of tau

        self.use_cb_bias = use_cb_bias
        self.cb_no_hidden = cb_no_hidden

        # RNN core
        self.inp = nn.Linear(input_size, self.hidden_size, bias=bias)
        self.hh  = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        # Readout heads
        self.heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.num_classes) for _ in range(self.num_readout_heads)]
        )

        # Cerebellar bias module
        if self.use_cb_bias:
            if self.cb_no_hidden and cb_input_size <= 0:
                raise ValueError("cb_no_hidden=True requires cb_input_size > 0 so CB can receive task input")
            self.cb = CB_bias(hidden_size=self.hidden_size, 
            gc_dim=cb_gc_dim, 
            pc_dim=cb_pc_dim,
            dcn_dim=cb_dcn_dim,
            input_size=cb_input_size,
            use_hidden=not cb_no_hidden,
            sparsity=cb_sparsity,
            scramble=False,
            device=device)
            self.multiply = multiply
            self.cb_input_size = cb_input_size
        else:
            self.cb = None

        self.rnn_eat = rnn_eat
        self.rnn_eat_lambda = rnn_eat_lambda
        self.rnn_eat_loss_type = rnn_eat_loss_type
        self.debug_stats = debug_stats
        self._hparams = dict(
        model_type="ElmanRNNMultiHead",
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_readout_heads=num_readout_heads,
        tau=tau,
        train_tau=train_tau,
        bias=bias,
        scramble=scramble,
        use_cb_bias=use_cb_bias,
        cb_gc_dim=cb_gc_dim if use_cb_bias else None,
        cb_pc_dim=cb_pc_dim if use_cb_bias else None,
        cb_dcn_dim=cb_dcn_dim if use_cb_bias else None,
        cb_sparsity=cb_sparsity if use_cb_bias else None,
        multiply=multiply if use_cb_bias else None,
        cb_no_hidden=cb_no_hidden if use_cb_bias else None,
        rnn_eat=rnn_eat if use_cb_bias else None,
        rnn_eat_lambda=rnn_eat_lambda if (use_cb_bias and rnn_eat) else None,
        rnn_eat_loss_type=rnn_eat_loss_type if (use_cb_bias and rnn_eat) else None,
        debug_stats=debug_stats,
        activation=afunc.__name__,
        )
        
    def forward(
        self,
        data: torch.Tensor,
        hs=None,
        classify_in_time: bool = False,
        savetime: bool = False,
        index_in_head=None,
        return_timewise: bool = False,
        task_id: int = None,
        return_gc: bool = False,
        return_cb: bool = False,
        return_dynamics: bool = False,
    ):
        T, B, _ = data.shape
        device = data.device
        gc_act = None
        cb_bias = None
        gc_seq = [] if (self.use_cb_bias and return_gc and return_cb) else None
        cb_bias_seq = [] if (self.use_cb_bias and return_cb) else None

        # Full dynamics collectors
        h_seq_full = [] if return_dynamics else None
        pre_seq = [] if return_dynamics else None
        post_seq = [] if return_dynamics else None
        xproj_seq = [] if return_dynamics else None

        gc_input_seq = [] if (self.use_cb_bias and return_dynamics) else None
        gc_seq_full = [] if (self.use_cb_bias and return_dynamics) else None
        pc_seq = [] if (self.use_cb_bias and return_dynamics) else None
        dcn_seq = [] if (self.use_cb_bias and return_dynamics) else None
        cb_bias_seq_full = [] if (self.use_cb_bias and return_dynamics) else None

        if hs is None:
            h_teacher = 0.1 * torch.rand(B, self.hidden_size, device=device)
        else:
            h_teacher = hs[0] if isinstance(hs, list) else hs

        h_student = h_teacher.clone() if (self.rnn_eat and self.use_cb_bias) else None

        alpha = 1.0 / float(self.tau_param)
        x_proj = self.inp(data)

        hs_t = [] if savetime else None
        cb_t = [] if (savetime and self.use_cb_bias) else None
        h_seq = [] if return_timewise else None
        cb_diff_seq = [] if (self.rnn_eat and self.use_cb_bias) else None
        eat_loss_sum = None
        eat_count = 0
        eat_bad_count = 0
        self._dbg = {}
        if self.debug_stats:
            # for debugging - track norms and nonfinite counts
            dbg = {
                "pre_mean_sum": torch.zeros((), device=device),
                "pre_max":      torch.zeros((), device=device),
                "post_mean_sum": torch.zeros((), device=device),
                "post_max":      torch.zeros((), device=device),
                "ht_mean_sum":   torch.zeros((), device=device),
                "ht_max":        torch.zeros((), device=device),
                "cb_mean_sum":   torch.zeros((), device=device),
                "cb_max":        torch.zeros((), device=device),

                "nonfinite_pre": torch.zeros((), device=device),
                "nonfinite_post": torch.zeros((), device=device),
                "nonfinite_ht":  torch.zeros((), device=device),
                "nonfinite_cb":  torch.zeros((), device=device),
                "max_abs_logit": torch.zeros((), device=device),

                "T": torch.zeros((), device=device),
            }

        for t in range(T):
            # Teacher trajectory (with CB)
            h_prev_teacher = h_teacher
            pre_teacher = x_proj[t] + self.hh(h_prev_teacher)

            cb_dict = None

            if self.use_cb_bias:
                if task_id is not None and self.cb_input_size > 0:
                    torch_oh = torch.zeros(data.shape[1], self.cb_input_size, device=device)
                    torch_oh[:, task_id] = 1.0

                    if return_dynamics:
                        cb_dict = self.cb(h_prev_teacher, x=torch_oh, return_all=True)
                        b_cb = cb_dict["cb_bias"]
                    elif return_gc or return_cb:
                        b_cb, gc_act = self.cb(h_prev_teacher, x=torch_oh, return_gc=True)
                    else:
                        b_cb = self.cb(h_prev_teacher, x=torch_oh)

                    b_cb = rescale_to_reference_norm(
                        vec=b_cb,
                        ref_vec=pre_teacher,
                        max_ratio=self.cb_max_ratio if hasattr(self, "cb_max_ratio") else 1.0,
                    )

                    if cb_dict is not None:
                        cb_dict["cb_bias"] = b_cb

                elif self.cb_input_size > 0:
                    x_t = data[t]

                    if return_dynamics:
                        cb_dict = self.cb(h_prev_teacher, x=x_t, return_all=True)
                        b_cb = cb_dict["cb_bias"]
                    elif return_gc or return_cb:
                        b_cb, gc_act = self.cb(h_prev_teacher, x=x_t, return_gc=True)
                    else:
                        b_cb = self.cb(h_prev_teacher, x=x_t)

                else:
                    if return_dynamics:
                        cb_dict = self.cb(h_prev_teacher, return_all=True)
                        b_cb = cb_dict["cb_bias"]
                    elif return_gc or return_cb:
                        b_cb, gc_act = self.cb(h_prev_teacher, return_gc=True)
                    else:
                        b_cb = self.cb(h_prev_teacher)

                if self.multiply:
                    post_teacher = pre_teacher - b_cb
                else:
                    post_teacher = pre_teacher + b_cb
            else:
                b_cb = None
                post_teacher = pre_teacher

            if self.use_cb_bias and return_cb:
                if b_cb is not None:
                    cb_bias_seq.append(b_cb)
                if gc_act is not None:
                    gc_seq.append(gc_act)

            h_teacher = (1 - alpha) * h_prev_teacher + alpha * self.afunc(post_teacher)

            if return_dynamics:
                xproj_seq.append(x_proj[t])
                pre_seq.append(pre_teacher)
                post_seq.append(post_teacher)
                h_seq_full.append(h_teacher)

                if self.use_cb_bias:
                    if cb_dict is not None:
                        gc_input_seq.append(cb_dict["gc_input"])
                        gc_seq_full.append(cb_dict["gc"])
                        pc_seq.append(cb_dict["pc"])
                        dcn_seq.append(cb_dict["dcn"])
                        cb_bias_seq_full.append(cb_dict["cb_bias"])
                    else:
                        # should rarely happen, but keep shapes aligned
                        gc_input_seq.append(torch.zeros(B, self.cb.gc.in_features, device=device))
                        gc_seq_full.append(torch.zeros(B, self.cb.gc_dim, device=device))
                        pc_seq.append(torch.zeros(B, self.cb.pc_dim, device=device))
                        dcn_seq.append(torch.zeros(B, self.cb.dcn_dim, device=device))
                        cb_bias_seq_full.append(torch.zeros(B, self.hidden_size, device=device))

            if self.debug_stats:
                with torch.no_grad():
                # pre
                    pre_n = pre_teacher.norm(dim=-1)  # [B]
                    dbg["pre_mean_sum"] += pre_n.mean()
                    dbg["pre_max"] = torch.maximum(dbg["pre_max"], pre_n.max())
                    dbg["nonfinite_pre"] += (~torch.isfinite(pre_teacher)).any(dim=-1).sum()

                # cb (only if used)
                    if b_cb is not None:
                        cb_n = b_cb.norm(dim=-1)
                        dbg["cb_mean_sum"] += cb_n.mean()
                        dbg["cb_max"] = torch.maximum(dbg["cb_max"], cb_n.max())
                        dbg["nonfinite_cb"] += (~torch.isfinite(b_cb)).any(dim=-1).sum()

                # post
                    post_n = post_teacher.norm(dim=-1)
                    dbg["post_mean_sum"] += post_n.mean()
                    dbg["post_max"] = torch.maximum(dbg["post_max"], post_n.max())
                    dbg["nonfinite_post"] += (~torch.isfinite(post_teacher)).any(dim=-1).sum()

                # ht
                    ht_n = h_teacher.norm(dim=-1)
                    dbg["ht_mean_sum"] += ht_n.mean()
                    dbg["ht_max"] = torch.maximum(dbg["ht_max"], ht_n.max())
                    dbg["nonfinite_ht"] += (~torch.isfinite(h_teacher)).any(dim=-1).sum()

                    dbg["T"] += 1

            # Student trajectory (no CB)
            if self.rnn_eat and self.use_cb_bias:
                h_prev_student = h_student
                pre_student = x_proj[t] + self.hh(h_prev_student)
                h_student = (1 - alpha) * h_prev_student + alpha * self.afunc(pre_student)
                cb_diff_seq.append(h_teacher.detach() - h_student.detach())
                
                if self.rnn_eat_loss_type == 'hidden':
                    step_e = F.mse_loss(h_student, h_teacher.detach(), reduction="mean")
                    if not torch.isfinite(step_e):
                        step_e = torch.tensor(0.0, device=device)
                        eat_bad_count += 1
                    eat_loss_sum = step_e if eat_loss_sum is None else (eat_loss_sum + step_e)
                    eat_count += 1
                # task loss type: h_student accumulated, loss computed externally after loop

            if savetime:
                hs_t.append([h_teacher.clone()])
                if cb_t is not None:
                    cb_t.append([b_cb.clone() if b_cb is not None else None])

            if return_timewise:
                h_seq.append(h_teacher)

        # --- finalize debug (ONE sync per metric) ---
        if self.debug_stats:
            T_ = int(torch.clamp(dbg["T"], min=1).item())

            self._dbg = {
                "pre_norm_mean": float((dbg["pre_mean_sum"] / T_).item()),
                "pre_norm_max":  float(dbg["pre_max"].item()),
                "post_norm_mean": float((dbg["post_mean_sum"] / T_).item()),
                "post_norm_max":  float(dbg["post_max"].item()),
                "ht_norm_mean":  float((dbg["ht_mean_sum"] / T_).item()),
                "ht_norm_max":   float(dbg["ht_max"].item()),
                "cb_norm_mean":  float((dbg["cb_mean_sum"] / T_).item()) if self.use_cb_bias else 0.0,
                "cb_norm_max":   float(dbg["cb_max"].item()) if self.use_cb_bias else 0.0,

                "nonfinite_pre": int(dbg["nonfinite_pre"].item()),
                "nonfinite_post": int(dbg["nonfinite_post"].item()),
                "nonfinite_ht":  int(dbg["nonfinite_ht"].item()),
                "nonfinite_cb":  int(dbg["nonfinite_cb"].item()) if self.use_cb_bias else 0,
                "max_abs_logit": float(dbg["max_abs_logit"].item()),

                "T": T_,
            }

        # --- EAT LOSS SUMMARY ---
        if self.rnn_eat and self.use_cb_bias:
            if self.rnn_eat_loss_type == 'hidden' and eat_count > 0:
                eat_loss = eat_loss_sum / float(eat_count)
                self._last_eat_valid = torch.isfinite(eat_loss).item()
                self._last_eat_loss = eat_loss if self._last_eat_valid else torch.zeros((), device=device)
                self._last_h_student = None  # not needed for hidden type
            elif self.rnn_eat_loss_type == 'task':
                self._last_h_student = h_student  # pass to training loop for external loss
                self._last_eat_loss = torch.zeros((), device=device)
                self._last_eat_valid = True
            else:
                self._last_eat_loss = torch.zeros((), device=device)
                self._last_eat_valid = False
                self._last_h_student = None
        else:
            self._last_eat_loss = torch.zeros((), device=device)
            self._last_eat_valid = False
            self._last_h_student = None
            
        # Readout heads use teacher state
        if classify_in_time:
            raise NotImplementedError("classify_in_time=True not implemented in this compatibility forward.")

        if index_in_head is None:
            heads_to_use = list(self.heads)
        else:
            heads_to_use = [self.heads[index_in_head]]

        if return_timewise:
            h_seq_tensor = torch.stack(h_seq, dim=0)   # [T,B,H]
            out_class = [head(h_seq_tensor) for head in heads_to_use]
            max_abs_out = max([out.abs().max().item() for out in out_class])
            self._dbg["max_abs_logit"] = max_abs_out
        else:
            out_class = [head(h_teacher) for head in heads_to_use]
            max_abs_out = max([out.abs().max().item() for out in out_class])
            self._dbg["max_abs_logit"] = max_abs_out

        hs_out = [h_teacher]
        cb_out = [b_cb] if self.use_cb_bias else None

        self._last_cb_t = cb_t if savetime else None
        self._last_cb = cb_out
        self._last_cb_diff_seq = cb_diff_seq

        dynamics = None

        if return_dynamics:
            dynamics = {
                "hidden": torch.stack(h_seq_full, dim=0) if h_seq_full is not None else None,     # [T,B,H]
                "x_proj": torch.stack(xproj_seq, dim=0) if xproj_seq is not None else None,       # [T,B,H]
                "pre": torch.stack(pre_seq, dim=0) if pre_seq is not None else None,              # [T,B,H]
                "post": torch.stack(post_seq, dim=0) if post_seq is not None else None,           # [T,B,H]
            }

            if self.use_cb_bias:
                dynamics.update({
                    "gc_input": torch.stack(gc_input_seq, dim=0) if gc_input_seq is not None else None,
                    "gc": torch.stack(gc_seq_full, dim=0) if gc_seq_full is not None else None,
                    "pc": torch.stack(pc_seq, dim=0) if pc_seq is not None else None,
                    "dcn": torch.stack(dcn_seq, dim=0) if dcn_seq is not None else None,
                    "cb_bias": torch.stack(cb_bias_seq_full, dim=0) if cb_bias_seq_full is not None else None,
                })

        elif return_cb and self.use_cb_bias:
            dynamics = {
                "gc": torch.stack(gc_seq, dim=0) if (gc_seq is not None and len(gc_seq) > 0) else None,
                "cb_bias": torch.stack(cb_bias_seq, dim=0) if (cb_bias_seq is not None and len(cb_bias_seq) > 0) else None,
            }
        else:
            dynamics = None

        if savetime:
            if return_dynamics or return_cb:
                return hs_t, out_class, dynamics
            return hs_t, out_class

        if return_dynamics or return_cb:
            return hs_out, out_class, dynamics

        return hs_out, out_class

# ok so this is the model init function that i will call from train.py
def init_model(
    INPUT_SIZE=1,
    HIDDEN_SIZE=64,
    NUM_CLASSES=2,
    NUM_READOUT_HEADS=150,
    TAU=1.5,
    AFUNC=nn.ReLU,
    BIAS=True,
    USE_CB_BIAS=True,
    SCRAMBLE=False,
    TRAIN_TAU=False,
    CB_GC_DIM=512,
    CB_SPARSITY=0.0,
    DEVICE='cpu',
):
    rnn = ElmanRNNMultiHead(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_readout_heads=NUM_READOUT_HEADS,
        tau=TAU,
        afunc=AFUNC,
        bias=BIAS,
        use_cb_bias=USE_CB_BIAS,
        scramble=SCRAMBLE,
        train_tau=TRAIN_TAU,
        cb_gc_dim=CB_GC_DIM,
        cb_sparsity=CB_SPARSITY,
        device=DEVICE,
    ).to(DEVICE)
    return rnn