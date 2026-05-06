import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models_cb import rescale_to_reference_norm

class CB_bias(nn.Module):
    """
    Cerebellar-inspired feedforward bias module.

    Receives h_t, and optionally x_t or a task one-hot vector, then returns
    a hidden-sized bias vector for the recurrent transition.
    """

    def __init__(
        self,
        hidden_size: int,
        gc_dim: int = 128,
        pc_dim: int = 64,
        dcn_dim: int = 64,
        input_size: int = 0,
        use_hidden: bool = True,
        sparsity: float = 0.0,
        scramble: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.gc_dim = gc_dim
        self.pc_dim = pc_dim
        self.dcn_dim = dcn_dim
        self.input_size = input_size
        self.sparsity = sparsity
        self.scramble = scramble
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
        else:
            self.dcn_proj = None

        # Start CB small so it does not dominate early recurrent dynamics.
        nn.init.zeros_(self.gc.bias)
        nn.init.zeros_(self.pc.bias)
        nn.init.zeros_(self.dcn.bias)

        nn.init.normal_(self.gc.weight, std=0.01)
        nn.init.normal_(self.pc.weight, std=0.1)
        nn.init.normal_(self.dcn.weight, std=0.1)

        if self.dcn_proj is not None:
            nn.init.zeros_(self.dcn_proj.bias)
            nn.init.normal_(self.dcn_proj.weight, std=0.1)

        self.cb_parameters = (
            list(self.gc.parameters())
            + list(self.pc.parameters())
            + list(self.dcn.parameters())
            + (list(self.dcn_proj.parameters()) if self.dcn_proj is not None else [])
        )

        self._hparams = dict(
            module_type="CB_bias",
            hidden_size=hidden_size,
            gc_dim=gc_dim,
            pc_dim=pc_dim,
            dcn_dim=dcn_dim,
            input_size=input_size,
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

        g_t = F.relu(self.gc(gc_input), inplace=False)

        # Optional top-k sparsity, currently off by default.
        if self.sparsity > 0.0 and self.k > 0:
            values, indices = torch.topk(g_t, k=self.k, dim=-1)
            sparse_g = torch.zeros_like(g_t)
            sparse_g.scatter_(dim=-1, index=indices, src=values)
            g_t = sparse_g

        self._last_gc = g_t

        p_t = F.relu(self.pc(g_t), inplace=False)
        d_t = self.dcn(p_t)

        if self.dcn_proj is not None:
            bias = self.dcn_proj(d_t)
        else:
            bias = d_t

        if scramble or self.scramble:
            bias = bias + torch.randn_like(bias) * 0.01

        if return_all:
            return {
                "gc_input": gc_input,
                "gc": g_t,
                "pc": p_t,
                "dcn": d_t,
                "cb_bias": bias,
            }

        if return_gc:
            return bias, g_t

        return bias


class GRUCellWithOptionalCB(nn.Module):
    """
    GRU cell where CB bias is injected into the candidate hidden-state
    preactivation, not into the update or reset gates.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        candidate_afunc=nn.Tanh,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.h_z = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x_r = nn.Linear(input_size, hidden_size, bias=bias)
        self.h_r = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x_n = nn.Linear(input_size, hidden_size, bias=bias)
        self.h_n = nn.Linear(hidden_size, hidden_size, bias=bias)

        if isinstance(candidate_afunc, nn.Module):
            self.candidate_afunc = candidate_afunc
            afunc_name = candidate_afunc.__class__.__name__
        else:
            self.candidate_afunc = candidate_afunc()
            afunc_name = candidate_afunc.__name__

        self._hparams = dict(
            cell_type="GRUCellWithOptionalCB",
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            candidate_afunc=afunc_name,
        )

    def forward(self, x_t, h_prev, cb_bias=None, multiply=False):
        """
        x_t:      [B, input_size]
        h_prev:  [B, hidden_size]
        cb_bias: [B, hidden_size] or None
        """

        z_t = torch.sigmoid(self.x_z(x_t) + self.h_z(h_prev))
        r_t = torch.sigmoid(self.x_r(x_t) + self.h_r(h_prev))

        candidate_pre = self.x_n(x_t) + self.h_n(r_t * h_prev)

        if cb_bias is not None:
            if multiply:
                # Kept for compatibility with your existing flag.
                candidate_pre = candidate_pre - cb_bias
            else:
                candidate_pre = candidate_pre + cb_bias

        candidate = self.candidate_afunc(candidate_pre)

        h_new = (1.0 - z_t) * h_prev + z_t * candidate

        return h_new, {
            "z": z_t,
            "r": r_t,
            "candidate_pre": candidate_pre,
            "candidate": candidate,
        }


class GRUMultiHeadWithCB(nn.Module):
    """
    Multi-head GRU model with optional cerebellar-inspired feedforward bias.

    Forward return format mirrors your ElmanRNNMultiHead:
        hs_out, out_class
        or
        hs_out, out_class, dynamics
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2,
        num_readout_heads: int = 1,
        candidate_afunc=nn.Tanh,
        bias: bool = True,
        use_cb_bias: bool = True,
        cb_gc_dim: int = 128,
        cb_pc_dim: int = 64,
        cb_dcn_dim: int = 64,
        cb_sparsity: float = 0.0,
        multiply: bool = False,
        cb_input_size: int = 0,
        cb_no_hidden: bool = False,
        cb_max_ratio: float = 1.0,
        debug_stats: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_readout_heads = num_readout_heads

        self.use_cb_bias = use_cb_bias
        self.multiply = multiply
        self.cb_input_size = cb_input_size
        self.cb_no_hidden = cb_no_hidden
        self.cb_max_ratio = cb_max_ratio
        self.debug_stats = debug_stats

        self.gru_cell = GRUCellWithOptionalCB(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            candidate_afunc=candidate_afunc,
        )

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_classes) for _ in range(num_readout_heads)]
        )

        if self.use_cb_bias:
            if self.cb_no_hidden and cb_input_size <= 0:
                raise ValueError("cb_no_hidden=True requires cb_input_size > 0 so CB can receive task input")
            self.cb = CB_bias(
                hidden_size=hidden_size,
                gc_dim=cb_gc_dim,
                pc_dim=cb_pc_dim,
                dcn_dim=cb_dcn_dim,
                input_size=cb_input_size,
                use_hidden=not cb_no_hidden,
                sparsity=cb_sparsity,
                scramble=False,
                device=device,
            )
        else:
            self.cb = None

        if isinstance(candidate_afunc, nn.Module):
            afunc_name = candidate_afunc.__class__.__name__
        else:
            afunc_name = candidate_afunc.__name__

        self._dbg = {}

        self._hparams = dict(
            model_type="GRUMultiHeadWithCB",
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_readout_heads=num_readout_heads,
            candidate_afunc=afunc_name,
            bias=bias,
            use_cb_bias=use_cb_bias,
            cb_gc_dim=cb_gc_dim if use_cb_bias else None,
            cb_pc_dim=cb_pc_dim if use_cb_bias else None,
            cb_dcn_dim=cb_dcn_dim if use_cb_bias else None,
            cb_sparsity=cb_sparsity if use_cb_bias else None,
            multiply=multiply if use_cb_bias else None,
            cb_input_size=cb_input_size if use_cb_bias else None,
            cb_no_hidden=cb_no_hidden if use_cb_bias else None,
            cb_max_ratio=cb_max_ratio if use_cb_bias else None,
            debug_stats=debug_stats,
        )

    def _get_cb_bias(
        self,
        h_prev,
        x_t=None,
        task_id=None,
        B=None,
        return_dynamics=False,
        return_gc=False,
    ):
        if not self.use_cb_bias:
            return None, None, None

        if self.cb_input_size > 0:
            if task_id is not None:
                cb_x = torch.zeros(B, self.cb_input_size, device=h_prev.device)
                cb_x[:, task_id] = 1.0
            else:
                cb_x = x_t
        else:
            cb_x = None

        if return_dynamics:
            if cb_x is not None:
                cb_dict = self.cb(h_prev, x=cb_x, return_all=True)
            else:
                cb_dict = self.cb(h_prev, return_all=True)

            b_cb = cb_dict["cb_bias"]
            gc_act = cb_dict["gc"]

        elif return_gc:
            if cb_x is not None:
                b_cb, gc_act = self.cb(h_prev, x=cb_x, return_gc=True)
            else:
                b_cb, gc_act = self.cb(h_prev, return_gc=True)

            cb_dict = None

        else:
            if cb_x is not None:
                b_cb = self.cb(h_prev, x=cb_x)
            else:
                b_cb = self.cb(h_prev)

            gc_act = None
            cb_dict = None

        return b_cb, cb_dict, gc_act

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
        if classify_in_time:
            raise NotImplementedError("classify_in_time=True is not implemented.")

        T, B, _ = data.shape
        device = data.device

        if hs is None:
            h = 0.1 * torch.rand(B, self.hidden_size, device=device)
        else:
            h = hs[0] if isinstance(hs, list) else hs

        h_seq = [] if return_timewise else None
        hs_t = [] if savetime else None

        h_seq_full = [] if return_dynamics else None
        z_seq = [] if return_dynamics else None
        r_seq = [] if return_dynamics else None
        candidate_pre_seq = [] if return_dynamics else None
        candidate_seq = [] if return_dynamics else None

        gc_input_seq = [] if (self.use_cb_bias and return_dynamics) else None
        gc_seq_full = [] if (self.use_cb_bias and return_dynamics) else None
        pc_seq = [] if (self.use_cb_bias and return_dynamics) else None
        dcn_seq = [] if (self.use_cb_bias and return_dynamics) else None
        cb_bias_seq_full = [] if (self.use_cb_bias and return_dynamics) else None

        cb_bias_seq = [] if (self.use_cb_bias and return_cb) else None
        gc_seq = [] if (self.use_cb_bias and return_gc and return_cb) else None

        dbg = None
        if self.debug_stats:
            dbg = {
                "cb_norm_sum": torch.zeros((), device=device),
                "cb_norm_max": torch.zeros((), device=device),
                "h_norm_sum": torch.zeros((), device=device),
                "h_norm_max": torch.zeros((), device=device),
                "candidate_pre_norm_sum": torch.zeros((), device=device),
                "candidate_pre_norm_max": torch.zeros((), device=device),
                "nonfinite_cb": torch.zeros((), device=device),
                "nonfinite_h": torch.zeros((), device=device),
                "nonfinite_candidate_pre": torch.zeros((), device=device),
                "T": torch.zeros((), device=device),
            }

        for t in range(T):
            x_t = data[t]
            h_prev = h

            b_cb, cb_dict, gc_act = self._get_cb_bias(
                h_prev=h_prev,
                x_t=x_t,
                task_id=task_id,
                B=B,
                return_dynamics=return_dynamics,
                return_gc=return_gc,
            )

            # Reference norm is the candidate preactivation before CB injection.
            if b_cb is not None:
                with torch.no_grad():
                    r_ref = torch.sigmoid(
                        self.gru_cell.x_r(x_t) + self.gru_cell.h_r(h_prev)
                    )
                    candidate_pre_ref = (
                        self.gru_cell.x_n(x_t)
                        + self.gru_cell.h_n(r_ref * h_prev)
                    )

                b_cb = rescale_to_reference_norm(
                    vec=b_cb,
                    ref_vec=candidate_pre_ref,
                    max_ratio=self.cb_max_ratio,
                )

                if cb_dict is not None:
                    cb_dict["cb_bias"] = b_cb

            h, gru_dict = self.gru_cell(
                x_t=x_t,
                h_prev=h_prev,
                cb_bias=b_cb,
                multiply=self.multiply,
            )

            if return_timewise:
                h_seq.append(h)

            if savetime:
                hs_t.append([h.clone()])

            if self.use_cb_bias and return_cb:
                cb_bias_seq.append(b_cb)
                if return_gc:
                    if gc_act is not None:
                        gc_seq.append(gc_act)
                    elif cb_dict is not None:
                        gc_seq.append(cb_dict["gc"])

            if return_dynamics:
                h_seq_full.append(h)
                z_seq.append(gru_dict["z"])
                r_seq.append(gru_dict["r"])
                candidate_pre_seq.append(gru_dict["candidate_pre"])
                candidate_seq.append(gru_dict["candidate"])

                if self.use_cb_bias:
                    if cb_dict is not None:
                        gc_input_seq.append(cb_dict["gc_input"])
                        gc_seq_full.append(cb_dict["gc"])
                        pc_seq.append(cb_dict["pc"])
                        dcn_seq.append(cb_dict["dcn"])
                        cb_bias_seq_full.append(cb_dict["cb_bias"])
                    else:
                        gc_input_seq.append(torch.zeros(B, self.cb.gc.in_features, device=device))
                        gc_seq_full.append(torch.zeros(B, self.cb.gc_dim, device=device))
                        pc_seq.append(torch.zeros(B, self.cb.pc_dim, device=device))
                        dcn_seq.append(torch.zeros(B, self.cb.dcn_dim, device=device))
                        cb_bias_seq_full.append(torch.zeros(B, self.hidden_size, device=device))

            if self.debug_stats:
                with torch.no_grad():
                    dbg["h_norm_sum"] += h.norm(dim=-1).mean()
                    dbg["h_norm_max"] = torch.maximum(
                        dbg["h_norm_max"], h.norm(dim=-1).max()
                    )
                    dbg["candidate_pre_norm_sum"] += gru_dict["candidate_pre"].norm(dim=-1).mean()
                    dbg["candidate_pre_norm_max"] = torch.maximum(
                        dbg["candidate_pre_norm_max"],
                        gru_dict["candidate_pre"].norm(dim=-1).max(),
                    )
                    dbg["nonfinite_h"] += (~torch.isfinite(h)).any(dim=-1).sum()
                    dbg["nonfinite_candidate_pre"] += (
                        ~torch.isfinite(gru_dict["candidate_pre"])
                    ).any(dim=-1).sum()

                    if b_cb is not None:
                        dbg["cb_norm_sum"] += b_cb.norm(dim=-1).mean()
                        dbg["cb_norm_max"] = torch.maximum(
                            dbg["cb_norm_max"], b_cb.norm(dim=-1).max()
                        )
                        dbg["nonfinite_cb"] += (~torch.isfinite(b_cb)).any(dim=-1).sum()

                    dbg["T"] += 1

        if index_in_head is None:
            heads_to_use = list(self.heads)
        else:
            heads_to_use = [self.heads[index_in_head]]

        if return_timewise:
            h_seq_tensor = torch.stack(h_seq, dim=0)
            out_class = [head(h_seq_tensor) for head in heads_to_use]
        else:
            out_class = [head(h) for head in heads_to_use]

        max_abs_out = max([out.abs().max().item() for out in out_class])

        self._dbg = {"max_abs_logit": max_abs_out}

        if self.debug_stats and dbg is not None:
            T_ = int(torch.clamp(dbg["T"], min=1).item())

            self._dbg.update({
                "h_norm_mean": float((dbg["h_norm_sum"] / T_).item()),
                "h_norm_max": float(dbg["h_norm_max"].item()),
                "candidate_pre_norm_mean": float((dbg["candidate_pre_norm_sum"] / T_).item()),
                "candidate_pre_norm_max": float(dbg["candidate_pre_norm_max"].item()),
                "cb_norm_mean": float((dbg["cb_norm_sum"] / T_).item()) if self.use_cb_bias else 0.0,
                "cb_norm_max": float(dbg["cb_norm_max"].item()) if self.use_cb_bias else 0.0,
                "nonfinite_h": int(dbg["nonfinite_h"].item()),
                "nonfinite_candidate_pre": int(dbg["nonfinite_candidate_pre"].item()),
                "nonfinite_cb": int(dbg["nonfinite_cb"].item()) if self.use_cb_bias else 0,
                "T": T_,
            })

        hs_out = [h]

        dynamics = None

        if return_dynamics:
            dynamics = {
                "hidden": torch.stack(h_seq_full, dim=0),
                "z_gate": torch.stack(z_seq, dim=0),
                "r_gate": torch.stack(r_seq, dim=0),
                "candidate_pre": torch.stack(candidate_pre_seq, dim=0),
                "candidate": torch.stack(candidate_seq, dim=0),
            }

            if self.use_cb_bias:
                dynamics.update({
                    "gc_input": torch.stack(gc_input_seq, dim=0),
                    "gc": torch.stack(gc_seq_full, dim=0),
                    "pc": torch.stack(pc_seq, dim=0),
                    "dcn": torch.stack(dcn_seq, dim=0),
                    "cb_bias": torch.stack(cb_bias_seq_full, dim=0),
                })

        elif return_cb and self.use_cb_bias:
            dynamics = {
                "cb_bias": (
                    torch.stack(cb_bias_seq, dim=0)
                    if cb_bias_seq is not None and len(cb_bias_seq) > 0
                    else None
                ),
                "gc": (
                    torch.stack(gc_seq, dim=0)
                    if gc_seq is not None and len(gc_seq) > 0
                    else None
                ),
            }

        self._last_cb = [b_cb] if self.use_cb_bias else None

        if savetime:
            if return_dynamics or return_cb:
                return hs_t, out_class, dynamics
            return hs_t, out_class

        if return_dynamics or return_cb:
            return hs_out, out_class, dynamics

        return hs_out, out_class


def init_model(
    INPUT_SIZE=1,
    HIDDEN_SIZE=64,
    NUM_CLASSES=2,
    NUM_READOUT_HEADS=150,
    AFUNC=nn.Tanh,
    BIAS=True,
    USE_CB_BIAS=True,
    CB_GC_DIM=512,
    CB_PC_DIM=64,
    CB_DCN_DIM=64,
    CB_SPARSITY=0.0,
    CB_INPUT_SIZE=0,
    CB_MAX_RATIO=1.0,
    MULTIPLY=False,
    DEBUG_STATS=False,
    DEVICE="cpu",
    **kwargs,
):
    """
    Init function designed to be called from train.py.

    Accepts **kwargs so old training scripts passing unused Elman arguments
    such as TAU, TRAIN_TAU, or SCRAMBLE will not crash.
    """

    model = GRUMultiHeadWithCB(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_readout_heads=NUM_READOUT_HEADS,
        candidate_afunc=AFUNC,
        bias=BIAS,
        use_cb_bias=USE_CB_BIAS,
        cb_gc_dim=CB_GC_DIM,
        cb_pc_dim=CB_PC_DIM,
        cb_dcn_dim=CB_DCN_DIM,
        cb_sparsity=CB_SPARSITY,
        multiply=MULTIPLY,
        cb_input_size=CB_INPUT_SIZE,
        cb_max_ratio=CB_MAX_RATIO,
        debug_stats=DEBUG_STATS,
        device=DEVICE,
    ).to(DEVICE)

    return model