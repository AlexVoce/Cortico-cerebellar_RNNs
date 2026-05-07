import torch
import torch.nn as nn
import torch.nn.functional as F


def rescale_to_reference_norm(vec, ref_vec, max_ratio=1.0, eps=1e-8):
    """
    Rescale vec so that ||vec|| <= max_ratio * ||ref_vec|| for each batch item.

    vec:      [B, D] tensor to be rescaled
    ref_vec:  [B, D] reference tensor
    max_ratio: maximum allowed ||vec|| / ||ref_vec||
    """
    vec_norm = vec.norm(dim=-1, keepdim=True).clamp_min(eps)
    ref_norm = ref_vec.norm(dim=-1, keepdim=True).clamp_min(eps)

    max_allowed = max_ratio * ref_norm
    scale = torch.minimum(torch.ones_like(vec_norm), max_allowed / vec_norm)

    return vec * scale


class CB_bias(nn.Module):
    """
    Cerebellar-inspired feedforward bias module.
    Receives h_t, optionally concatenated with x_t, and returns a hidden-sized
    bias vector for the recurrent transition.
    Architecture:
        GC -> PC -> DCN
    """

    def __init__(
        self,
        hidden_size: int,
        gc_dim: int = 128,
        pc_dim: int = 64,
        input_size: int = 0,
        use_hidden: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.gc_dim = gc_dim
        self.pc_dim = pc_dim
        self.input_size = input_size
        self.use_hidden = use_hidden
        self.use_input = input_size > 0

        if not self.use_hidden and not self.use_input:
            raise ValueError(
                "CB_bias must receive at least one of hidden state or task input."
            )

        gc_input_size = 0
        if self.use_hidden:
            gc_input_size += hidden_size
        if self.use_input:
            gc_input_size += input_size

        self.gc = nn.Linear(gc_input_size, gc_dim, bias=True)
        self.pc = nn.Linear(gc_dim, pc_dim, bias=True)
        self.dcn = nn.Linear(pc_dim, hidden_size, bias=True)

        # Initialise CB near zero so the model starts close to the base RNN.
        nn.init.zeros_(self.gc.bias)
        nn.init.zeros_(self.pc.bias)
        nn.init.zeros_(self.dcn.bias)

        nn.init.normal_(self.gc.weight, std=0.01)
        nn.init.normal_(self.pc.weight, std=0.1)
        nn.init.normal_(self.dcn.weight, std=0.1)

        self.cb_parameters = (
            list(self.gc.parameters())
            + list(self.pc.parameters())
            + list(self.dcn.parameters())
        )

        self._hparams = dict(
            hidden_size=hidden_size,
            gc_dim=gc_dim,
            pc_dim=pc_dim,
            input_size=input_size,
            use_hidden=use_hidden,
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor = None,
        return_all: bool = False,
    ):
        if self.use_hidden and self.use_input:
            if x is None:
                raise ValueError(
                    "CB_bias was configured to use task input, but x is None."
                )
            gc_input = torch.cat([h, x], dim=-1)

        elif self.use_hidden:
            gc_input = h

        else:
            if x is None:
                raise ValueError(
                    "CB_bias was configured to use input only, but x is None."
                )
            gc_input = x

        g_t = F.relu(self.gc(gc_input), inplace=False)
        p_t = F.relu(self.pc(g_t), inplace=False)
        d_t = self.dcn(p_t)
        cb_bias = d_t

        if return_all:
            return {
                "gc_input": gc_input,
                "gc": g_t,
                "pc": p_t,
                "dcn": d_t,
                "cb_bias": cb_bias,
            }

        return cb_bias


class ElmanRNNMultiHead(nn.Module):
    """
    Elman RNN with an optional cerebellar-inspired feedforward bias module.

    Recurrent update:

        h_{t+1} = (1 - alpha) h_t + alpha f(W_x x_t + W_h h_t + b + b_cb)

    where b_cb is produced by a GC-PC-DCN feedforward module from h_t and,
    optionally, the current input x_t.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2,
        num_readout_heads: int = 1,
        tau: float = 1.5,
        afunc=nn.LeakyReLU,
        bias: bool = True,
        use_cb_bias: bool = True,
        cb_gc_dim: int = 128,
        cb_pc_dim: int = 64,
        cb_no_hidden: bool = False,
        cb_input_size: int = 0,
        cb_max_ratio: float = 1.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_readout_heads = num_readout_heads
        self.tau = float(tau)
        self.afunc = afunc()

        self.use_cb_bias = use_cb_bias
        self.cb_no_hidden = cb_no_hidden
        self.cb_input_size = cb_input_size
        self.cb_max_ratio = cb_max_ratio

        # RNN core
        self.inp = nn.Linear(input_size, hidden_size, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=bias)

        # One readout head per task/curriculum level, depending on training setup.
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_classes) for _ in range(num_readout_heads)]
        )

        # Cerebellar bias module
        if self.use_cb_bias:
            if self.cb_no_hidden and cb_input_size <= 0:
                raise ValueError(
                    "cb_no_hidden=True requires cb_input_size > 0 so CB can receive task input."
                )

            self.cb = CB_bias(
                hidden_size=hidden_size,
                gc_dim=cb_gc_dim,
                pc_dim=cb_pc_dim,
                input_size=cb_input_size,
                use_hidden=not cb_no_hidden,
            )
        else:
            self.cb = None

        self._hparams = dict(
            model_type="ElmanRNNMultiHead",
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_readout_heads=num_readout_heads,
            tau=tau,
            bias=bias,
            use_cb_bias=use_cb_bias,
            cb_gc_dim=cb_gc_dim if use_cb_bias else None,
            cb_pc_dim=cb_pc_dim if use_cb_bias else None,
            cb_no_hidden=cb_no_hidden if use_cb_bias else None,
            cb_input_size=cb_input_size if use_cb_bias else None,
            cb_max_ratio=cb_max_ratio if use_cb_bias else None,
            activation=afunc.__name__,
        )

    def forward(
        self,
        data: torch.Tensor,
        hs=None,
        index_in_head=None,
        return_timewise: bool = False,
        task_id: int = None,
        return_cb: bool = False,
        return_dynamics: bool = False,
    ):
        """
        Parameters
        ----------
        data:
            Input tensor with shape [T, B, input_size].
        hs:
            Optional initial hidden state. Can be tensor [B, H] or list containing it.
        index_in_head:
            If None, all readout heads are returned. Otherwise, only the selected head is used.
        return_timewise:
            If True, readouts are returned for every timestep as [T, B, C].
            If False, readouts are returned only from the final hidden state as [B, C].
        task_id:
            Optional task index. If provided and cb_input_size > 0, CB receives a one-hot task vector
            instead of the raw task input.
        return_cb:
            If True, returns GC activity and CB bias over time.
        return_dynamics:
            If True, returns full recurrent and CB dynamics over time.
        """
        T, B, _ = data.shape
        device = data.device
        alpha = 1.0 / self.tau

        if hs is None:
            h = 0.1 * torch.rand(B, self.hidden_size, device=device)
        else:
            h = hs[0] if isinstance(hs, list) else hs

        x_proj = self.inp(data)

        # Optional collectors
        h_seq = [] if return_timewise else None

        h_seq_full = [] if return_dynamics else None
        pre_seq = [] if return_dynamics else None
        post_seq = [] if return_dynamics else None
        xproj_seq = [] if return_dynamics else None

        collect_cb = self.use_cb_bias and (return_cb or return_dynamics)

        gc_input_seq = [] if (collect_cb and return_dynamics) else None
        gc_seq = [] if collect_cb else None
        pc_seq = [] if (collect_cb and return_dynamics) else None
        dcn_seq = [] if (collect_cb and return_dynamics) else None
        cb_bias_seq = [] if collect_cb else None

        for t in range(T):
            h_prev = h

            pre = x_proj[t] + self.hh(h_prev)

            if self.use_cb_bias:
                if task_id is not None and self.cb_input_size > 0:
                    cb_input = torch.zeros(B, self.cb_input_size, device=device)
                    cb_input[:, task_id] = 1.0

                    if collect_cb:
                        cb_dict = self.cb(h_prev, x=cb_input, return_all=True)
                        b_cb = cb_dict["cb_bias"]
                    else:
                        cb_dict = None
                        b_cb = self.cb(h_prev, x=cb_input, return_all=False)

                    # rescaling not used in original code, only used with task id input which isnt in paper
                    b_cb = rescale_to_reference_norm(
                        vec=b_cb,
                        ref_vec=pre,
                        max_ratio=self.cb_max_ratio,
                    )

                    if cb_dict is not None:
                        cb_dict["cb_bias"] = b_cb

                elif self.cb_input_size > 0:
                    cb_input = data[t]

                    if collect_cb:
                        cb_dict = self.cb(h_prev, x=cb_input, return_all=True)
                        b_cb = cb_dict["cb_bias"]
                    else:
                        cb_dict = None
                        b_cb = self.cb(h_prev, x=cb_input, return_all=False)

                    # No rescaling here, matching original code.

                else:
                    cb_input = None

                    if collect_cb:
                        cb_dict = self.cb(h_prev, x=None, return_all=True)
                        b_cb = cb_dict["cb_bias"]
                    else:
                        cb_dict = None
                        b_cb = self.cb(h_prev, x=None, return_all=False)

                    # No rescaling here, matching original code.

                post = pre + b_cb

            else:
                cb_dict = None
                b_cb = None
                post = pre

            h = (1.0 - alpha) * h_prev + alpha * self.afunc(post)

            if return_timewise:
                h_seq.append(h)

            if return_dynamics:
                xproj_seq.append(x_proj[t])
                pre_seq.append(pre)
                post_seq.append(post)
                h_seq_full.append(h)

                if self.use_cb_bias:
                    gc_input_seq.append(cb_dict["gc_input"])
                    gc_seq.append(cb_dict["gc"])
                    pc_seq.append(cb_dict["pc"])
                    dcn_seq.append(cb_dict["dcn"])
                    cb_bias_seq.append(cb_dict["cb_bias"])

            elif return_cb and self.use_cb_bias:
                gc_seq.append(cb_dict["gc"])
                cb_bias_seq.append(cb_dict["cb_bias"])

        if index_in_head is None:
            heads_to_use = list(self.heads)
        else:
            heads_to_use = [self.heads[index_in_head]]

        if return_timewise:
            h_seq_tensor = torch.stack(h_seq, dim=0)
            out_class = [head(h_seq_tensor) for head in heads_to_use]
        else:
            out_class = [head(h) for head in heads_to_use]

        hs_out = [h]

        if return_dynamics:
            dynamics = {
                "hidden": torch.stack(h_seq_full, dim=0),
                "x_proj": torch.stack(xproj_seq, dim=0),
                "pre": torch.stack(pre_seq, dim=0),
                "post": torch.stack(post_seq, dim=0),
            }

            if self.use_cb_bias:
                dynamics.update(
                    {
                        "gc_input": torch.stack(gc_input_seq, dim=0),
                        "gc": torch.stack(gc_seq, dim=0),
                        "pc": torch.stack(pc_seq, dim=0),
                        "dcn": torch.stack(dcn_seq, dim=0),
                        "cb_bias": torch.stack(cb_bias_seq, dim=0),
                    }
                )

            return hs_out, out_class, dynamics

        if return_cb and self.use_cb_bias:
            dynamics = {
                "gc": torch.stack(gc_seq, dim=0),
                "cb_bias": torch.stack(cb_bias_seq, dim=0),
            }

            return hs_out, out_class, dynamics

        return hs_out, out_class