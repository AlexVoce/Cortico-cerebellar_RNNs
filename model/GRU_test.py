"""
GRU variant of the cortico-cerebellar RNN model.

Implements GRUMultiHeadWithCB, a multi-head GRU with an optional
cerebellar-inspired feedforward bias module. The CB_bias class is shared
with the Elman model and imported from model.models_cb.
"""

import torch
import torch.nn as nn

from model.models_cb import CB_bias, rescale_to_reference_norm


class GRUCellWithOptionalCB(nn.Module):
    """
    GRU cell where the CB bias is injected into the candidate hidden-state
    preactivation, not into the update or reset gates.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        candidate_afunc=nn.LeakyReLU,
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

    def candidate_preactivation(self, x_t, h_prev):
        """
        Candidate hidden-state preactivation before CB injection.
        """
        r_t = torch.sigmoid(self.x_r(x_t) + self.h_r(h_prev))
        candidate_pre = self.x_n(x_t) + self.h_n(r_t * h_prev)
        return candidate_pre

    def forward(self, x_t, h_prev, cb_bias=None):
        """
        Parameters
        ----------
        x_t:
            Input at one timestep, shape [B, input_size].
        h_prev:
            Previous hidden state, shape [B, hidden_size].
        cb_bias:
            Optional CB bias, shape [B, hidden_size].
        """
        z_t = torch.sigmoid(self.x_z(x_t) + self.h_z(h_prev))
        r_t = torch.sigmoid(self.x_r(x_t) + self.h_r(h_prev))

        candidate_pre = self.x_n(x_t) + self.h_n(r_t * h_prev)

        if cb_bias is not None:
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
    Multi-head GRU model with an optional cerebellar-inspired feedforward bias.

    The CB module receives h_t and optionally x_t/task_id, then injects a
    hidden-sized bias into the GRU candidate preactivation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2,
        num_readout_heads: int = 1,
        candidate_afunc=nn.LeakyReLU,
        bias: bool = True,
        use_cb_bias: bool = True,
        cb_gc_dim: int = 128,
        cb_pc_dim: int = 64,
        cb_input_size: int = 0,
        cb_no_hidden: bool = False,
        cb_max_ratio: float = 1.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_readout_heads = num_readout_heads

        self.use_cb_bias = use_cb_bias
        self.cb_input_size = cb_input_size
        self.cb_no_hidden = cb_no_hidden
        self.cb_max_ratio = cb_max_ratio

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

        if isinstance(candidate_afunc, nn.Module):
            afunc_name = candidate_afunc.__class__.__name__
        else:
            afunc_name = candidate_afunc.__name__

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
            cb_input_size=cb_input_size if use_cb_bias else None,
            cb_no_hidden=cb_no_hidden if use_cb_bias else None,
            cb_max_ratio=cb_max_ratio if use_cb_bias else None,
        )

    def _get_cb_input(self, x_t, h_prev, task_id, B):
        """
        Select CB input according to the configured input mode.
        """
        if self.cb_input_size <= 0:
            return None

        if task_id is not None:
            cb_x = torch.zeros(B, self.cb_input_size, device=h_prev.device)
            cb_x[:, task_id] = 1.0
            return cb_x

        return x_t

    def _get_cb_bias(
        self,
        h_prev,
        x_t,
        task_id,
        B,
        return_all: bool = False,
    ):
        """
        Compute CB bias and optional CB dynamics.
        """
        if not self.use_cb_bias:
            return None, None

        cb_x = self._get_cb_input(
            x_t=x_t,
            h_prev=h_prev,
            task_id=task_id,
            B=B,
        )

        if return_all:
            cb_dict = self.cb(h_prev, x=cb_x, return_all=True)
            b_cb = cb_dict["cb_bias"]
            return b_cb, cb_dict

        b_cb = self.cb(h_prev, x=cb_x, return_all=False)
        return b_cb, None

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
            instead of the raw input.
        return_cb:
            If True, returns GC activity and CB bias over time.
        return_dynamics:
            If True, returns full GRU and CB dynamics over time.
        """
        T, B, _ = data.shape
        device = data.device

        if hs is None:
            h = 0.1 * torch.rand(B, self.hidden_size, device=device)
        else:
            h = hs[0] if isinstance(hs, list) else hs

        h_seq = [] if return_timewise else None

        h_seq_full = [] if return_dynamics else None
        z_seq = [] if return_dynamics else None
        r_seq = [] if return_dynamics else None
        candidate_pre_seq = [] if return_dynamics else None
        candidate_seq = [] if return_dynamics else None

        collect_cb = self.use_cb_bias and (return_cb or return_dynamics)

        gc_input_seq = [] if (collect_cb and return_dynamics) else None
        gc_seq = [] if collect_cb else None
        pc_seq = [] if (collect_cb and return_dynamics) else None
        dcn_seq = [] if (collect_cb and return_dynamics) else None
        cb_bias_seq = [] if collect_cb else None

        for t in range(T):
            x_t = data[t]
            h_prev = h

            if self.use_cb_bias:
                b_cb, cb_dict = self._get_cb_bias(
                    h_prev=h_prev,
                    x_t=x_t,
                    task_id=task_id,
                    B=B,
                    return_all=collect_cb,
                )

                # Match the existing GRU implementation:
                # CB bias is rescaled whenever CB is active.
                with torch.no_grad():
                    candidate_pre_ref = self.gru_cell.candidate_preactivation(
                        x_t=x_t,
                        h_prev=h_prev,
                    )

                b_cb = rescale_to_reference_norm(
                    vec=b_cb,
                    ref_vec=candidate_pre_ref,
                    max_ratio=self.cb_max_ratio,
                )

                if cb_dict is not None:
                    cb_dict["cb_bias"] = b_cb

            else:
                b_cb = None
                cb_dict = None

            h, gru_dict = self.gru_cell(
                x_t=x_t,
                h_prev=h_prev,
                cb_bias=b_cb,
            )

            if return_timewise:
                h_seq.append(h)

            if return_dynamics:
                h_seq_full.append(h)
                z_seq.append(gru_dict["z"])
                r_seq.append(gru_dict["r"])
                candidate_pre_seq.append(gru_dict["candidate_pre"])
                candidate_seq.append(gru_dict["candidate"])

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
                "z_gate": torch.stack(z_seq, dim=0),
                "r_gate": torch.stack(r_seq, dim=0),
                "candidate_pre": torch.stack(candidate_pre_seq, dim=0),
                "candidate": torch.stack(candidate_seq, dim=0),
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
    CB_INPUT_SIZE=0,
    CB_NO_HIDDEN=False,
    CB_MAX_RATIO=1.0,
    DEVICE="cpu",
    **kwargs,
):
    """
    Initialise a GRU model with optional CB bias.
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
        cb_input_size=CB_INPUT_SIZE,
        cb_no_hidden=CB_NO_HIDDEN,
        cb_max_ratio=CB_MAX_RATIO,
    ).to(DEVICE)

    return model