# training/variants.py

from training.base import compute_active_set, train_steps, evaluate


def _is_reservoir_refresh_n(next_n: int, anchor_n: int, interval: int) -> bool:
    """
    Return True when the reservoir/interleaved-reservoir schedule should
    briefly re-enable recurrent plasticity at this curriculum level.
    """
    if interval <= 0:
        raise ValueError("reservoir_interval must be > 0.")

    if next_n <= anchor_n:
        return False

    return (next_n - anchor_n) % interval == 0


def variant_cb_only_reservoir(
    *,
    model,
    task_fn,
    active_Ns,
    current_N,
    Ns_init,
    batch_size,
    training_steps_n,
    test_steps,
    device,
    criterion,
    head_idx,
    optimizer,
    get_grad_norms,
    global_epoch,
    set_active_module,
    readout_head_dyn,
    n_heads,
    n_forget,
    MAX_STAGE_EPOCHS,
    MAX_GLOBAL_EPOCHS,
    shared_optimiser,
    spec,
    cb_l2=0.0,
    learning_alg="bptt",
    reservoir_interval=10,
    reservoir_anchor_n=None,
    next_N_override=None,
    on_log=None,
    **kwargs,
):
    """
    Reservoir/interleaved-reservoir curriculum stage.

    Behaviour
    ---------
    - The base phase is handled outside this function.
    - At most curriculum levels, only the CB module and readout heads are trained.
    - At reservoir refresh levels, recurrent plasticity is briefly re-enabled.

    Example
    -------
    If Ns_init[0] = 2 and reservoir_interval = 10:

        N=3..11  -> CB_ONLY
        N=12     -> BOTH
        N=13..21 -> CB_ONLY
        N=22     -> BOTH

    Returns
    -------
    success, new_active_Ns, new_current_N, logs, updated_global_epoch
    """
    logs = []

    next_N = int(next_N_override) if next_N_override is not None else int(current_N + 1)
    reservoir_anchor_n = int(Ns_init[0]) if reservoir_anchor_n is None else int(reservoir_anchor_n)

    active_set = sorted(
        compute_active_set(
            active_Ns,
            next_N,
            readout_head_dyn,
            n_heads,
            n_forget,
        )
    )

    if _is_reservoir_refresh_n(
        next_n=next_N,
        anchor_n=reservoir_anchor_n,
        interval=int(reservoir_interval),
    ):
        phase = "BOTH"
        print(
            f"[reservoir] refresh N={next_N} "
            f"(anchor={reservoir_anchor_n}, interval={reservoir_interval}) -> BOTH",
            flush=True,
        )
    else:
        phase = "CB_ONLY"

    set_active_module(model, phase)
    solved = False

    for _ep in range(MAX_STAGE_EPOCHS):
        global_epoch += 1

        if global_epoch >= MAX_GLOBAL_EPOCHS:
            print(f"Reached MAX_GLOBAL_EPOCHS during {phase} phase.", flush=True)
            break

        summ = train_steps(
            model=model,
            task_fn=task_fn,
            active_Ns=active_set,
            batch_size=batch_size,
            phase=phase,
            device=device,
            criterion=criterion,
            head_idx=head_idx,
            optimizer=optimizer,
            get_grad_norms=get_grad_norms,
            training_steps=training_steps_n,
            shared_optimiser=shared_optimiser,
            spec=spec,
        )

        metrics = evaluate(
            model=model,
            task_fn=task_fn,
            active_Ns=active_set,
            batch_size=batch_size,
            test_steps=test_steps,
            device=device,
            criterion=criterion,
            head_idx=head_idx,
            spec=spec,
        )

        acc = metrics["score"] if "score" in metrics else 0.0

        row = {
            "phase": phase,
            "N": next_N,
            "loss": summ["loss"],
            "acc": acc,
            "gRNN": summ["gRNN"],
            "gCB": summ["gCB"],
            "gRNN_pre": summ.get("gRNN_pre", summ["gRNN"]),
            "gCB_pre": summ.get("gCB_pre", summ["gCB"]),
            "task_loss": summ.get("task_loss", summ["loss"]),
            "epoch": global_epoch,
        }

        logs.append(row)

        if on_log is not None:
            on_log(row)

        print(
            f"Phase {phase} | N={next_N} | Global Epoch={global_epoch} | "
            f"Loss={row['loss']:.4f} | Acc={row['acc']:.2f}% | "
            f"gRNN={row['gRNN']:.4f} | gCB={row['gCB']:.4f}",
            flush=True,
        )

        if spec["advance_fn"](metrics):
            solved = True
            print(f"{phase} solved N={next_N}", metrics, flush=True)
            break

    if not solved:
        return False, active_Ns, current_N, logs, global_epoch

    return True, active_set, next_N, logs, global_epoch