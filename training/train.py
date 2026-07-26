import sys
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import re
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.task_registry import TASK_SPECS, compute_loss
from save import save_model, find_next_free_network_number, make_unique_dir

from model.models_cb import ElmanRNNMultiHead
from model.GRU_test import GRUMultiHeadWithCB

from tasks.multitask_impl import multitask_train, MULTITASK_TASKS
from tasks.continual_impl import continual_train
from tasks.task_switch_one import switch_train
from training.train_reservoir import train_reservoir


def parse_optional_int(value):
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in {"none", "null"}:
        return None

    return int(value)


def train(
    model,
    curriculum_type,
    task,
    num_epochs,
    Ns,
    args,
    run_number,
    spec,
    optimizer,
    batch_size,
    training_steps,
    test_steps,
    device,
    base_path,
    affixes,
    n_forget,
    target_end_n=150,
    threshold_final=98.0,
    patience=1,
    subdir_override=None,
    stage_tag=None,
):
    stats = {
        "stage": [],
        "task": [],
        "n_task": [],
        "phase": [],
        "loss": [],
        "accuracy": [],
        "grad_rnn": [],
        "grad_cb": [],
    }

    task_function_local = spec["batch_fn"]
    criterion_local = spec["criterion_ctor"]()

    if subdir_override is None:
        subdir = save_model(
            model,
            curriculum_type=curriculum_type,
            n_heads=len(Ns) if curriculum_type == "cumulative" else 1,
            n_forget=n_forget,
            task=task,
            network_number=run_number,
            N_max=Ns[-1],
            N_min=Ns[0],
            init=True,
            args=args,
            base_path=base_path,
            affixes=affixes,
        )
    else:
        subdir = subdir_override
        os.makedirs(subdir, exist_ok=True)

    # Resume: append to existing stats if present
    if getattr(args, "resume_ckpt", None) and subdir_override is not None:
        stats_path = os.path.join(subdir, "stats.npy")

        if os.path.exists(stats_path):
            try:
                old_stats = np.load(stats_path, allow_pickle=True).item()

                for key in stats:
                    if key in old_stats and isinstance(old_stats[key], list):
                        stats[key] = list(old_stats[key])

                print(
                    f"[resume] Loaded existing stats from {stats_path} "
                    f"(rows={len(stats['n_task'])})",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"[resume] Warning: failed to load existing stats ({e}); starting fresh.",
                    flush=True,
                )

    print(f"Saving results to: {subdir}", flush=True)

    losses = []
    accuracies = []

    try:
        solved_streak = 0

        for epoch in tqdm(range(num_epochs)):
            losses_step = []
            grad_rnn_step = []

            for i in range(training_steps):
                sequences, labels = task_function_local(Ns, batch_size)
                labels = labels if isinstance(labels, list) else [labels]

                sequences = sequences.to(device)
                labels = [label.to(device) for label in labels]

                optimizer.zero_grad(set_to_none=True)

                _, out_heads = model(
                    sequences,
                    return_timewise=spec["timewise_output"],
                )
                out_heads = out_heads if isinstance(out_heads, list) else [out_heads]
                out_heads = out_heads[:len(Ns)]

                task_loss = compute_loss(
                    out_heads,
                    labels,
                    spec["target_type"],
                    criterion_local,
                )

                if not torch.isfinite(task_loss):
                    print("Skipping non-finite batch loss", flush=True)
                    continue

                task_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=7.5)
                optimizer.step()

                losses_step.append(float(task_loss.item()))

                total_grad = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5

                grad_rnn_step.append(total_grad)

            losses.append(np.mean(losses_step) if losses_step else np.nan)

            # ---- Testing ----
            metric_accumulator = []

            model.eval()
            with torch.no_grad():
                for _ in range(test_steps):
                    sequences, labels = task_function_local(Ns, batch_size)
                    labels = labels if isinstance(labels, list) else [labels]

                    sequences = sequences.to(device)
                    labels = [label.to(device) for label in labels]

                    _, out_heads = model(
                        sequences,
                        return_timewise=spec["timewise_output"],
                    )
                    out_heads = out_heads if isinstance(out_heads, list) else [out_heads]
                    out_heads = out_heads[:len(Ns)]

                    metric_accumulator.append(
                        spec["metric_fn"](out_heads, labels)
                    )

            model.train()

            mean_score = float(np.mean([m["score"] for m in metric_accumulator]))
            accuracy = np.array([mean_score for _ in Ns])
            accuracies.append(accuracy)

            loss_name = spec.get("loss_name", "loss")
            metric_name = (
                metric_accumulator[0].get("name", "score")
                if metric_accumulator
                else "score"
            )

            finite_losses = [x for x in losses_step if np.isfinite(x)]
            mean_loss = float(np.nanmean(finite_losses)) if finite_losses else float("nan")

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{training_steps}] | "
                f"Loss ({loss_name}): {mean_loss:.4f} | "
                f"Performance ({metric_name}): {mean_score:.4f}",
                flush=True,
            )

            if metric_accumulator and "per_head" in metric_accumulator[0]:
                per_head_mean = np.mean(
                    np.array([m["per_head"] for m in metric_accumulator]),
                    axis=0,
                )

                print(
                    f"Per-N Performance ({metric_name}):\n"
                    + "".join(
                        f"  N={Ns[idx]}: {per_head_mean[idx]:.4f}\n"
                        for idx in range(len(Ns))
                    ),
                    flush=True,
                )

            if metric_accumulator and "endpoint_error" in metric_accumulator[0]:
                mean_ep = float(
                    np.mean([m["endpoint_error"] for m in metric_accumulator])
                )

                print(f"Aux metric (endpoint_error): {mean_ep:.4f}", flush=True)

                if "per_head_endpoint_error" in metric_accumulator[0]:
                    per_ep = np.mean(
                        np.array(
                            [
                                m["per_head_endpoint_error"]
                                for m in metric_accumulator
                            ]
                        ),
                        axis=0,
                    )

                    print(
                        "Per-N Aux metric (endpoint_error):\n"
                        + "".join(
                            f"  N={Ns[idx]}: {per_ep[idx]:.4f}\n"
                            for idx in range(len(Ns))
                        ),
                        flush=True,
                    )

            if target_end_n is not None:
                if Ns[-1] >= target_end_n and mean_score >= threshold_final:
                    solved_streak += 1
                else:
                    solved_streak = 0

                if solved_streak >= patience:
                    print(
                        f"[STAGE DONE] task={task} reached target_end_n={target_end_n} "
                        f"with score={mean_score:.2f} for {patience} evals. Stopping.",
                        flush=True,
                    )
                    break

            stats["n_task"].append(Ns[-1])
            stats["phase"].append("RNN")
            stats["loss"].append(mean_loss)
            stats["accuracy"].append(float(np.mean(accuracy)))
            stats["grad_rnn"].append(
                float(np.mean(grad_rnn_step)) if grad_rnn_step else float("nan")
            )
            stats["grad_cb"].append(0.0)
            stats["stage"].append(stage_tag if stage_tag is not None else "")
            stats["task"].append(task)

            np.save(os.path.join(subdir, "stats.npy"), stats)

            # ---- Curriculum advancement ----
            metric_for_curriculum = {
                "score": float(np.mean([m["score"] for m in metric_accumulator]))
            }

            excluded_keys = {
                "score",
                "per_head",
                "name",
                "per_head_endpoint_error",
                "per_head_mse",
            }

            for key in metric_accumulator[0]:
                if key in excluded_keys:
                    continue

                vals = [m[key] for m in metric_accumulator if key in m]

                if vals:
                    metric_for_curriculum[key] = float(np.mean(vals))

            if spec["advance_fn"](metric_for_curriculum):
                if target_end_n is not None and Ns[-1] >= target_end_n:
                    pass

                else:
                    old_Ns = list(Ns)
                    ct = str(curriculum_type).strip().lower()

                    print(
                        f"[ADVANCE TRIGGERED] curriculum_type={repr(curriculum_type)} | "
                        f"old_Ns={old_Ns}",
                        flush=True,
                    )

                    if subdir_override is None:
                        save_model(
                            model,
                            curriculum_type=curriculum_type,
                            n_heads=len(Ns),
                            n_forget=n_forget,
                            task=task,
                            network_number=run_number,
                            N_max=Ns[-1],
                            N_min=Ns[0],
                            base_path=base_path,
                            args=args,
                            affixes=affixes,
                        )
                    else:
                        ckpt_dir = os.path.join(subdir, "checkpoints")
                        os.makedirs(ckpt_dir, exist_ok=True)

                        ckpt_path = os.path.join(
                            ckpt_dir,
                            f"{task}_{stage_tag or 'stage'}_ep{epoch + 1}_N{Ns[-1]}.pt",
                        )

                        torch.save({"state_dict": model.state_dict()}, ckpt_path)

                    if ct == "cumulative":
                        Ns = list(Ns) + [
                            int(Ns[-1]) + 1 + i for i in range(args.add_heads)
                        ]

                    elif ct == "sliding":
                        Ns = list(Ns[n_forget:]) + [
                            int(Ns[-1]) + 1 + i for i in range(n_forget)
                        ]

                    elif ct == "single":
                        Ns = [int(Ns[0]) + 1]

                    elif ct == "single_nocurr":
                        print("[ADVANCE] single_nocurr -> stopping", flush=True)
                        break

                    else:
                        raise ValueError(f"Unknown curriculum_type: {repr(curriculum_type)}")

                    print(f"[ADVANCED] {old_Ns} -> {Ns}", flush=True)

    except KeyboardInterrupt:
        print(
            f"\nTraining interrupted. Saving stats to {subdir}/stats.npy...",
            flush=True,
        )
        np.save(os.path.join(subdir, "stats.npy"), stats)
        return stats

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--base_path", type=str, dest="base_path")
    parser.add_argument("-nn", "--num_neurons", type=int, dest="num_neurons")
    parser.add_argument("-ni", "--ns_init", type=int, dest="ns_init")
    parser.add_argument("-a", "--afunc", type=str, dest="afunc")
    parser.add_argument("-c", "--curriculum_type", type=str, dest="curriculum_type")
    parser.add_argument("-t", "--task", type=str, dest="task")
    parser.add_argument("-T", "--tau", type=float, dest="tau")
    parser.add_argument("-n", "--network_number", type=int, dest="network_number")
    parser.add_argument("-ih", "--init_heads", type=int, dest="init_heads")
    parser.add_argument("-dh", "--add_heads", type=int, dest="add_heads")
    parser.add_argument("-fh", "--forget_heads", type=int, dest="forget_heads")
    parser.add_argument("-s", "--seed", type=int, dest="seed")

    parser.add_argument("--no_cb", dest="use_cb_bias", action="store_false", default=True)
    parser.add_argument("--readout_mode", type=str, dest="readout_mode", default="single")
    parser.add_argument("--cb_store", dest="cb_store", action="store_true", default=False)
    parser.add_argument("--rnn_lr", type=float, dest="rnn_lr", default=0.05)
    parser.add_argument("--cb_lr", type=float, dest="cb_lr", default=0.05)

    parser.add_argument(
        "--no_shared_optimiser",
        dest="shared_optimiser",
        action="store_false",
        default=True,
    )

    parser.add_argument("--affixes", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=3000)

    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--resume_subdir", type=str, default=None)
    parser.add_argument("--resume_start_n", type=int, default=None)
    parser.add_argument(
        "--no_resume_strict",
        dest="resume_strict",
        action="store_false",
        default=True,
    )

    parser.add_argument("--skip_init_phase", action="store_true", default=False)

    parser.add_argument("--gc_dim", type=int, default=512)
    parser.add_argument("--pc_dim", type=int, default=64)
    parser.add_argument("--cb_max_ratio", type=float, default=1.0)

    parser.add_argument("--cb_sees_input", action="store_true", default=False)
    parser.add_argument("--cb_no_hidden", dest="cb_no_hidden", action="store_true", default=False)
    parser.add_argument("--cb_sees_hidden", dest="cb_no_hidden", action="store_false")

    parser.add_argument("--task_switch", action="store_true", default=False)
    parser.add_argument("--task_switch_plan", type=str, default="dms,violation,dms")
    parser.add_argument("--task_switch_targetN", type=int, default=13)
    parser.add_argument(
        "--use_reservoir_in_switch",
        action="store_true",
        default=False,
    )

    # Backward-compatible alias for old scripts.
    parser.add_argument(
        "--use_alternating_in_switch",
        dest="use_reservoir_in_switch",
        action="store_true",
        default=False,
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--training_steps", type=int, default=100)
    parser.add_argument("--test_steps", type=int, default=50)

    parser.add_argument("--multitask", action="store_true", default=False)
    parser.add_argument("--mt_target_n", type=parse_optional_int, default=150)
    parser.add_argument("--mt_advance_threshold", type=float, default=98.0)
    parser.add_argument("--mt_patience", type=int, default=1)
    parser.add_argument(
        "--mt_reservoir_mode",
        type=str,
        default="off",
        choices=["off", "global_once", "all_tasks_once", "periodic_refresh"],
    )

    parser.add_argument("--continual", action="store_true", default=False)
    parser.add_argument("--continual_plan", type=str, default="dms,parity,dms")
    parser.add_argument("--continual_epochs_per_phase", type=int, default=100)
    parser.add_argument("--continual_target_n", type=parse_optional_int, default=150)
    parser.add_argument("--continual_patience", type=int, default=1)
    parser.add_argument("--ct_advance_threshold", type=float, default=98.0)
    parser.add_argument("--continual_savings_threshold", type=float, default=98.0)
    parser.add_argument(
        "--continual_reservoir_mode",
        type=str,
        default="off",
        choices=["off", "global_once", "per_task_once", "periodic_refresh"],
    )

    parser.add_argument("--reservoir_interval_n", type=int, default=10)

    # Backward-compatible alias for old reservoir scripts.
    parser.add_argument(
        "--cb_reservoir_n",
        dest="reservoir_interval_n",
        type=int,
    )

    parser.add_argument("--ct_switch", action="store_true", default=False)
    parser.add_argument("--switch_n", type=int, default=10)

    parser.add_argument(
        "--model_type",
        type=str,
        default="elman",
        choices=["elman", "gru"],
        help="Type of recurrent model to use.",
    )

    parser.add_argument("--aux_rnn_cb", action="store_true", default=False,
                        help="Use an auxiliary RNN for the cerebellum (CB) module.")
    parser.add_argument("--aux_rnn_hidden_size", type=int, default=139,
                        help="Hidden size for the auxiliary RNN in the CB module.")

    parser.set_defaults(
        num_neurons=500,
        curriculum_type="cumulative",
        task="parity",
        tau=None,
        network_number=1,
        init_heads=1,
        add_heads=1,
        forget_heads=1,
        seed=np.random.choice(2**32),
    )

    args = parser.parse_args()

    if args.gc_dim <= 0:
        raise ValueError(f"--gc_dim must be positive, got {args.gc_dim}")

    print("num_neurons:", args.num_neurons)
    print("curriculum_type:", args.curriculum_type)
    print("task:", args.task)
    print("network_number:", args.network_number)

    BASE_PATH = args.base_path
    NS_INIT = args.ns_init
    NUM_NEURONS = args.num_neurons
    AFFIXES = []

    CURRICULUM = args.curriculum_type
    TASK = args.task
    NETWORK_NUMBER = args.network_number
    TAU = args.tau if args.tau is not None else 1.5

    INIT_HEADS = args.init_heads
    NUM_ADD = args.add_heads
    NUM_FORGET = args.forget_heads

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    if TASK not in TASK_SPECS:
        raise ValueError(f"Unrecognized task: {TASK}")

    spec = TASK_SPECS[TASK]
    task_function = spec["batch_fn"]

    INPUT_SIZE = spec["input_size"]
    NUM_CLASSES = spec["output_size"]
    START_N = spec.get("start_n", 2)

    # ---- Curriculum setup ----
    if CURRICULUM == "cumulative":
        Ns_init = list(np.arange(START_N, START_N + INIT_HEADS))

    elif CURRICULUM == "sliding":
        if INIT_HEADS < NUM_FORGET:
            INIT_HEADS = NUM_FORGET
        Ns_init = list(np.arange(START_N, START_N + INIT_HEADS))

    elif CURRICULUM in ("single", "single_nocurr", "reservoir", "alternating"):
        Ns_init = [START_N]
        INIT_HEADS = 1
        NUM_FORGET = 1

    else:
        raise ValueError(f"Unrecognized curriculum type: {CURRICULUM}")

    # Backward compatibility: old scripts may still say "alternating".
    if CURRICULUM == "alternating":
        print("[compat] curriculum_type='alternating' mapped to reservoir training.", flush=True)

    if NS_INIT is not None:
        Ns_init = [N - 2 + NS_INIT for N in Ns_init]

    def infer_resume_n_from_path(path):
        name = os.path.basename(str(path))
        n_tokens = re.findall(r"N(\d+)", name)
        return int(n_tokens[-1]) if n_tokens else None

    # ---- Readout head count ----
    if CURRICULUM == "cumulative":
        MAX_N = 150
        NUM_READOUT_HEADS = MAX_N - START_N + 1

    elif CURRICULUM == "sliding":
        NUM_READOUT_HEADS = INIT_HEADS

    elif CURRICULUM in ("single", "single_nocurr", "reservoir", "alternating"):
        NUM_READOUT_HEADS = 1

    else:
        NUM_READOUT_HEADS = INIT_HEADS

    BIAS = True

    AFUNC_MAP = {
        "leakyrelu": nn.LeakyReLU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }

    AFUNC = AFUNC_MAP.get(args.afunc, nn.LeakyReLU)

    if args.afunc and args.afunc not in AFUNC_MAP:
        print(
            f"Unrecognized activation function: {args.afunc}, defaulting to LeakyReLU",
            flush=True,
        )

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    TRAINING_STEPS = args.training_steps
    TEST_STEPS = args.test_steps
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Build affixes ----
    AFFIXES += [args.model_type]

    if NUM_NEURONS != 500:
        AFFIXES += ["size", str(NUM_NEURONS)]

    if args.use_cb_bias:
        if args.aux_rnn_cb:
            AFFIXES += [f"auxH{args.aux_rnn_hidden_size}"]
        else:
            AFFIXES += ["CB", f"gc{args.gc_dim}"]

            if args.pc_dim != 64:
                AFFIXES += [f"pc{args.pc_dim}"]
    else:
        AFFIXES += ["noCB"]

    if args.readout_mode == "sliding":
        AFFIXES += ["sliding"]

    elif args.readout_mode == "cumulative":
        AFFIXES += ["cumulative"]

    if args.cb_store:
        AFFIXES += ["CBstore"]

    if args.rnn_lr != 0.05:
        AFFIXES += [f"RNNlr{args.rnn_lr}"]

    if args.cb_lr != 0.05:
        AFFIXES += [f"CBlr{args.cb_lr}"]

    if args.cb_sees_input or args.cb_no_hidden:
        AFFIXES += ["CBinput"]

    if args.cb_no_hidden:
        AFFIXES += ["noH"]

    if CURRICULUM in ("reservoir", "alternating"):
        AFFIXES += ["reservoir", f"k{args.reservoir_interval_n}"]

    if not args.shared_optimiser:
        AFFIXES += ["opts"]

    if args.affixes:
        AFFIXES += [args.affixes]

    def build_model(
        input_size,
        hidden_size,
        num_classes,
        num_readout_heads,
        cb_input_size,
        cb_no_hidden=False,
    ):
        if args.model_type == "elman":
            return ElmanRNNMultiHead(
                input_size=input_size,
                hidden_size=hidden_size,
                cb_gc_dim=args.gc_dim,
                cb_pc_dim=args.pc_dim,
                num_classes=num_classes,
                num_readout_heads=num_readout_heads,
                tau=TAU,
                afunc=AFUNC,
                bias=BIAS,
                use_cb_bias=args.use_cb_bias,
                cb_input_size=cb_input_size,
                cb_no_hidden=cb_no_hidden,
                cb_max_ratio=args.cb_max_ratio,
                rnn_aux_ctrl=args.aux_rnn_cb,
                aux_hidden_size=args.aux_rnn_hidden_size,
            ).to(device)

        if args.model_type == "gru":
            return GRUMultiHeadWithCB(
                input_size=input_size,
                hidden_size=hidden_size,
                cb_gc_dim=args.gc_dim,
                cb_pc_dim=args.pc_dim,
                num_classes=num_classes,
                num_readout_heads=num_readout_heads,
                candidate_afunc=AFUNC,
                bias=BIAS,
                use_cb_bias=args.use_cb_bias,
                cb_input_size=cb_input_size,
                cb_no_hidden=cb_no_hidden,
                cb_max_ratio=args.cb_max_ratio,
            ).to(device)

        raise ValueError(f"Unknown model_type: {args.model_type}")

    # ---- Build model ----
    cb_input_size = INPUT_SIZE if (args.cb_sees_input or args.cb_no_hidden) else 0

    if args.multitask:
        NUM_TASKS = len(MULTITASK_TASKS)

        rnn = build_model(
            input_size=INPUT_SIZE,
            hidden_size=NUM_NEURONS,
            num_classes=NUM_CLASSES,
            num_readout_heads=NUM_TASKS,
            cb_input_size=cb_input_size,
            cb_no_hidden=args.cb_no_hidden,
        )

    elif args.continual or args.ct_switch:
        plan = [t.strip() for t in args.continual_plan.split(",") if t.strip()]
        unique_tasks = list(dict.fromkeys(plan))

        INPUT_SIZE += 1

        cb_input_size = INPUT_SIZE if (args.cb_sees_input or args.cb_no_hidden) else 0

        rnn = build_model(
            input_size=INPUT_SIZE,
            hidden_size=NUM_NEURONS,
            num_classes=NUM_CLASSES,
            num_readout_heads=len(unique_tasks),
            cb_input_size=cb_input_size,
            cb_no_hidden=args.cb_no_hidden,
        )

    else:
        rnn = build_model(
            input_size=INPUT_SIZE,
            hidden_size=NUM_NEURONS,
            num_classes=NUM_CLASSES,
            num_readout_heads=NUM_READOUT_HEADS,
            cb_input_size=cb_input_size,
            cb_no_hidden=args.cb_no_hidden,
        )

    # ---- Optional resume ----
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        rnn.load_state_dict(state_dict, strict=args.resume_strict)

        print(
            f"[resume] Loaded checkpoint: {args.resume_ckpt} "
            f"(strict={args.resume_strict})",
            flush=True,
        )

        if args.resume_subdir is None:
            inferred = os.path.dirname(args.resume_ckpt)

            if os.path.basename(inferred) == "checkpoints":
                inferred = os.path.dirname(inferred)

            args.resume_subdir = inferred

        print(f"[resume] Writing into: {args.resume_subdir}", flush=True)

        resume_n = args.resume_start_n or infer_resume_n_from_path(args.resume_ckpt)

        if resume_n is not None:
            if CURRICULUM in ("single", "single_nocurr", "reservoir", "alternating"):
                Ns_init = [int(resume_n)]
            else:
                Ns_init = list(np.arange(int(resume_n), int(resume_n) + INIT_HEADS))

            print(f"[resume] Starting from N={resume_n} -> Ns_init={Ns_init}", flush=True)

        else:
            print("[resume] Could not infer N from checkpoint name; keeping current Ns_init.", flush=True)

    # ---------------------------------------------------------------
    # TASK SWITCHING
    # ---------------------------------------------------------------
    if args.task_switch:
        plan = [t.strip() for t in args.task_switch_plan.split(",") if t.strip()]

        if not plan:
            raise ValueError("task_switch_plan is empty")

        for task_name in plan:
            if task_name not in TASK_SPECS:
                raise ValueError(f"Unrecognized task in task_switch_plan: {task_name}")

        print(f"\n=== TASK SWITCH ENABLED === plan={plan} ===\n", flush=True)

        if args.use_cb_bias:
            args.use_reservoir_in_switch = True

        BLOCK_STEP = 10
        BLOCK_END = 10
        MAX_BLOCK_END = int(args.task_switch_targetN)
        MIN_START_N = 2
        BLOCK_TASKS = sorted(set(plan))

        switch_optimizer = torch.optim.SGD(
            list(rnn.parameters()),
            lr=args.rnn_lr,
            momentum=0.1,
            nesterov=True,
        )

        switch_affixes = AFFIXES + ["SWITCH"]

        run_number = find_next_free_network_number(
            base_path=BASE_PATH,
            curriculum_type="task_switch",
            task="switch",
            affixes=switch_affixes,
            n_heads=1,
            n_forget=NUM_FORGET,
        )

        base_subdir = save_model(
            rnn,
            curriculum_type="task_switch",
            n_heads=1,
            n_forget=NUM_FORGET,
            task="switch",
            network_number=run_number,
            N_max=MAX_BLOCK_END,
            N_min=START_N,
            init=True,
            args=args,
            base_path=BASE_PATH,
            affixes=switch_affixes,
        )

        print(f"[task_switch] Base folder: {base_subdir}", flush=True)

        done_this_block = set()
        last_N_by_task = {}
        stage_i = 0

        while True:
            if (
                BLOCK_END >= MAX_BLOCK_END
                and done_this_block == set(BLOCK_TASKS)
                and stage_i > 0
            ):
                break

            stage_task = plan[stage_i % len(plan)]
            spec_stage = TASK_SPECS[stage_task]
            task_fn_stage = spec_stage["batch_fn"]

            startN = max(
                int(last_N_by_task.get(stage_task, spec_stage.get("start_n", 2))),
                MIN_START_N,
            )
            stage_endN = int(min(BLOCK_END, MAX_BLOCK_END))

            if startN >= stage_endN:
                done_this_block.add(stage_task)
                print(
                    f"[STAGE {stage_i}] task={stage_task} already at "
                    f"startN={startN} >= block_target={stage_endN}",
                    flush=True,
                )

            else:
                stage_dir = make_unique_dir(
                    os.path.join(base_subdir, f"switch{stage_i}_{stage_task}")
                )

                print(
                    f"[STAGE {stage_i}] task={stage_task} "
                    f"startN={startN} -> target_endN={stage_endN} dir={stage_dir}",
                    flush=True,
                )

                if args.use_reservoir_in_switch:
                    train_reservoir(
                        model=rnn,
                        curriculum_type="reservoir",
                        task_name=stage_task,
                        task_function=task_fn_stage,
                        num_epochs=NUM_EPOCHS,
                        Ns_init=[startN],
                        run_number=0,
                        batch_size=BATCH_SIZE,
                        training_steps=TRAINING_STEPS,
                        test_steps=TEST_STEPS,
                        device=device,
                        base_path=BASE_PATH,
                        affixes=AFFIXES + [f"switch{stage_i}", stage_task],
                        readout_head_dyn=args.readout_mode,
                        n_heads=1,
                        n_forget=NUM_FORGET,
                        args=args,
                        shared_optimiser=args.shared_optimiser,
                        threshold_final=98.0,
                        spec=spec_stage,
                        skip_init_phase=args.skip_init_phase,
                        target_end_n=stage_endN,
                        subdir_override=stage_dir,
                        stage_tag=f"switch{stage_i}",
                        rnn_lr=args.rnn_lr,
                        cb_lr=args.cb_lr,
                        reservoir_interval_n=args.reservoir_interval_n,
                    )

                else:
                    train(
                        rnn,
                        curriculum_type="single",
                        task=stage_task,
                        num_epochs=NUM_EPOCHS,
                        Ns=[startN],
                        args=args,
                        run_number=0,
                        spec=spec_stage,
                        optimizer=switch_optimizer,
                        batch_size=BATCH_SIZE,
                        training_steps=TRAINING_STEPS,
                        test_steps=TEST_STEPS,
                        device=device,
                        base_path=BASE_PATH,
                        affixes=AFFIXES,
                        n_forget=NUM_FORGET,
                        target_end_n=stage_endN,
                        threshold_final=98.0,
                        patience=3,
                        subdir_override=stage_dir,
                        stage_tag=f"switch{stage_i}",
                    )

                st = np.load(
                    os.path.join(stage_dir, "stats.npy"),
                    allow_pickle=True,
                ).item()

                lastN = int(st["n_task"][-1])
                last_N_by_task[stage_task] = lastN

                if lastN >= stage_endN:
                    done_this_block.add(stage_task)

                print(
                    f"[STAGE {stage_i} DONE] task={stage_task} lastN={lastN} "
                    f"block_target={stage_endN} done_this_block={sorted(done_this_block)}",
                    flush=True,
                )

            if done_this_block == set(BLOCK_TASKS):
                if BLOCK_END < MAX_BLOCK_END:
                    BLOCK_END = min(BLOCK_END + BLOCK_STEP, MAX_BLOCK_END)
                    print(f"[BLOCK ADVANCE] New BLOCK_END={BLOCK_END}", flush=True)
                    done_this_block = set()
                else:
                    print("[task_switch] Final block complete. Stopping.", flush=True)
                    break

            stage_i += 1

        print("[task_switch] COMPLETE", flush=True)
        sys.exit(0)

    # ---------------------------------------------------------------
    # MULTITASK
    # ---------------------------------------------------------------
    if args.multitask:
        for task_name in MULTITASK_TASKS:
            if task_name not in TASK_SPECS:
                raise ValueError(f"Task '{task_name}' missing from TASK_SPECS")

        mt_affixes = AFFIXES + ["MULTITASK"]

        run_number = find_next_free_network_number(
            base_path=BASE_PATH,
            curriculum_type="multitask",
            task="mt",
            affixes=mt_affixes,
            n_heads=len(MULTITASK_TASKS),
            n_forget=1,
        )

        mt_subdir = save_model(
            rnn,
            curriculum_type="multitask",
            n_heads=len(MULTITASK_TASKS),
            n_forget=1,
            task="mt",
            network_number=run_number,
            N_max=args.mt_target_n,
            N_min=2,
            init=True,
            args=args,
            base_path=BASE_PATH,
            affixes=mt_affixes,
        )

        if getattr(args, "aux_rnn_cb", False) and getattr(rnn, "cb", None) is not None:
            cb_param_ids = {id(p) for p in rnn.cb.parameters()}
            non_cb_params = [p for p in rnn.parameters() if id(p) not in cb_param_ids]
            cb_params = [p for p in rnn.parameters() if id(p) in cb_param_ids]
            optimizer = torch.optim.SGD(
                [{"params": non_cb_params, "lr": args.rnn_lr},
                {"params": cb_params, "lr": args.cb_lr}],
                momentum=0.1, nesterov=True,
            )
        else:
            optimizer = torch.optim.SGD(
                list(rnn.parameters()), lr=args.rnn_lr, momentum=0.1, nesterov=True,
            )

        def make_optimizer(params):
            params = list(params)
            cb_p = [p for p in params if id(p) in cb_param_ids]
            non_cb_p = [p for p in params if id(p) not in cb_param_ids]

            groups = []

            if non_cb_p:
                groups.append({"params": non_cb_p, "lr": args.rnn_lr})

            if cb_p:
                groups.append({"params": cb_p, "lr": args.cb_lr})

            return torch.optim.SGD(groups, momentum=0.1, nesterov=True)

        multitask_train(
            model=rnn,
            task_specs=TASK_SPECS,
            optimizer_ctor=make_optimizer,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            training_steps=TRAINING_STEPS,
            test_steps=TEST_STEPS,
            device=device,
            subdir=mt_subdir,
            advance_patience=args.mt_patience,
            target_end_n=args.mt_target_n,
            task_names=MULTITASK_TASKS,
            mt_advance_threshold=args.mt_advance_threshold,
            reservoir_mode=args.mt_reservoir_mode,
            reservoir_interval_n=args.reservoir_interval_n,
        )

        sys.exit(0)

    # ---------------------------------------------------------------
    # CONTINUAL
    # ---------------------------------------------------------------
    if args.continual:
        plan = [t.strip() for t in args.continual_plan.split(",") if t.strip()]

        for task_name in plan:
            if task_name not in TASK_SPECS:
                raise ValueError(f"Task '{task_name}' not in TASK_SPECS")

        cont_affixes = AFFIXES + ["CONTINUAL"]

        run_number = find_next_free_network_number(
            base_path=BASE_PATH,
            curriculum_type="continual",
            task="cont",
            affixes=cont_affixes,
            n_heads=len(set(plan)),
            n_forget=1,
        )

        cont_subdir = save_model(
            rnn,
            curriculum_type="continual",
            n_heads=len(set(plan)),
            n_forget=1,
            task="cont",
            network_number=run_number,
            N_max=args.continual_target_n,
            N_min=2,
            init=True,
            args=args,
            base_path=BASE_PATH,
            affixes=cont_affixes,
        )

        if getattr(args, "aux_rnn_cb", False) and getattr(rnn, "cb", None) is not None:
            cb_param_ids = {id(p) for p in rnn.cb.parameters()}
            non_cb_params = [p for p in rnn.parameters() if id(p) not in cb_param_ids]
            cb_params = [p for p in rnn.parameters() if id(p) in cb_param_ids]
            optimizer = torch.optim.SGD(
                [{"params": non_cb_params, "lr": args.rnn_lr},
                {"params": cb_params, "lr": args.cb_lr}],
                momentum=0.1, nesterov=True,
            )
        else:
            optimizer = torch.optim.SGD(
                list(rnn.parameters()), lr=args.rnn_lr, momentum=0.1, nesterov=True,
            )

        def make_optimizer(params):
            params = list(params)
            cb_p = [p for p in params if id(p) in cb_param_ids]
            non_cb_p = [p for p in params if id(p) not in cb_param_ids]

            groups = []

            if non_cb_p:
                groups.append({"params": non_cb_p, "lr": args.rnn_lr})

            if cb_p:
                groups.append({"params": cb_p, "lr": args.cb_lr})

            return torch.optim.SGD(groups, momentum=0.1, nesterov=True)

        continual_train(
            model=rnn,
            task_specs=TASK_SPECS,
            optimizer_ctor=make_optimizer,
            plan=plan,
            epochs_per_phase=args.continual_epochs_per_phase,
            max_global_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            training_steps=TRAINING_STEPS,
            test_steps=TEST_STEPS,
            device=device,
            subdir=cont_subdir,
            target_end_n=args.continual_target_n,
            advance_patience=args.continual_patience,
            ct_advance_threshold=args.ct_advance_threshold,
            savings_threshold=args.continual_savings_threshold,
            reservoir_mode=args.continual_reservoir_mode,
            reservoir_interval_n=args.reservoir_interval_n,
        )

        sys.exit(0)

    # ---------------------------------------------------------------
    # CT SWITCH
    # ---------------------------------------------------------------
    if args.ct_switch:
        plan = [t.strip() for t in args.continual_plan.split(",") if t.strip()]

        for task_name in plan:
            if task_name not in TASK_SPECS:
                raise ValueError(f"Task '{task_name}' not in TASK_SPECS")

        switch_affixes = AFFIXES + ["CTSWITCH"]

        run_number = find_next_free_network_number(
            base_path=BASE_PATH,
            curriculum_type="ct_switch",
            task="ctswitch",
            affixes=switch_affixes,
            n_heads=len(set(plan)),
            n_forget=1,
        )

        switch_subdir = save_model(
            rnn,
            curriculum_type="ct_switch",
            n_heads=len(set(plan)),
            n_forget=1,
            task="ctswitch",
            network_number=run_number,
            N_max=args.continual_target_n,
            N_min=2,
            init=True,
            args=args,
            base_path=BASE_PATH,
            affixes=switch_affixes,
        )

        if getattr(args, "aux_rnn_cb", False) and getattr(rnn, "cb", None) is not None:
            cb_param_ids = {id(p) for p in rnn.cb.parameters()}
            non_cb_params = [p for p in rnn.parameters() if id(p) not in cb_param_ids]
            cb_params = [p for p in rnn.parameters() if id(p) in cb_param_ids]
            optimizer = torch.optim.SGD(
                [{"params": non_cb_params, "lr": args.rnn_lr},
                {"params": cb_params, "lr": args.cb_lr}],
                momentum=0.1, nesterov=True,
            )
        else:
            optimizer = torch.optim.SGD(
                list(rnn.parameters()), lr=args.rnn_lr, momentum=0.1, nesterov=True,
            )
        def make_optimizer(params):
            params = list(params)
            cb_p = [p for p in params if id(p) in cb_param_ids]
            non_cb_p = [p for p in params if id(p) not in cb_param_ids]

            groups = []

            if non_cb_p:
                groups.append({"params": non_cb_p, "lr": args.rnn_lr})

            if cb_p:
                groups.append({"params": cb_p, "lr": args.cb_lr})

            return torch.optim.SGD(groups, momentum=0.1, nesterov=True)

        switch_train(
            model=rnn,
            task_specs=TASK_SPECS,
            optimizer_ctor=make_optimizer,
            plan=plan,
            epochs_per_phase=args.continual_epochs_per_phase,
            max_global_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            training_steps=TRAINING_STEPS,
            test_steps=TEST_STEPS,
            device=device,
            subdir=switch_subdir,
            target_end_n=args.continual_target_n,
            advance_patience=args.continual_patience,
            ct_advance_threshold=args.ct_advance_threshold,
            savings_threshold=args.continual_savings_threshold,
            reservoir_mode=args.continual_reservoir_mode,
            reservoir_interval_n=args.reservoir_interval_n,
            switch_n=args.switch_n,
        )

        sys.exit(0)

    # ---------------------------------------------------------------
    # RESERVOIR / STANDARD TRAINING
    # ---------------------------------------------------------------
    if CURRICULUM in ("reservoir", "alternating"):
        train_reservoir(
            model=rnn,
            curriculum_type="reservoir",
            task_name=TASK,
            task_function=task_function,
            num_epochs=NUM_EPOCHS,
            Ns_init=Ns_init,
            run_number=NETWORK_NUMBER,
            batch_size=BATCH_SIZE,
            training_steps=TRAINING_STEPS,
            test_steps=TEST_STEPS,
            device=device,
            base_path=BASE_PATH,
            affixes=AFFIXES,
            readout_head_dyn=args.readout_mode,
            n_heads=INIT_HEADS,
            n_forget=NUM_FORGET,
            cb_lr=args.cb_lr,
            rnn_lr=args.rnn_lr,
            args=args,
            shared_optimiser=args.shared_optimiser,
            threshold_final=98.0,
            spec=spec,
            skip_init_phase=args.skip_init_phase,
            reservoir_interval_n=args.reservoir_interval_n,
            subdir_override=args.resume_subdir,
        )

    else:
        if args.resume_subdir is None:
            run_number = find_next_free_network_number(
                base_path=BASE_PATH,
                curriculum_type=CURRICULUM,
                task=TASK,
                affixes=AFFIXES,
                n_heads=len(Ns_init) if CURRICULUM == "cumulative" else 1,
                n_forget=NUM_FORGET,
            )

            print(f"=== Starting Standard Training (Run {run_number}) ===", flush=True)

        else:
            run_number = NETWORK_NUMBER

            print(
                f"=== Resuming Standard Training in: {args.resume_subdir} ===",
                flush=True,
            )

        if getattr(args, "aux_rnn_cb", False) and getattr(rnn, "cb", None) is not None:
            cb_param_ids = {id(p) for p in rnn.cb.parameters()}
            non_cb_params = [p for p in rnn.parameters() if id(p) not in cb_param_ids]
            cb_params = [p for p in rnn.parameters() if id(p) in cb_param_ids]
            optimizer = torch.optim.SGD(
                [{"params": non_cb_params, "lr": args.rnn_lr},
                {"params": cb_params, "lr": args.cb_lr}],
                momentum=0.1, nesterov=True,
            )
        else:
            optimizer = torch.optim.SGD(
                list(rnn.parameters()), lr=args.rnn_lr, momentum=0.1, nesterov=True,
            )

        non_cb_params = [p for p in rnn.parameters() if id(p) not in cb_param_ids]
        cb_params = [p for p in rnn.parameters() if id(p) in cb_param_ids]

        param_groups = []
        if non_cb_params:
            param_groups.append({"params": non_cb_params, "lr": args.rnn_lr})
        if cb_params:
            param_groups.append({"params": cb_params, "lr": args.cb_lr})

        optimizer = torch.optim.SGD(param_groups, momentum=0.1, nesterov=True)

        train(
            rnn,
            curriculum_type=CURRICULUM,
            task=TASK,
            num_epochs=NUM_EPOCHS,
            Ns=Ns_init,
            args=args,
            run_number=run_number,
            spec=spec,
            optimizer=optimizer,
            batch_size=BATCH_SIZE,
            training_steps=TRAINING_STEPS,
            test_steps=TEST_STEPS,
            device=device,
            base_path=BASE_PATH,
            affixes=AFFIXES,
            n_forget=NUM_FORGET,
            subdir_override=args.resume_subdir,
        )