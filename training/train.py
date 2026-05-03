import sys
import os
# Get the absolute path of the parent directory of 'src'
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
alex_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '../alex_crap'))
# Add both the parent directory and the 'src' directory to the module search path
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, alex_dir)

import torch
import torch.nn as nn

import numpy as np
import argparse
import re
from tqdm import tqdm

from src.models import RNN_Stack, RNN_Mod
import src.tasks as tasks
from src.tasks.task_registry import TASK_SPECS, compute_loss
from src.utils.save import save_model, find_next_free_network_number,make_unique_dir
from models_cb import ElmanRNNMultiHead
from multitask_impl import multitask_train, MULTITASK_TASKS
from continual_impl import continual_train
from task_switch_one import switch_train
from alex_training.train_alternating import train_alternating
from alex_training.rflo import init_rflo_state, rflo_step


def parse_optional_int(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return None
    return int(value)

def train(model,
          curriculum_type,
          task,
          num_epochs,
          Ns,
          args,
          run_number,
          spec,
          target_end_n=150,
          threshold_final=98.0,
          patience=3,
          subdir_override=None,
          stage_tag=None):

    stats = {
        'stage': [],
        'task': [],
        'n_task': [], 
        'phase': [], 
        'loss': [], 
        'accuracy': [], 
        'grad_rnn': [], 
        'grad_cb': [] 
    }

    losses = []
    accuracies = []

    task_function_local= spec["batch_fn"]
    criterion_local = spec["criterion_ctor"]()

    # save init (or reuse provided folder)
    if subdir_override is None:
        subdir = save_model(
            model,
            curriculum_type=curriculum_type,
            n_heads=len(Ns) if curriculum_type == 'cumulative' else 1,
            n_forget=NUM_FORGET,
            task=task,
            network_number=run_number,
            N_max=Ns[-1],
            N_min=Ns[0],
            init=True,
            args=args,
            base_path=BASE_PATH,
            affixes=AFFIXES
        )
    else:
        subdir = subdir_override
        os.makedirs(subdir, exist_ok=True)

    # Resume mode: append to existing stats if present.
    if getattr(args, 'resume_ckpt', None) and subdir_override is not None:
        stats_path = os.path.join(subdir, 'stats.npy')
        if os.path.exists(stats_path):
            try:
                old_stats = np.load(stats_path, allow_pickle=True).item()
                for k in stats.keys():
                    if k in old_stats and isinstance(old_stats[k], list):
                        stats[k] = list(old_stats[k])
                print(f"[resume] Loaded existing stats from {stats_path} (rows={len(stats['n_task'])})", flush=True)
            except Exception as e:
                print(f"[resume] Warning: failed to load existing stats ({e}); starting fresh stats in same folder.", flush=True)

    print(f"Saving results to: {subdir}", flush=True)

    # Train the model
    try:
        solved_streak = 0
        for epoch in tqdm(range(num_epochs)):
            losses_step = []
            grad_rnn_step = [] # Track grads per step
            frac_active_step = [] # Track fraction of active neurons per step
            
            for i in range(TRAINING_STEPS):

                task_args = (Ns, BATCH_SIZE)
                sequences, labels = task_function_local(*task_args)
                sequences = sequences.to(device)
                labels = [l.to(device) for l in labels]

                OPTIMIZER.zero_grad()
                
                _, out_heads = model(sequences, return_timewise=spec["timewise_output"])
                out_heads = out_heads[:len(Ns)] # Select only the heads corresponding to active Ns
                if not isinstance(out_heads, list):
                    out_heads = [out_heads]


                task_loss = compute_loss(out_heads, labels, spec["target_type"], criterion_local)

                if not torch.isfinite(task_loss):
                    print("Skipping non-finite batch loss", flush=True)
                    continue

                if args.rnn_eat and getattr(model, "_last_eat_loss", None) is not None:
                    loss = task_loss + args.rnn_eat_lambda * model._last_eat_loss
                else:
                    loss = task_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=7.5)  # DO NOT CHANGE
                OPTIMIZER.step()

                losses_step.append(float(loss.item()))
                total_grad = sum(
                    p.grad.data.norm(2).item() ** 2 
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                grad_rnn_step.append(total_grad)

            losses.append(np.mean(losses_step))
            
            # --- TESTING ---
            metric_accumulator = []
            for j in range(TEST_STEPS):
                with torch.no_grad():
                    task_args = (Ns, BATCH_SIZE)
                    sequences, labels = task_function_local(*task_args)
                    sequences = sequences.to(device)
                    labels = [l.to(device) for l in labels]

                    _, out_heads = model(sequences, return_timewise=spec["timewise_output"])
                    metric = spec["metric_fn"](out_heads, labels)
                    metric_accumulator.append(metric)

            # Aggregate metrics across test batches
            mean_score = float(np.mean([m["score"] for m in metric_accumulator]))
            accuracy = np.array([mean_score for _ in Ns])  # keep old var name for compatibility
            accuracies.append(accuracy)

            # Names for display
            loss_name = spec.get("loss_name", "loss")
            metric_name = metric_accumulator[0].get("name", "score") if len(metric_accumulator) > 0 else "score"

            # Safer epoch loss for printing (avoid poisoned running mean)
            finite_losses = [x for x in losses_step if np.isfinite(x)]
            mean_loss = float(np.nanmean(finite_losses)) if len(finite_losses) > 0 else float("nan")

            # Main status line
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{TRAINING_STEPS}] | "
                f"Loss ({loss_name}): {mean_loss:.4f} | "
                f"Performance ({metric_name}): {mean_score:.4f}",
                flush=True
            )
            # Per-head/task metric
            per_head_mean = (
                np.mean(np.array([m["per_head"] for m in metric_accumulator]), axis=0)
                if len(metric_accumulator) > 0 else []
            )

            print(
                f"Per-N Performance ({metric_name}):\n" +
                "".join([f"  N={Ns[idx]}: {per_head_mean[idx]:.4f}\n" for idx in range(len(Ns))]),
                flush=True
            )

            # Optional extra task-specific metrics (e.g. DCO endpoint error)
            if len(metric_accumulator) > 0 and "endpoint_error" in metric_accumulator[0]:
                mean_endpoint = float(np.mean([m["endpoint_error"] for m in metric_accumulator]))
                per_head_endpoint = np.mean(
                    np.array([m["per_head_endpoint_error"] for m in metric_accumulator]), axis=0
                )

                print(f"Aux metric (endpoint_error): {mean_endpoint:.4f}", flush=True)
                print(
                    "Per-N Aux metric (endpoint_error):\n" +
                    "".join([f"  N={Ns[idx]}: {per_head_endpoint[idx]:.4f}\n" for idx in range(len(Ns))]),
                    flush=True
                )
            if target_end_n is not None:
                if (Ns[-1] >= target_end_n) and (mean_score >= threshold_final):
                    solved_streak += 1
                else:
                    solved_streak = 0
                if solved_streak >= patience:
                    print(f"[STAGE DONE] task={task} reached target end n = {target_end_n} with "
                          f"score={mean_score:.2f} for {patience} evals. stopping stage", flush=True)
                    break
                    
            # --- UPDATE STATS ---
            stats['n_task'].append(Ns[-1])       # Log the hardest active N
            stats['phase'].append('RNN')    # Label it 'RNN'
            stats['loss'].append(np.mean(losses_step))
            stats['accuracy'].append(np.mean(accuracy))
            stats['grad_rnn'].append(np.mean(grad_rnn_step))
            stats['grad_cb'].append(0.0)         # Zero for baseline
            stats['stage'].append(stage_tag if stage_tag is not None else "")
            stats['task'].append(task)
            
            # Save Stats
            np.save(f'{subdir}/stats.npy', stats)

            # Curriculum Logic
            # Use task-specific curriculum criterion
            metric_for_curriculum = {
                "score": float(np.mean([m["score"] for m in metric_accumulator]))
            }
            # Carry optional auxiliary metrics if present (task-specific).
            # Exclude non-scalar/reporting keys.
            excluded_keys = {"score", "per_head", "name", "per_head_endpoint_error", "per_head_mse"}
            for k in metric_accumulator[0].keys():
                if k in excluded_keys:
                    continue
                vals = [m[k] for m in metric_accumulator if k in m]
                if len(vals) > 0:
                    metric_for_curriculum[k] = float(np.mean(vals))

            if spec["advance_fn"](metric_for_curriculum):
                if target_end_n is not None and Ns[-1] >= target_end_n:
                    pass # Don't advance if we've already reached the target end N, even if the criterion is met
                else:
                    old_Ns = list(Ns)
                    ct = str(curriculum_type).strip().lower()

                    print(f"[ADVANCE TRIGGERED] curriculum_type={repr(curriculum_type)} | ct={repr(ct)} | old_Ns={old_Ns}", flush=True)
                    print(f"Saving model for N = {old_Ns}...", flush=True)
                    if subdir_override is None:
                        save_model(
                            model,
                            curriculum_type=curriculum_type,
                            n_heads=len(Ns),
                            n_forget=NUM_FORGET,
                            task=task,
                            network_number=run_number,
                            N_max=Ns[-1],
                            N_min=Ns[0],
                            base_path=BASE_PATH,
                            args=args,
                            affixes=AFFIXES
                        )
                    else:
                        # Save a checkpoint inside the stage subdir (no new top-level folders)
                        ckpt_dir = os.path.join(subdir, "checkpoints")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        ckpt_path = os.path.join(
                            ckpt_dir,
                            f"{task}_{stage_tag if stage_tag else 'stage'}_ep{epoch+1}_N{Ns[-1]}.pt"
                        )
                        torch.save({"state_dict": model.state_dict()}, ckpt_path)

                    if ct == 'cumulative':
                        Ns = list(Ns) + [int(Ns[-1]) + 1 + i for i in range(NUM_ADD)]

                    elif ct == 'sliding':
                        Ns = list(Ns[NUM_FORGET:]) + [int(Ns[-1]) + 1 + i for i in range(NUM_FORGET)]

                    elif ct == 'single':
                        Ns = [int(Ns[0]) + 1]

                    elif ct == 'single_nocurr':
                        print("[ADVANCE] single_nocurr -> stopping", flush=True)
                        break

                    else:
                        raise ValueError(f"Unknown curriculum_type inside train(): {repr(curriculum_type)}")

                    print(f"[ADVANCED] {old_Ns} -> {Ns}", flush=True)
                    print(f"N = {Ns[0]}, {Ns[-1]}", flush=True)

    except KeyboardInterrupt:
        print(f"\n\nTraining Interrupted! Saving current stats to {subdir}/stats.npy...", flush=True)
        np.save(f'{subdir}/stats.npy', stats)
        return stats # Return what we have so far

    return stats


###############################################################

if __name__ == '__main__':

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('-b', '--base_path', type=str, dest='base_path',
                        help='The base path to save results. (str)')
    parser.add_argument('-nn', '--num_neurons', type=int, dest='num_neurons',
                        help='The number of hidden neurons in the RNN. (int)')
    parser.add_argument('-ni', '--ns_init', type=int, dest='ns_init',
                        help='The starting value of N for the task. (int)')
    parser.add_argument('-m', '--model_type', type=str, dest='model_type',
                        help='Model types: (default, mod,elman). (str)')
    parser.add_argument('-a', '--afunc', type=str, dest='afunc',
                        help='Acitvation functions: (leakyrelu, relu, tanh, sigmoid). (str)')
    parser.add_argument('-c', '--curriculum_type', type=str, dest='curriculum_type',
                        help='Curriculum type: (cumulative, sliding, single, single_nocurr). (str)')
    parser.add_argument('-t', '--task', type=str, dest='task',
                        help='Task: (parity, dms, threshold). (str)')
    parser.add_argument('-T', '--tau', type=float, dest='tau',
                        help='The value of tau each neuron starts with. If set, taus will not be trainable. '
                             'Default = None. (float > 1)')
    parser.add_argument('-n', '--network_number', type=int, dest='network_number',
                        help='The run number of the network, to be used as a naming suffix for savefiles. (int)')
    parser.add_argument('-ih', '--init_heads', type=int, dest='init_heads',
                        help='Number of heads to start with. (int)')
    parser.add_argument('-dh', '--add_heads', type=int, dest='add_heads',
                        help='Number of heads to add per new curricula. (int)')
    parser.add_argument('-fh', '--forget_heads', type=int, dest='forget_heads',
                        help='Number of heads to forget for the sliding window curriculum type. (int)')
    parser.add_argument('-s', '--seed', type=int, dest='seed',
                        help='Random seed. (int)')
    parser.add_argument('--no_cb', dest='use_cb_bias', action='store_false', default=True,
                    help='Disable Cerebellar Bias (for Baseline A)')
    parser.add_argument('--scramble', dest='scramble', action='store_true', default=False,
                    help='If True, the Cerebellum is never trained (outputs random crap).')
    parser.add_argument('--readout_mode', type=str, dest='readout_mode', default='single',
                    help='Readout dynamics: "sliding" or "cumulative" or "single". (str)')
    parser.add_argument('--cb_store', dest='cb_store', action='store_true', default=False,
                    help='If True, readout heads are stored/frozen in CB (Librarian Mode).')
    parser.add_argument('--rnn_lr', type=float, dest='rnn_lr', default=0.05,
                    help='Learning rate for the RNN module, defaults to 0.05. (float)')
    parser.add_argument('--cb_lr', type=float, dest='cb_lr', default=0.05,
                    help='Learning rate for the cerebellar module, defaults to 0.05 (same as RNN). (float)')
    parser.add_argument("--alt_variant", type=str, default="cb_then_rnn",
                    choices=["cb_then_rnn", "rnn_then_cb_finetune", "interleaved_finetune", "interleaved_curr", "cb_only", "train_simultaneous","cb_only_reservoir"],)
    parser.add_argument("--cb_reservoir_n", type=int, default=8,)
    parser.add_argument("--th_rnn_lead", type=float, default=85.0) # default is 85.0 in train_alternating
    parser.add_argument("--th_cb_lead", type=float, default=60.0) # default is 60.0 in train_alternating
    parser.add_argument("--no_shared_optimiser", dest='shared_optimiser', action='store_false', default=True,
                        help="Use separate optimisers with SGD for RNN and Adam for CB (instead of shared SGD optimiser).")
    parser.add_argument('--affixes', type=str,default='', help='Additional affixes to add to the save directory name, separated by underscores. (str)')
    parser.add_argument('--num_epochs', type=int, default=3000, help='Number of training epochs. Default is 3000. (int)')
    parser.add_argument('--learning_alg',type=str,default='bptt', help='Learning algorithm to use: "bptt" or "rflo". Default is "bptt". (str)')
    parser.add_argument('--multiply',default=False, action='store_true', help='Toggle whether to turn CB bias from additive to multiply')
    parser.add_argument('--cb_l2',default=0.0, type=float, help='L2 regularization strength for CB bias module (only relevant if use_cb_bias is True). Default is 0.0 (no regularization). (float)')
    parser.add_argument('--cb_l1',default=0.0, type=float, help='L1 regularization strength for CB bias module (only relevant if use_cb_bias is True). Default is 0.0 (no regularization). (float)')
    parser.add_argument('--rnn_eat',default=False, action='store_true', help='Whether to use CB bias as loss term to encourage the RNN to "eat its own tail" and internalize the CB bias (inspired by Hwang et al. 2023). Only relevant if use_cb_bias is True.')
    parser.add_argument('--rnn_eat_lambda',default=0.1,type=float,help='Lambd for the RNN eat CB loss term. Default is 0.1. Only relevant if rnn_eat is True. (float)')
    parser.add_argument("--task_switch", action="store_true", default=False,
                        help="Enable task switching schedule.")
    parser.add_argument("--task_switch_plan", type=str, default="dms,violation,dms",
                        help="Comma-separated task names.")
    parser.add_argument("--task_switch_targetN", type=int, default=13,
                        help="Target N per stage.")
    parser.add_argument("--use_alternating_in_switch", action="store_true", default=False,
                        help="If set, use train_alternating inside each stage (Elman CB models).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--training_steps", type=int, default=100)
    parser.add_argument("--test_steps", type=int, default=50)
    parser.add_argument("--rnn_eat_loss_type", type=str, default="hidden", choices=["task", "hidden"],
                        help="Whether the RNN eat loss should be computed on the task loss discrepancy between RNN alone and + CB(task) or on the MSE between hidden state trajectories. Default is 'hidden'. (str)")
    parser.add_argument('--cb_schedule',default=False, action='store_true', 
                        help='Whether to schedule CB learning rate by N.')
    parser.add_argument('--cb_lr_decay', default=0.02, type=float,
                        help='Decay factor for continuous CB LR schedule by N: lr = base_lr / (1 + decay * (N - N_start)).')
    parser.add_argument('--cb_lr_min_frac', default=0.05, type=float,
                        help='Minimum CB LR as a fraction of base CB LR when cb_schedule is enabled.')
    parser.add_argument('--debug_forward_stats', default=False, action='store_true',
                        help='Enable expensive per-time-step debug stat collection in model forward pass.')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Path to checkpoint to resume model weights from. Supports either raw state_dict or {"state_dict": ...}.')
    parser.add_argument('--resume_subdir', type=str, default=None,
                        help='Existing run directory to continue writing into (e.g., .../alternating_*_network_1).')
    parser.add_argument('--resume_start_n', type=int, default=None,
                        help='If set, override start N after loading checkpoint (useful to continue from solved N).')
    parser.add_argument('--no_resume_strict', dest='resume_strict', action='store_false', default=True,
                        help='Load checkpoint with strict=False instead of strict=True.')
    parser.add_argument('--skip_init_phase', action='store_true', default=False,
                        help='Alternating only: skip initial RNN-only base phase and start directly with alternating variant.')
    parser.add_argument('--gc_dim',type=int,default=512,help="size of granule cell layer in CB model. only relevant if cb enabled")
    parser.add_argument('--pc_dim',type=int,default=64,help="size of Purkinje cell layer in CB model. only relevant if cb enabled")
    parser.add_argument('--dcn_dim',type=int,default=64,help="size of DCN layer in CB model. only relevant if cb enabled")
    parser.add_argument('--cb_sees_input', action='store_true', default=False,
                        help='Whether the CB bias module should see the task input.')
    parser.add_argument("--multitask",action="store_true",default=False, 
                        help="Run multi-task training (DMS + parity + oddball simultaneously).")
    parser.add_argument("--mt_target_n", type=parse_optional_int, default=150,
                        help="Target N per task for multi-task curriculum."
                             " Use 'none' to disable the cap and run until num_epochs.")
    parser.add_argument("--mt_advance_threshold", type=float, default=98.0,
                        help="Accuracy threshold (percent) for multitask curriculum advancement."
                             " Set to 98.0 to match single-task behavior.")
    parser.add_argument("--mt_patience", type=int, default=1,
                        help="Advance patience for multitask curriculum.")
    parser.add_argument("--mt_reservoir_mode",type=str,default="off",choices=["off", "global_once", "all_tasks_once", "periodic_refresh"],
                        help="Reservoir mode for multitask curriculum: 'off' = normal training,'global_once' = one global N=2 warmup, " \
                        "'all_tasks_once' = each task gets one N=2 warmup, " \
                        "'periodic_refresh' = warmup once then train RNN+CB together every --reservoir_interval_n Ns.")
    parser.add_argument("--continual", action="store_true", default=False,
        help="Run continual learning experiment.")
    parser.add_argument("--continual_plan", type=str, default="dms,parity,dms",
        help="Comma-separated task sequence, repeats allowed. e.g. dms,parity,dms")
    parser.add_argument("--continual_epochs_per_phase", type=int, default=100,
        help="Epochs per phase in continual learning.")
    parser.add_argument("--continual_target_n", type=parse_optional_int, default=150,
        help="Target N per task in continual learning."
             " Use 'none' to disable the cap and run until epochs end.")
    parser.add_argument("--continual_patience", type=int, default=1,
        help="Advance patience for continual curriculum.")
    parser.add_argument("--ct_advance_threshold", type=float, default=98.0,
        help="Accuracy threshold (percent) for continual curriculum advancement.")
    parser.add_argument("--continual_savings_threshold", type=float, default=98.0,
        help="Accuracy threshold (percent) used by savings analysis for epochs-to-criterion.")
    parser.add_argument("--continual_reservoir_mode",type=str,default="off",choices=["off", "global_once", "per_task_once", "periodic_refresh"],
                        help="Reservoir mode for continual learning: 'off' = normal training,'global_once' = one global N=2 warmup, " \
                        "'per_task_once' = each task gets one N=2 warmup, " \
                        "'periodic_refresh' = warmup once then train RNN+CB together every --reservoir_interval_n Ns.")
    parser.add_argument("--reservoir_interval_n", type=int, default=10,
        help="When using periodic reservoir refresh mode, re-enable RNN every this many Ns.")
    parser.add_argument("--ct_switch", action="store_true", default=False,
        help="If set, use task switching schedule in continual learning instead of blockwise. Requires --continual_plan to have at least 2 tasks.")
    parser.add_argument("--switch_n", type=int, default=10,
        help="N at which to switch tasks in the task switching schedule for continual learning.")

    parser.set_defaults(
        model_type='default',
        num_neurons=500,
        # afunc='leakyrelu',
        curriculum_type='cumulative',
        task='parity',
        tau=None,
        network_number=1,
        init_heads=1,
        add_heads=1,
        forget_heads=1,
        seed=np.random.choice(2 ** 32),
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.gc_dim <= 0:
        raise ValueError(f"--gc_dim must be a positive integer, got {args.gc_dim}")

    # Access the values of the arguments
    print('num_neurons:', args.num_neurons)
    print('curriculum_type:', args.curriculum_type)
    print('task:', args.task)
    print('network number:', args.network_number)

    BASE_PATH = args.base_path
    NS_INIT = args.ns_init
    NUM_NEURONS = args.num_neurons
    AFFIXES = []

    # USER ARGUMENTS (curriculum type/task and related params)
    MODEL = args.model_type
    AFUNC = args.afunc
    CURRICULUM = args.curriculum_type
    TASK = args.task
    NETWORK_NUMBER = args.network_number
    TAU = args.tau
    if TAU is not None:
        TRAIN_TAU = False
    else:
        TAU = 1.5
        TRAIN_TAU = False

    INIT_HEADS = args.init_heads  # how many heads/tasks to start with
    NUM_ADD = args.add_heads  # how many heads/tasks to add per new curricula (only relevant for cumulative curriculum)
    NUM_FORGET = args.forget_heads  # how many heads to forget per new curricula (only relevant for sliding curriculum)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # Figure out task spec
    if TASK not in TASK_SPECS:
        raise ValueError(f"Unrecognized task: {TASK}")

    spec = TASK_SPECS[TASK]
    task_function = spec["batch_fn"]

    # Task-driven dims + behavior
    INPUT_SIZE = spec["input_size"]
    NUM_CLASSES = spec["output_size"]  
    TARGET_TYPE = spec["target_type"]
    TIMEWISE_OUTPUT = spec["timewise_output"]
    CRITERION = spec["criterion_ctor"]()
    METRIC_FN = spec["metric_fn"]
    ADVANCE_FN = spec["advance_fn"]

    START_N = spec.get("start_n", 2)

    # Set up the correct curriculum
    if CURRICULUM == 'cumulative':
        Ns_init = list(np.arange(START_N, START_N + INIT_HEADS))
    elif CURRICULUM == 'sliding':
        if INIT_HEADS < NUM_FORGET:
            INIT_HEADS = NUM_FORGET
        Ns_init = list(np.arange(START_N, START_N + INIT_HEADS))
    elif CURRICULUM == 'single' or CURRICULUM == 'single_nocurr' or CURRICULUM == 'alternating':
        Ns_init = [START_N]
        INIT_HEADS = 1
        NUM_FORGET = 1
    else:
        print('Unrecognized curriculum type: ', CURRICULUM)
    ###############################################################

    if NS_INIT is not None:
        Ns_init = [N - 2 + NS_INIT for N in Ns_init]

    def infer_resume_n_from_path(path):
        """Infer N from checkpoint naming conventions like *_N13.pt or rnn_N2_N13."""
        name = os.path.basename(str(path))
        n_tokens = re.findall(r'N(\d+)', name)
        if len(n_tokens) == 0:
            return None
        return int(n_tokens[-1])

    # MODEL PARAMS
    # INPUT_SIZE and NUM_CLASSES already set by task config above
    if CURRICULUM == 'cumulative':
        MAX_N = 150  # arbitrary large number to allow for many heads to be added
        NUM_READOUT_HEADS = MAX_N - START_N + 1
    elif CURRICULUM == 'sliding':
        NUM_READOUT_HEADS = INIT_HEADS
    elif CURRICULUM == 'single' or CURRICULUM == 'single_nocurr' or CURRICULUM == 'alternating':
        NUM_READOUT_HEADS = 1
    else:
        print('Unrecognized curriculum type: ', CURRICULUM)

    NET_SIZE = [NUM_NEURONS]
    BIAS = True
    TRAIN_TAU = True

    if AFUNC == 'leakyrelu':
        AFUNC = nn.LeakyReLU
    elif AFUNC == 'relu':
        AFUNC = nn.ReLU
    elif AFUNC == 'sigmoid':
        AFUNC = nn.Sigmoid
    elif AFUNC == 'tanh':
        AFUNC = nn.Tanh
    else:
        print('Unrecognized activation function: ', AFUNC)

    # TRAINING PARAMS
    NUM_EPOCHS = args.num_epochs 
    BATCH_SIZE = args.batch_size
    TRAINING_STEPS = args.training_steps
    TEST_STEPS = args.test_steps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_ALG = args.learning_alg

    # init new model
    if MODEL == 'mod':
        AFFIXES += ['mod', AFUNC]
        if args.tau is not None:
            AFFIXES += ['T', str(TAU)]

        if AFUNC == 'leakyrelu':
            AFUNC = nn.LeakyReLU
        elif AFUNC == 'relu':
            AFUNC = nn.ReLU
        elif AFUNC == 'sigmoid':
            AFUNC = nn.Sigmoid
        elif AFUNC == 'tanh':
            AFUNC = nn.Tanh
        else:
            print('Unrecognized activation function: ', AFUNC)

        rnn = RNN_Mod(
            input_size=INPUT_SIZE,
            net_size=NET_SIZE,
            num_classes=NUM_CLASSES,
            bias=BIAS,
            num_readout_heads=NUM_READOUT_HEADS,
            tau=TAU,
            afunc=AFUNC,
            train_tau=TRAIN_TAU,
        ).to(device)

    elif MODEL == 'default':

        if NUM_NEURONS != 500:
            AFFIXES += ['size', str(NUM_NEURONS)]
        if args.tau is not None:
            AFFIXES += ['T', str(TAU)]

        rnn = RNN_Stack(
            input_size=INPUT_SIZE,
            net_size=NET_SIZE,
            num_classes=NUM_CLASSES,
            bias=BIAS,
            num_readout_heads=NUM_READOUT_HEADS,
            tau=TAU,
            train_tau=TRAIN_TAU,
        ).to(device)

        rnn.to(device)

    elif MODEL == 'elman':
        AFFIXES += ['elman']
        if NUM_NEURONS != 500:
            AFFIXES += ['size', str(NUM_NEURONS)]
        if args.tau is not None:
            AFFIXES += ['T', str(TAU)]
        if args.use_cb_bias:
            AFFIXES += ['CB']
            if args.gc_dim:
                AFFIXES += [f'gc{args.gc_dim}']
            if args.pc_dim != 64:
                AFFIXES += [f'pc{args.pc_dim}']
            if args.dcn_dim != 64:
                AFFIXES += [f'dcn{args.dcn_dim}']
        else:
            AFFIXES += ['noCB']
        if args.scramble:
            AFFIXES += ['SCRAMBLED']
        if args.readout_mode == 'sliding':
            AFFIXES += ['sliding']
        elif args.readout_mode == 'cumulative':
            AFFIXES += ['cumulative']
        if args.cb_store:
            AFFIXES += ['CBstore']
        if args.rnn_lr != 0.05:
            AFFIXES += [f'RNNlr{args.rnn_lr}']
        if args.cb_lr != 0.05:
            if args.cb_schedule:
                AFFIXES += [f'CBlr{args.cb_lr}sched']
            else:
                AFFIXES += [f'CBlr{args.cb_lr}']
        if args.cb_sees_input:
            AFFIXES += ['CBinput']
        if args.alt_variant == 'rnn_then_cb_finetune':
            AFFIXES += ['base_fin']
        elif args.alt_variant == 'interleaved_finetune':
            AFFIXES += ['int_fin']
        elif args.alt_variant == 'interleaved_curr':
            AFFIXES += ['int_curr']
        elif args.alt_variant == 'cb_only':
            AFFIXES += ['cb_only']
        elif args.alt_variant == 'cb_only_reservoir':
            AFFIXES += ['interleaved_res']
        elif args.alt_variant == 'train_simultaneous':
            AFFIXES += ['simult']
        if args.shared_optimiser == False:
            AFFIXES += ['opts']
        if args.affixes:
            AFFIXES += [args.affixes]
        if args.learning_alg != 'bptt':
            AFFIXES += [args.learning_alg]
        if args.multiply:
            AFFIXES += ['mult']
        if args.cb_l2 > 0.0:
            AFFIXES += [f'cbl2_{args.cb_l2}']
        if args.cb_l1 > 0.0:
            AFFIXES += [f'cbl1{args.cb_l1}']
        if args.rnn_eat:
            AFFIXES += [f'yum{args.rnn_eat_lambda}']
            if args.rnn_eat_loss_type != 'hidden':
                AFFIXES += [f'{args.rnn_eat_loss_type}']
        
        if args.multitask:
            NUM_TASKS = len(MULTITASK_TASKS)           # 3                
            rnn = ElmanRNNMultiHead(
                input_size=INPUT_SIZE,
                hidden_size=NUM_NEURONS,
                cb_gc_dim=args.gc_dim,
                cb_pc_dim=args.pc_dim,
                cb_dcn_dim=args.dcn_dim,
                num_classes=NUM_CLASSES,               # still 2 (binary output per task)
                num_readout_heads=NUM_TASKS,           # one head per task
                tau=TAU,
                afunc=AFUNC,
                bias=BIAS,
                use_cb_bias=args.use_cb_bias,
                multiply=args.multiply,
                rnn_eat=args.rnn_eat,
                cb_input_size=INPUT_SIZE if args.cb_sees_input else 0,           # CB sees full input incl. task ID
                rnn_eat_lambda=args.rnn_eat_lambda,
                debug_stats=args.debug_forward_stats,
                train_tau=False,
            ).to(device)
        elif args.continual or args.ct_switch:
            plan = [t.strip() for t in args.continual_plan.split(",") if t.strip()]
            unique_tasks = list(dict.fromkeys(plan))   # preserves order, deduplicates
            NUM_CONT_TASKS = len(unique_tasks)
            INPUT_SIZE += 1
        
            rnn = ElmanRNNMultiHead(
                input_size        = INPUT_SIZE,         # add 1 for task ID input in continual setting
                hidden_size       = NUM_NEURONS,
                cb_gc_dim         = args.gc_dim,
                cb_pc_dim         = args.pc_dim,
                cb_dcn_dim        = args.dcn_dim,
                num_classes       = NUM_CLASSES,        # 2
                num_readout_heads = NUM_CONT_TASKS,
                tau               = TAU,
                afunc             = AFUNC,
                bias              = BIAS,
                use_cb_bias       = args.use_cb_bias,
                multiply          = args.multiply,
                rnn_eat           = args.rnn_eat,
                cb_input_size     = INPUT_SIZE if args.cb_sees_input else 0,
                rnn_eat_lambda    = args.rnn_eat_lambda,
                debug_stats       = args.debug_forward_stats,
                train_tau         = False,
            ).to(device)
        else:
            rnn = ElmanRNNMultiHead(
                input_size=INPUT_SIZE,
                hidden_size=NUM_NEURONS,
                cb_gc_dim=args.gc_dim,
                cb_pc_dim=args.pc_dim,
                cb_dcn_dim=args.dcn_dim,
                num_classes=NUM_CLASSES,
                num_readout_heads=NUM_READOUT_HEADS,
                tau=TAU,
                scramble=args.scramble,
                afunc=AFUNC,
                bias=BIAS,
                use_cb_bias=args.use_cb_bias,
                multiply=args.multiply,
                rnn_eat=args.rnn_eat,
                cb_input_size=INPUT_SIZE if args.cb_sees_input else 0,
                rnn_eat_lambda=args.rnn_eat_lambda,
                debug_stats=args.debug_forward_stats,
                train_tau=False,
            ).to(device)

        rnn.to(device)
    else:
        print('Unrecognized model type: ', MODEL)

    # Optional resume: load model weights before any training starts.
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        rnn.load_state_dict(sd, strict=args.resume_strict)
        print(f"[resume] Loaded checkpoint: {args.resume_ckpt} (strict={args.resume_strict})", flush=True)

        if args.resume_subdir is None:
            inferred_subdir = os.path.dirname(args.resume_ckpt)
            # If checkpoint is inside a checkpoints/ folder, write into its parent run folder.
            if os.path.basename(inferred_subdir) == 'checkpoints':
                inferred_subdir = os.path.dirname(inferred_subdir)
            args.resume_subdir = inferred_subdir
        print(f"[resume] Writing outputs into existing run dir: {args.resume_subdir}", flush=True)

        resume_n = args.resume_start_n
        if resume_n is None:
            resume_n = infer_resume_n_from_path(args.resume_ckpt)

        if resume_n is not None:
            if CURRICULUM in ['single', 'single_nocurr', 'alternating']:
                Ns_init = [int(resume_n)]
            elif CURRICULUM in ['cumulative', 'sliding']:
                Ns_init = list(np.arange(int(resume_n), int(resume_n) + INIT_HEADS))
            print(f"[resume] Starting curriculum from N={resume_n} -> Ns_init={Ns_init}", flush=True)
        else:
            print("[resume] Could not infer N from checkpoint name; keeping current Ns_init.", flush=True)

    # TASK SWITCHING MODE
    if args.task_switch:
        plan = [t.strip() for t in args.task_switch_plan.split(',') if t.strip()]
        if len(plan) == 0:
            raise ValueError("task_switch_plan is empty")
        # Validate tasks exist
        for t in plan:
            if t not in TASK_SPECS:
                raise ValueError(
                    f"Unrecognized task in task switch plan: {t} "
                    f"(available tasks: {list(TASK_SPECS.keys())})"
                )
        print(f"\n\n=== TASK SWITCH ENABLED === plan={plan} ===\n\n", flush=True)
        # auto-route CB models through train_alternating unless user explicitly disables
        if MODEL == "elman" and args.use_cb_bias:
            args.use_alternating_in_switch = True
        # ---- block schedule defaults (edit these) ----
        BLOCK_STEP = 10            # block targets: 5,10,15,...
        BLOCK_END = 10             # starting shared target
        MAX_BLOCK_END = int(args.task_switch_targetN)  # hard cap (e.g. 25)
        MIN_STAGE_STARTN = 2      # safety floor for starting N per task
        # One persistent optimiser across all stages (keeps momentum etc.)
        momentum = 0.1
        OPTIMIZER = torch.optim.SGD(
            list(rnn.parameters()),
            lr=args.rnn_lr,
            momentum=momentum,
            nesterov=True
        )
        # ---- create ONE base folder for the entire switch run ----
        switch_affixes = AFFIXES + ["SWITCH"]
        run_number = find_next_free_network_number(
            base_path=BASE_PATH,
            curriculum_type="task_switch",
            task="switch",
            affixes=switch_affixes,
            n_heads=1,
            n_forget=NUM_FORGET
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
            affixes=switch_affixes
        )
        print(f"[task_switch] Base folder: {base_subdir}", flush=True)

        # ---- repeat the plan as a cycle until we reach the cap ----
        cycle = plan[:]  # e.g. ["dms", "violation", "dms"] (you can just pass "dms,violation")
        BLOCK_TASKS = sorted(list(set(cycle)))  # tasks that must complete each block
        done_this_block = set()
        last_N_by_task = {}

        stage_i = 0
        while True:
            # stop condition: once we've hit the cap AND completed that cap block for all tasks
            if BLOCK_END >= MAX_BLOCK_END and set(done_this_block) == set(BLOCK_TASKS) and stage_i > 0:
                break

            stage_task = cycle[stage_i % len(cycle)]
            spec_stage = TASK_SPECS[stage_task]
            task_fn_stage = spec_stage["batch_fn"]

            default_startN = int(spec_stage.get("start_n", 2))
            startN = int(last_N_by_task.get(stage_task, default_startN))
            startN = max(startN, MIN_STAGE_STARTN)

            stage_endN = int(min(BLOCK_END, MAX_BLOCK_END))

            if startN >= stage_endN:
                done_this_block.add(stage_task)
                print(
                    f"[STAGE {stage_i}] task={stage_task} already at startN={startN} >= block_target={stage_endN} "
                    f"(done_this_block={sorted(done_this_block)})",
                    flush=True
                )
            else:
                Ns_stage = [startN]
                stage_dir = make_unique_dir(os.path.join(base_subdir, f"switch{stage_i}_{stage_task}"))

                print(
                    f"[STAGE {stage_i}] task={stage_task} startN={startN} -> target_endN={stage_endN} "
                    f"(BLOCK_END={BLOCK_END}, cap={MAX_BLOCK_END}) dir={stage_dir}",
                    flush=True
                )

                if args.use_alternating_in_switch:
                    _ = train_alternating(
                        model=rnn,
                        curriculum_type="alternating",
                        task_name=stage_task,
                        task_function=task_fn_stage,
                        num_epochs=NUM_EPOCHS,
                        Ns_init=Ns_stage,
                        run_number=0,
                        batch_size=BATCH_SIZE,
                        training_steps=TRAINING_STEPS,
                        test_steps=TEST_STEPS,
                        device=device,
                        base_path=BASE_PATH,
                        affixes=AFFIXES + [f"switch{stage_i}", stage_task],
                        scramble=args.scramble,
                        rnn_lr=args.rnn_lr,
                        cb_lr=args.cb_lr,
                        readout_head_dyn=args.readout_mode,
                        cb_store=args.cb_store,
                        n_heads=1,
                        n_forget=NUM_FORGET,
                        alt_variant=args.alt_variant,
                        args=args,
                        shared_optimiser=args.shared_optimiser,
                        threshold_cb_lead=args.th_cb_lead,
                        threshold_rnn_lead=args.th_rnn_lead,
                        threshold_final=98.0,
                        learning_alg=args.learning_alg,
                        spec=spec_stage,
                        cb_input_size=INPUT_SIZE if args.cb_sees_input else 0,
                        cb_l2=args.cb_l2,
                        cb_l1=args.cb_l1,
                        rnn_eat=args.rnn_eat,
                        rnn_eat_lambda=args.rnn_eat_lambda,
                        rnn_eat_loss_type=args.rnn_eat_loss_type,
                        cb_schedule=args.cb_schedule,
                        cb_lr_decay=args.cb_lr_decay,
                        cb_lr_min_frac=args.cb_lr_min_frac,
                        skip_init_phase=args.skip_init_phase,
                        target_end_n=stage_endN,
                        subdir_override=stage_dir,
                        stage_tag=f"switch{stage_i}"
                    )
                else:
                    _ = train(
                        rnn,
                        curriculum_type="single",
                        task=stage_task,
                        num_epochs=NUM_EPOCHS,
                        Ns=Ns_stage,
                        args=args,
                        run_number=0,
                        spec=spec_stage,
                        target_end_n=stage_endN,
                        threshold_final=98.0,
                        patience=3,
                        subdir_override=stage_dir,
                        stage_tag=f"switch{stage_i}"
                    )

                st = np.load(os.path.join(stage_dir, "stats.npy"), allow_pickle=True).item()
                lastN = int(st["n_task"][-1])
                last_N_by_task[stage_task] = lastN

                if lastN >= stage_endN:
                    done_this_block.add(stage_task)

                print(
                    f"[STAGE {stage_i} DONE] task={stage_task} lastN={lastN} block_target={stage_endN} "
                    f"done_this_block={sorted(done_this_block)}",
                    flush=True
                )

            if set(done_this_block) == set(BLOCK_TASKS):
                if BLOCK_END < MAX_BLOCK_END:
                    BLOCK_END = min(BLOCK_END + BLOCK_STEP, MAX_BLOCK_END)
                    print(f"[BLOCK ADVANCE] All tasks reached block. New BLOCK_END={BLOCK_END}", flush=True)
                    done_this_block = set()
                else:
                    # FINAL BLOCK COMPLETE — exit here instead of resetting
                    print("[task_switch] Final block complete. Stopping.", flush=True)
                    break  

            stage_i += 1

        print("[task_switch] COMPLETE", flush=True)
        sys.exit(0)

    if args.multitask:
        # Validate tasks
        for t in MULTITASK_TASKS:
            if t not in TASK_SPECS:
                raise ValueError(f"Task '{t}' missing from TASK_SPECS")
    
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
    
        cb_param_ids = set()
        if getattr(rnn, "cb", None) is not None:
            cb_param_ids = {id(p) for p in rnn.cb.parameters()}

        def optimizer_ctor(params):
            params = list(params)

            cb_params = [p for p in params if id(p) in cb_param_ids]
            non_cb_params = [p for p in params if id(p) not in cb_param_ids]

            param_groups = []
            if len(non_cb_params) > 0:
                param_groups.append({"params": non_cb_params, "lr": args.rnn_lr})
            if len(cb_params) > 0:
                param_groups.append({"params": cb_params, "lr": args.cb_lr})

            return torch.optim.SGD(
                param_groups,
                momentum=0.1,
                nesterov=True,
            )

        multitask_train(
            model             = rnn,
            task_specs        = TASK_SPECS,
            optimizer_ctor    = optimizer_ctor,
            num_epochs     = NUM_EPOCHS,
            batch_size     = BATCH_SIZE,
            training_steps = TRAINING_STEPS,
            test_steps     = TEST_STEPS,
            device         = device,
            subdir         = mt_subdir,
            advance_patience = args.mt_patience,
            target_end_n   = args.mt_target_n,
            rnn_eat        = args.rnn_eat,
            rnn_eat_lambda = args.rnn_eat_lambda,
            task_names     = MULTITASK_TASKS,
            mt_advance_threshold = args.mt_advance_threshold,
            reservoir_mode  = args.mt_reservoir_mode,
            reservoir_interval_n = args.reservoir_interval_n,
        )
        sys.exit(0)
    elif args.continual:
        plan = [t.strip() for t in args.continual_plan.split(",") if t.strip()]
        for t in plan:
            if t not in TASK_SPECS:
                raise ValueError(f"Task '{t}' not in TASK_SPECS")
    
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
        
        cb_param_ids = set()
        if getattr(rnn, "cb", None) is not None:
            cb_param_ids = {id(p) for p in rnn.cb.parameters()}

        def optimizer_ctor(params):
            params = list(params)

            cb_params = [p for p in params if id(p) in cb_param_ids]
            non_cb_params = [p for p in params if id(p) not in cb_param_ids]

            param_groups = []
            if len(non_cb_params) > 0:
                param_groups.append({"params": non_cb_params, "lr": args.rnn_lr})
            if len(cb_params) > 0:
                param_groups.append({"params": cb_params, "lr": args.cb_lr})

            return torch.optim.SGD(
                param_groups,
                momentum=0.1,
                nesterov=True,
            )
            
        continual_train(
            model                 = rnn,
            task_specs            = TASK_SPECS,
            optimizer_ctor        = optimizer_ctor,
            plan                  = plan,
            epochs_per_phase      = args.continual_epochs_per_phase,
            max_global_epochs     = NUM_EPOCHS,
            batch_size            = BATCH_SIZE,
            training_steps        = TRAINING_STEPS,
            test_steps            = TEST_STEPS,
            device                = device,
            subdir                = cont_subdir,
            target_end_n          = args.continual_target_n,
            rnn_eat               = args.rnn_eat,
            rnn_eat_lambda        = args.rnn_eat_lambda,
            advance_patience      = args.continual_patience,
            ct_advance_threshold  = args.ct_advance_threshold,
            savings_threshold     = args.continual_savings_threshold,
            reservoir_mode        = args.continual_reservoir_mode,
            reservoir_interval_n  = args.reservoir_interval_n,
        )
        sys.exit(0)
    elif args.ct_switch:
        plan = [t.strip() for t in args.continual_plan.split(",") if t.strip()]
        for t in plan:
            if t not in TASK_SPECS:
                raise ValueError(f"Task '{t}' not in TASK_SPECS")

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

        cb_param_ids = set()
        if getattr(rnn, "cb", None) is not None:
            cb_param_ids = {id(p) for p in rnn.cb.parameters()}

        def optimizer_ctor(params):
            params = list(params)
            cb_params = [p for p in params if id(p) in cb_param_ids]
            non_cb_params = [p for p in params if id(p) not in cb_param_ids]

            param_groups = []
            if len(non_cb_params) > 0:
                param_groups.append({"params": non_cb_params, "lr": args.rnn_lr})
            if len(cb_params) > 0:
                param_groups.append({"params": cb_params, "lr": args.cb_lr})

            return torch.optim.SGD(
                param_groups,
                momentum=0.1,
                nesterov=True,
            )

        switch_train(
            model                 = rnn,
            task_specs            = TASK_SPECS,
            optimizer_ctor        = optimizer_ctor,
            plan                  = plan,
            epochs_per_phase      = args.continual_epochs_per_phase,
            max_global_epochs     = NUM_EPOCHS,
            batch_size            = BATCH_SIZE,
            training_steps        = TRAINING_STEPS,
            test_steps            = TEST_STEPS,
            device                = device,
            subdir                = switch_subdir,
            target_end_n          = args.continual_target_n,
            rnn_eat               = args.rnn_eat,
            rnn_eat_lambda        = args.rnn_eat_lambda,
            advance_patience      = args.continual_patience,
            ct_advance_threshold  = args.ct_advance_threshold,
            savings_threshold     = args.continual_savings_threshold,
            reservoir_mode        = args.continual_reservoir_mode,
            reservoir_interval_n  = args.reservoir_interval_n,
            switch_n              = args.switch_n,
        )
        sys.exit(0)

    # --- MODIFIED TRAINING BLOCK ---
    if CURRICULUM == 'alternating':
        stats = train_alternating(
            model=rnn,
            curriculum_type=CURRICULUM,
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
            scramble=args.scramble,
            readout_head_dyn=args.readout_mode,
            cb_store=args.cb_store,
            n_heads = INIT_HEADS,
            n_forget = NUM_FORGET,
            cb_lr = args.cb_lr,
            rnn_lr = args.rnn_lr,
            args=args,
            alt_variant = args.alt_variant,
            threshold_rnn_lead = args.th_rnn_lead,
            threshold_cb_lead = args.th_cb_lead,
            shared_optimiser = args.shared_optimiser,
            threshold_final = 98.0,
            spec=spec,
            cb_input_size=INPUT_SIZE if args.cb_sees_input else 0,
            learning_alg = args.learning_alg,
            cb_l2 = args.cb_l2,
            cb_l1 = args.cb_l1,
            rnn_eat = args.rnn_eat,
            rnn_eat_lambda = args.rnn_eat_lambda,
            rnn_eat_loss_type = args.rnn_eat_loss_type,
            cb_schedule = args.cb_schedule,
            cb_lr_decay = args.cb_lr_decay,
            cb_lr_min_frac = args.cb_lr_min_frac,
            skip_init_phase = args.skip_init_phase,
            reservoir_interval_n = args.cb_reservoir_n,
            subdir_override = args.resume_subdir,
            affixes=AFFIXES
        )
    else:
        # This ensures we don't overwrite previous baseline runs unless user asked to resume in-place.
        if args.resume_subdir is None:
            run_number = find_next_free_network_number(
                base_path=BASE_PATH,
                curriculum_type=CURRICULUM,
                task=TASK,
                affixes=AFFIXES,
                n_heads=len(Ns_init) if CURRICULUM == 'cumulative' else 1,
                n_forget=NUM_FORGET
            )
            print(f"=== Starting Standard Training (Run {run_number}) ===", flush=True)
        else:
            run_number = NETWORK_NUMBER
            print(f"=== Resuming Standard Training in existing dir: {args.resume_subdir} ===", flush=True)

        # Standard Training
        momentum = 0.1
        # make readout head have lower learning rate to prevent instability (especially important for elman with CB bias, which can have large initial losses)
        rnn_params = [p for p in rnn.parameters()]
        OPTIMIZER = torch.optim.SGD(rnn_params, lr=args.rnn_lr,
            momentum=momentum,
            nesterov=True
        )        
        stats = train(
            rnn,
            curriculum_type=CURRICULUM,
            task=TASK,
            num_epochs=NUM_EPOCHS,
            Ns=Ns_init,
            args=args,
            run_number=run_number,
            spec=spec,
            subdir_override=args.resume_subdir,
        )