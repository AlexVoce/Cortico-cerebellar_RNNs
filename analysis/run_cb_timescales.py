from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cb_timescales_utils import (
    analyze_checkpoint_timescales,
    find_available_Ns_legacy,
    find_multitask_checkpoints,
    load_state_dict_from_checkpoint,
    load_state_dict_legacy,
    make_random_binary_input,
    make_random_multitask_input,
    save_timescale_result,
    set_cpu_threads,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to one run directory containing config.json and checkpoints.")
    parser.add_argument("--mode", type=str, choices=["legacy_single", "multitask_pt"], required=True,
                        help="legacy_single: rnn_N{N}_N{N}[.pt] checkpoints; multitask_pt: checkpoints/*.pt")
    parser.add_argument("--Ns", type=str, default=None,
                        help="Comma-separated Ns to analyze, e.g. '2,5,10'. If omitted, analyze all available.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save results. Defaults inside run_dir/timescales/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--T", type=int, default=20000,
                        help="Total simulated timesteps including burn-in.")
    parser.add_argument("--burn_T", type=int, default=500,
                        help="Burn-in timesteps removed before AC computation.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of parallel sequences / trials.")
    parser.add_argument("--max_lag", type=int, default=200)
    parser.add_argument("--fit_lag", type=int, default=30)
    parser.add_argument("--input_mode", type=str, choices=["single_binary", "multitask_binary"], default=None,
                        help="If omitted: legacy_single -> single_binary, multitask_pt -> multitask_binary.")
    parser.add_argument("--omp_threads", type=str, default="2")
    parser.add_argument("--blas_threads", type=str, default="2")
    parser.add_argument("--torch_threads", type=int, default=10)

    return parser.parse_args()


def parse_Ns_arg(Ns_arg):
    if Ns_arg is None or str(Ns_arg).strip() == "":
        return None
    return [int(x.strip()) for x in str(Ns_arg).split(",") if x.strip()]


def get_input_builder(input_mode: str):
    if input_mode == "single_binary":
        return lambda T, B, device: make_random_binary_input(T, B, input_size=1, device=device)
    elif input_mode == "multitask_binary":
        return lambda T, B, device: make_random_multitask_input(T, B, device=device)
    else:
        raise ValueError(f"Unknown input_mode={input_mode}")


def main():
    args = parse_args()
    set_cpu_threads(
        omp_threads=args.omp_threads,
        blas_threads=args.blas_threads,
        torch_threads=args.torch_threads,
    )

    run_dir = Path(args.run_dir)
    if args.save_dir is None:
        save_dir = run_dir / "timescales"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    requested_Ns = parse_Ns_arg(args.Ns)

    if args.input_mode is None:
        input_mode = "single_binary" if args.mode == "legacy_single" else "multitask_binary"
    else:
        input_mode = args.input_mode

    input_builder = get_input_builder(input_mode)

    if args.mode == "legacy_single":
        available_Ns = find_available_Ns_legacy(run_dir)
        Ns_to_run = available_Ns if requested_Ns is None else [N for N in requested_Ns if N in available_Ns]

        if len(Ns_to_run) == 0:
            raise RuntimeError("No matching Ns found for legacy_single mode.")

        print(f"Found legacy Ns: {available_Ns}")
        print(f"Running Ns: {Ns_to_run}")

        for N in Ns_to_run:
            print(f"\n=== Analyzing legacy checkpoint N={N} ===")
            sd = load_state_dict_legacy(run_dir, N, device=args.device)

            result = analyze_checkpoint_timescales(
                run_dir=run_dir,
                state_dict=sd,
                T=args.T,
                batch_size=args.batch_size,
                burn_T=args.burn_T,
                max_lag=args.max_lag,
                fit_lag=args.fit_lag,
                device=args.device,
                input_builder=input_builder,
            )

            result["N"] = N
            result["checkpoint_type"] = "legacy_single"

            save_path = save_dir / f"timescales_N{N}.pkl"
            save_timescale_result(result, save_path)
            print(f"Saved -> {save_path}")

    elif args.mode == "multitask_pt":
        ckpts = find_multitask_checkpoints(run_dir)
        available_Ns = [c["N"] for c in ckpts]
        Ns_to_run = available_Ns if requested_Ns is None else [N for N in requested_Ns if N in available_Ns]

        if len(Ns_to_run) == 0:
            raise RuntimeError("No matching Ns found for multitask_pt mode.")

        print(f"Found multitask Ns: {available_Ns}")
        print(f"Running Ns: {Ns_to_run}")

        ckpt_by_N = {c["N"]: c for c in ckpts}

        for N in Ns_to_run:
            ck = ckpt_by_N[N]
            print(f"\n=== Analyzing multitask checkpoint N={N} | {ck['name']} ===")
            sd = load_state_dict_from_checkpoint(ck["path"], device=args.device)

            result = analyze_checkpoint_timescales(
                run_dir=run_dir,
                state_dict=sd,
                T=args.T,
                batch_size=args.batch_size,
                burn_T=args.burn_T,
                max_lag=args.max_lag,
                fit_lag=args.fit_lag,
                device=args.device,
                input_builder=input_builder,
            )

            result["N"] = N
            result["epoch"] = ck["epoch"]
            result["checkpoint_name"] = ck["name"]
            result["checkpoint_type"] = "multitask_pt"

            save_path = save_dir / f"timescales_N{N}.pkl"
            save_timescale_result(result, save_path)
            print(f"Saved -> {save_path}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()