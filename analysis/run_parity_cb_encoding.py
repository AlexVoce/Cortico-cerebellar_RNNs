"""Parity CB-computation analysis (decoding, CB geometry, encoding models, causal
swaps, lookup). parity_cb_swap_scores.csv is used in Fig 6c-d.

Run with: python -m analysis.run_parity_cb_encoding
"""
import os
import time
import numpy as np
import pandas as pd

from analysis.parity_trials import (
    RNN_RUNS, CB_RUNS, HIDDEN_SIZES, load_model, make_fixed_parity_trials, shared_available_Ns,
)
from analysis.predictive_analysis import _select_Ns
from analysis.parity_dynamics import (
    collect_parity_dynamics, extract_transitions, build_reference_geometry,
)
from analysis.parity_update_rule import (
    verify_sliding_window_update, pool_full_variables_by_t_before_end,
    decode_variable_over_position, compute_group_id, GROUP_LABELS,
)
from analysis.parity_cb_encoding import (
    condition_geometry, encoding_model_r2, new_causal_swap_conditions, lookup_quantification,
)

OUT_DIR = "results/mechanistic_analysis/parity_cb_encoding"
DATA_DIR = OUT_DIR

BATCH_SIZE = 64
N_BATCHES = 8
MAX_NS_MECH = None  # None = every shared checkpoint
EVAL_SEED = 24680

DEVICE = "cpu"
PCA_EXAMPLE_SEED = 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [parity_cb] {msg}", flush=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    log("Part 0: verifying sliding-window XOR update rule against the real task generator...")
    shared_Ns = shared_available_Ns(RNN_RUNS + CB_RUNS)
    Ns_use = _select_Ns(shared_Ns, max_Ns=MAX_NS_MECH)
    verify_df = verify_sliding_window_update(Ns_use)
    verify_df.to_csv(os.path.join(DATA_DIR, "parity_cb_variable_verification.csv"), index=False)
    log("\n" + verify_df.to_string(index=False))
    assert (verify_df["xor3_accuracy"] > 0.999).all(), (
        "XOR-3 sliding-window update rule failed to verify -- refusing to proceed with "
        "Parts 2-6, which assume this indexing for outgoing_bit."
    )
    assert (verify_df["naive_accuracy"] - 0.5).abs().max() < 0.02, (
        "naive 2-state rule unexpectedly deviates from chance -- re-check before proceeding."
    )
    log(f"Representative Ns for Parts 1-6: {Ns_use}")

    decode_rows, geom_summary_rows, geom_pair_rows = [], [], []
    encoding_rows, causal_rows, margin_rows, lookup_rows = [], [], [], []
    pca_example_saved = False
    pca_target_N = Ns_use[len(Ns_use) // 2]

    for arch, runs in [("rnn_only", RNN_RUNS), ("cb_rnn", CB_RUNS)]:
        for seed_idx, run_path in enumerate(runs, start=1):
            for N in Ns_use:
                model = load_model(run_path, N, device=DEVICE)
                batches = make_fixed_parity_trials(N, batch_size=BATCH_SIZE, n_batches=N_BATCHES, seed=EVAL_SEED)
                per_batch = collect_parity_dynamics(model, batches, device=DEVICE)

                # Part 1: decode incoming/outgoing/next parity
                by_pos = pool_full_variables_by_t_before_end(per_batch, N)
                for var in ["incoming_bit", "outgoing_bit", "next_parity"]:
                    auc_h = decode_variable_over_position(by_pos, var, feature_key="hidden", seed=EVAL_SEED)
                    if model.use_cb_bias:
                        auc_cb = decode_variable_over_position(by_pos, var, feature_key="cb_bias", seed=EVAL_SEED)
                    else:
                        auc_cb = np.full(len(by_pos), np.nan)
                    for t in range(len(by_pos)):
                        decode_rows.append(dict(
                            arch=arch, seed=seed_idx, N=N, variable=var, t_before_end=t,
                            auc_hidden=auc_h[t], auc_cb=auc_cb[t],
                        ))

                # Parts 2-6: CB-RNN only
                trans = extract_transitions(per_batch, N=N)
                if trans["h_after"] is None or trans["h_after"].shape[0] < 50 or not model.use_cb_bias:
                    continue

                m_hat, midpoint, train_idx, eval_idx = build_reference_geometry(
                    trans, parity_key_after="wp_after", seed=EVAL_SEED,
                )

                p_before = trans["wp_before"][eval_idx]
                x_bit = (trans["x_t"][eval_idx].squeeze(-1) > 0.5).astype(int)
                outgoing = trans["outgoing_bit"][eval_idx]
                next_parity = trans["wp_after"][eval_idx]
                group_id = compute_group_id(p_before, x_bit, outgoing)
                cb_vectors = trans["cb_before"][eval_idx]
                H = HIDDEN_SIZES[arch]

                # Part 2: 8-condition geometry
                geom = condition_geometry(cb_vectors, group_id, hidden_size=H)
                for g in range(8):
                    geom_summary_rows.append(dict(
                        arch=arch, seed=seed_idx, N=N, group_id=g, group_label=GROUP_LABELS[g],
                        count=int(geom["counts"][g]), magnitude_per_unit=float(geom["magnitude_per_unit"][g]),
                        within_rms_per_unit=float(geom["within_rms_per_unit"][g]),
                        between_rms_per_unit=geom["between_rms_per_unit"],
                    ))
                for gi in range(8):
                    for gj in range(8):
                        geom_pair_rows.append(dict(
                            arch=arch, seed=seed_idx, N=N, group_i=gi, group_j=gj,
                            cosine=float(geom["cos_matrix"][gi, gj]), dist_per_unit=float(geom["dist_matrix_per_unit"][gi, gj]),
                        ))

                if not pca_example_saved and seed_idx == PCA_EXAMPLE_SEED and N == pca_target_N:
                    np.savez(
                        os.path.join(DATA_DIR, "parity_cb_pca_example.npz"),
                        cb_vectors=cb_vectors, group_id=group_id, next_parity=next_parity,
                        N=N, seed=seed_idx,
                    )
                    pca_example_saved = True

                # Part 3: encoding models A -> D
                batch_idx_eval = trans["batch_idx"][eval_idx]
                enc = encoding_model_r2(cb_vectors, p_before, x_bit, outgoing, next_parity, batch_idx_eval, seed=EVAL_SEED)
                for model_name, res in enc.items():
                    encoding_rows.append(dict(arch=arch, seed=seed_idx, N=N, model=model_name, **res))

                # Parts 4-5: causal swaps and per-condition margin
                causal_new, margin = new_causal_swap_conditions(model, trans, m_hat, midpoint, eval_idx, device=DEVICE, seed=EVAL_SEED)
                if causal_new is not None:
                    # normalise by half the even/odd centroid separation: +1 = correct, -1 = wrong
                    h_tr, p_tr = trans["h_after"][train_idx], trans["wp_after"][train_idx]
                    half_sep = float(np.linalg.norm(h_tr[p_tr == 0].mean(axis=0) - h_tr[p_tr == 1].mean(axis=0)) / 2)
                    row = dict(arch=arch, seed=seed_idx, N=N, half_sep=half_sep)
                    row.update({f"score_{k}": v["mean_score"] for k, v in causal_new.items()})
                    row.update({f"sem_{k}": v["sem_score"] for k, v in causal_new.items()})
                    row.update({f"normscore_{k}": v["mean_score"] / half_sep for k, v in causal_new.items()})
                    causal_rows.append(row)
                    for mrow in margin:
                        margin_rows.append(dict(arch=arch, seed=seed_idx, N=N, group_label=GROUP_LABELS[mrow["group_id"]], **mrow))

                # Part 6: lookup quantification
                lookup = lookup_quantification(model, trans, eval_idx, device=DEVICE)
                if lookup is not None:
                    lookup_rows.append(dict(arch=arch, seed=seed_idx, N=N, **lookup))

            log(f"{arch} seed {seed_idx} done")

    pd.DataFrame(decode_rows).to_csv(os.path.join(DATA_DIR, "parity_cb_decode_variables.csv"), index=False)
    pd.DataFrame(geom_summary_rows).to_csv(os.path.join(DATA_DIR, "parity_cb_condition_summary.csv"), index=False)
    pd.DataFrame(geom_pair_rows).to_csv(os.path.join(DATA_DIR, "parity_cb_condition_geometry.csv"), index=False)
    pd.DataFrame(encoding_rows).to_csv(os.path.join(DATA_DIR, "parity_cb_encoding_models.csv"), index=False)
    pd.DataFrame(causal_rows).to_csv(os.path.join(DATA_DIR, "parity_cb_swap_scores.csv"), index=False)
    pd.DataFrame(margin_rows).to_csv(os.path.join(DATA_DIR, "parity_cb_condition_margin.csv"), index=False)
    pd.DataFrame(lookup_rows).to_csv(os.path.join(DATA_DIR, "parity_cb_lookup.csv"), index=False)

    log("Parity CB-computation data collection complete.")


if __name__ == "__main__":
    main()
