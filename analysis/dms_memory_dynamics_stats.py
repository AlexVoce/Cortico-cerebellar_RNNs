"""Delay-averaged Q per seed and N, plus seed-level statistics (n = 8 seeds).

Run with: python -m analysis.dms_memory_dynamics_stats
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = "results/mechanistic_analysis/dms_memory_propagation"
EARLY_NS = list(range(5, 10))
LATE_NS = list(range(26, 31))
N_BOOT, BOOT_SEED = 10000, 0
CONDITIONS = [  # (label, arch, per-transition column)
    ("RNN-only", "rnn_only", "q_full"),
    ("CB-RNN (full)", "cb_rnn", "q_full"),
    ("CB-RNN (CB detached)", "cb_rnn", "q_det"),
    ("CB contribution", "cb_rnn", "q_cb"),
]


def boot_ci(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(BOOT_SEED)
    b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def summarise(x, **info):
    x = np.asarray(x, float)
    lo, hi = boot_ci(x)
    t = stats.ttest_1samp(x, 0.0)
    w = stats.wilcoxon(x)
    sd = x.std(ddof=1)
    return dict(**info, n_seeds=len(x), mean=x.mean(), sem=stats.sem(x), ci95_lo=lo, ci95_hi=hi,
                t=t.statistic, df=len(x) - 1, p_t=t.pvalue, wilcoxon_W=w.statistic, p_wilcoxon=w.pvalue,
                cohens_dz=x.mean() / sd if sd > 0 else np.nan, n_seeds_positive=int((x > 0).sum()),
                seed_values=";".join(f"{v:.4f}" for v in x))


def delay_averages(trans):
    rows = []
    for label, arch, col in CONDITIONS:
        d = trans[trans.arch == arch]
        g = d.groupby(["variant", "seed", "N"]).agg(Q=(col, "mean"), n_delay_transitions=(col, "size"),
                                                   n_trials_per_transition=("n_trials", "first"))
        rows.append(g.reset_index().assign(condition=label, arch=arch))
    return pd.concat(rows, ignore_index=True)[
        ["variant", "condition", "arch", "seed", "N", "Q", "n_delay_transitions", "n_trials_per_transition"]]


def seed_level_tests(final):
    out = []
    for (variant, cond), d in final.groupby(["variant", "condition"]):
        w = d.pivot(index="seed", columns="N", values="Q")
        early, late = w[EARLY_NS].mean(axis=1), w[LATE_NS].mean(axis=1)
        slopes = [np.polyfit(w.columns.values, w.loc[s].values, 1)[0] for s in w.index]
        rhos = [stats.spearmanr(w.columns.values, w.loc[s].values).statistic for s in w.index]
        base = dict(variant=variant, condition=cond)
        out.append(summarise(early, **base, quantity=f"early window mean (N={EARLY_NS[0]}-{EARLY_NS[-1]})"))
        out.append(summarise(late, **base, quantity=f"late window mean (N={LATE_NS[0]}-{LATE_NS[-1]})"))
        if cond != "CB contribution":
            out.append(summarise(early - 1, **base, quantity="early window mean - 1"))
            out.append(summarise(late - 1, **base, quantity="late window mean - 1"))
        out.append(summarise(late - early, **base, quantity="late - early (paired)"))
        out.append(summarise(slopes, **base, quantity="within-seed OLS slope of Q vs N (per level)"))
        out.append(summarise(rhos, **base, quantity="within-seed Spearman rho(N, Q)"))
    return pd.DataFrame(out)


def leave_one_seed_out(final, variant="primary_eval_seed_12345"):
    rows = []
    d = final[final.variant == variant]
    for cond in ["CB-RNN (CB detached)", "CB contribution", "CB-RNN (full)"]:
        w = d[d.condition == cond].pivot(index="seed", columns="N", values="Q")
        change = (w[LATE_NS].mean(axis=1) - w[EARLY_NS].mean(axis=1))
        for s in change.index:
            x = change.drop(s).values
            rows.append(dict(condition=cond, dropped_seed=s, mean_late_minus_early=x.mean(),
                             p_t=stats.ttest_1samp(x, 0).pvalue, p_wilcoxon=stats.wilcoxon(x).pvalue))
    return pd.DataFrame(rows)


def main():
    trans = pd.read_csv(os.path.join(OUT_DIR, "dms_memory_dynamics_by_transition.csv"))
    final = delay_averages(trans)
    final.to_csv(os.path.join(OUT_DIR, "dms_memory_delay_averages.csv"), index=False)

    # identity at the delay-average level
    p = final[final.variant == "primary_eval_seed_12345"].pivot_table(
        index=["seed", "N"], columns="condition", values="Q")
    ident_err = float((p["CB contribution"] - (p["CB-RNN (full)"] - p["CB-RNN (CB detached)"])).abs().max())

    tests = seed_level_tests(final)
    tests.to_csv(os.path.join(OUT_DIR, "dms_memory_dynamics_statistics.csv"), index=False)
    loso = leave_one_seed_out(final)
    loso.to_csv(os.path.join(OUT_DIR, "dms_memory_dynamics_leave_one_seed_out.csv"), index=False)

    pd.set_option("display.width", 250)
    prim = tests[tests.variant == "primary_eval_seed_12345"]
    print(f"max |Q_CB - (Q_full - Q_det)| = {ident_err:.2e}")
    print(prim[["condition", "quantity", "mean", "sem", "ci95_lo", "ci95_hi", "t", "p_t", "p_wilcoxon",
                "n_seeds_positive"]].round(4).to_string(index=False))
    rob = tests[tests.quantity == "late - early (paired)"]
    print(rob[["variant", "condition", "mean", "sem", "p_t", "p_wilcoxon", "n_seeds_positive"]].round(4).to_string(index=False))
    print(loso.groupby("condition").agg(mean_min=("mean_late_minus_early", "min"),
                                        mean_max=("mean_late_minus_early", "max"),
                                        p_t_max=("p_t", "max"), p_w_max=("p_wilcoxon", "max")))


if __name__ == "__main__":
    main()
