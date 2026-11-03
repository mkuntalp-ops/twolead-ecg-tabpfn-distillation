#!/usr/bin/env python3
"""WP3 secondary analyses from the stored WP3 out-of-fold predictions (no model is refitted).

The tuned models use the same nested selection as wp3_analyze.py (student-validation Macro F1 only).
Metric definitions are those of WP4 (`revizyon_v22/analysis/wp4_calibration_discrimination_subgroups.py`).

Usage (project root):  python revizyon_v22/wp3/wp3_secondary.py
Outputs (revizyon_v22/results/wp3/secondary/):
  wp3_full_metrics_per_seed.csv   all WP4 metrics (incl. AUPRC, Brier, class-wise ECE, binary sens./spec.) per model
  wp3_classwise.csv               per-class F1, precision, recall, AUROC, AUPRC (per cohort/seed/model)
  wp3_confusion_matrices.csv      5x5 confusion matrices (rows = reference class, columns = prediction)
  wp3_subgroups.csv               sex and age (<65 / >=65) subgroups, single-partition cohorts (PTB-XL 15,160;
                                  Chapman 9,315), with within-subgroup patient bootstrap intervals (B = 2000)
  wp3_reliability_<cohort>.{pdf,png}  top-label reliability diagrams for the single-partition cohorts
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp3_analyze as A  # noqa: E402
from wp3_common import CLASSES, COHORTS, K, OUT_ROOT, load_cohort  # noqa: E402

wp4 = A.wp4
OUT = OUT_ROOT / "secondary"
B, SEED = 2000, 2026
REPORT = {("teacher_tabpfn", "teacher"): "TabPFN teacher", ("mlp", "H"): "MLP hard", ("mlp", "S_fixed"): "MLP smooth (fixed)",
          ("mlp", "D_fixed"): "MLP distil (fixed)", ("mlp", "S_tuned"): "MLP smooth (tuned)",
          ("mlp", "D_tuned"): "MLP distil (tuned)", ("mlp", "EM_tuned"): "MLP entropy-matched (tuned)",
          ("xgb", "H"): "XGB hard", ("xgb", "S_fixed"): "XGB smooth (fixed)", ("xgb", "D_fixed"): "XGB distil (fixed)",
          ("xgb", "D_tuned"): "XGB distil (tuned)", ("xgb", "S_tuned"): "XGB smooth (tuned)",
          ("xgb", "EM_tuned"): "XGB entropy-matched (tuned)"}
SUB_METRICS = ["macro_F1", "macro_AUROC", "binary_sensitivity", "binary_specificity", "HYP_recall", "ECE_top_label_15bin"]


def cohort_models(cohort: str, seed: int):
    cdir = OUT_ROOT / cohort
    z = dict(np.load(cdir / f"seed{seed}_mlp.npz", allow_pickle=True))
    zx = dict(np.load(cdir / f"seed{seed}_xgb.npz", allow_pickle=True))
    assert np.array_equal(zx["uid"], z["uid"])
    z.update({k: v for k, v in zx.items() if k.startswith("xgb_")})
    val = pd.concat([pd.read_csv(cdir / f"seed{seed}_mlp_val.csv"), pd.read_csv(cdir / f"seed{seed}_xgb_val.csv")])
    sel = A.select(val)
    fold = z["fold"].astype(int)
    probs = {("teacher_tabpfn", "teacher"): z["teacher_tabpfn"].astype(np.float64)}
    for lrn in ("mlp", "xgb"):
        for name, p in A.model_probs(z, fold, sel, lrn).items():
            probs[(lrn, name)] = p
    return z["y"].astype(int), z["uid"].astype(str), {REPORT[k]: probs[k] for k in REPORT}


def reliability(cohort, y, probs, path):
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    style = {"TabPFN teacher": ("#6b6a66", "D"), "MLP hard": ("#1baf7a", "^"), "MLP smooth (fixed)": ("#eb6834", "s"),
             "MLP distil (fixed)": ("#2a78d6", "o"), "MLP distil (tuned)": ("#8a5cd1", "v")}
    for m, (col, mk) in style.items():
        p = probs[m]; conf = p.max(1); corr = p.argmax(1) == y
        bins = np.minimum((conf * wp4.N_BINS).astype(int), wp4.N_BINS - 1)
        xs, ys = [], []
        for b_ in range(wp4.N_BINS):
            s = bins == b_
            if s.sum() >= 30:
                xs.append(conf[s].mean()); ys.append(corr[s].mean())
        ax.plot(xs, ys, marker=mk, color=col, lw=1.6, ms=5, label=f"{m} (ECE {100 * wp4.top_label_ece(y, p):.1f}%)")
    ax.plot([0, 1], [0, 1], color="#52514e", lw=0.8, ls="--")
    ax.set_xlabel("Mean top-label confidence (15 equal-width bins, n ≥ 30)"); ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0.2, 1.0); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"Top-label reliability, {cohort} (N = {len(y):,})", fontsize=9, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mrows, crows, cmrows, grows = [], [], [], []
    for cohort in COHORTS:
        d, seeds = load_cohort(cohort)
        for seed in seeds:
            y, uid, probs = cohort_models(cohort, seed)
            assert np.array_equal(uid, d["uid"].to_numpy()), "record order differs from the cohort table"
            for m, p in probs.items():
                assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-4)
                pred = p.argmax(1)
                mrows.append({"cohort": cohort, "seed": seed, "model": m, "n": len(y), **wp4.metrics(y, p)})
                pr, rc, f1, n = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
                for c in range(K):
                    crows.append({"cohort": cohort, "seed": seed, "model": m, "class": CLASSES[c], "n": int(n[c]),
                                  "precision": 100 * pr[c], "recall": 100 * rc[c], "F1": 100 * f1[c],
                                  "AUROC": 100 * roc_auc_score(y == c, p[:, c]),
                                  "AUPRC": 100 * average_precision_score(y == c, p[:, c])})
                cm = confusion_matrix(y, pred, labels=list(range(K)))
                assert cm.sum() == len(y)
                for i in range(K):
                    cmrows.append({"cohort": cohort, "seed": seed, "model": m, "reference": CLASSES[i],
                                   **{f"pred_{CLASSES[j]}": int(cm[i, j]) for j in range(K)}})
            if len(seeds) == 1:
                reliability(cohort, y, probs, OUT / f"wp3_reliability_{cohort}")
                if cohort.startswith("ptbxl"):
                    female = d["sex"].to_numpy() == 1          # PTB-XL: 0 = male, 1 = female
                else:
                    female = d["sex_male"].to_numpy() == 0
                age = d["age"].to_numpy(np.float64)
                groups = {"male": ~female, "female": female, "age<65": age < 65, "age>=65": age >= 65}
                for g, mask in groups.items():
                    gi = np.where(mask)[0]; yg = y[gi]
                    rng = np.random.default_rng(SEED + 1)  # one record per patient in these cohorts
                    idx = [rng.integers(0, len(gi), len(gi)) for _ in range(B)]
                    gp = {m: probs[m][gi] for m in ("TabPFN teacher", "MLP hard", "MLP smooth (fixed)",
                                                    "MLP distil (fixed)", "MLP distil (tuned)", "XGB hard")}
                    draws = wp4.bootstrap_ci(yg, gp, idx, SUB_METRICS)
                    for m, p in gp.items():
                        pt = wp4.metrics(yg, p)
                        for k in SUB_METRICS:
                            lo, hi = np.percentile(draws[m][k], [2.5, 97.5])
                            grows.append({"cohort": cohort, "subgroup": g, "n": len(gi),
                                          "n_per_class": "/".join(map(str, np.bincount(yg, minlength=K))),
                                          "model": m, "metric": k, "estimate": pt[k], "ci95_low": lo, "ci95_high": hi})
            print(f"{cohort} seed {seed} done", flush=True)
    pd.DataFrame(mrows).to_csv(OUT / "wp3_full_metrics_per_seed.csv", index=False)
    pd.DataFrame(crows).to_csv(OUT / "wp3_classwise.csv", index=False)
    pd.DataFrame(cmrows).to_csv(OUT / "wp3_confusion_matrices.csv", index=False)
    pd.DataFrame(grows).to_csv(OUT / "wp3_subgroups.csv", index=False)


if __name__ == "__main__":
    main()
