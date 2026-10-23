"""Recompute the headline manuscript metrics from the stored out-of-fold (OOF) predictions.

This script does not retrain any model. It re-scores the saved Phase 3B/3C/3D predictions with
scikit-learn and with an independent NumPy macro-F1 implementation, and re-estimates the paired
patient-blocked bootstrap intervals with its own random seed (so interval endpoints can differ from the
manuscript in the second decimal; point estimates must match exactly).

Usage:
    python verify_reported_metrics.py [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parent
LABELS = [0, 1, 2, 3, 4]  # 0=NORM, 1=AFIB/AFL, 2=CD, 3=STTC/MI, 4=HYP

PHASE3B = ROOT / "predictions" / "phase3b_repeated_oof_predictions.npz"
PHASE3C = ROOT / "experiments_phase3" / "phase3c_sensitivity" / "phase3c_oof_predictions.npz"
PHASE3D = ROOT / "experiments_phase3" / "phase3d_full_cohort" / "phase3d_full_cohort_oof_predictions.npz"


def macro_f1_sklearn(y: np.ndarray, p: np.ndarray) -> float:
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def macro_f1_numpy(y: np.ndarray, p: np.ndarray) -> float:
    """Independent macro-F1: mean of per-class 2TP/(2TP+FP+FN), 0 when TP=0."""
    scores = []
    for c in LABELS:
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def paired_blocked_bootstrap(y, preds_a, preds_b, n_boot, rng):
    """Paired patient-index bootstrap; the same index vector is applied to every seed and both models."""
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = np.mean([macro_f1_numpy(y[idx], a[idx]) - macro_f1_numpy(y[idx], c[idx])
                            for a, c in zip(preds_a, preds_b)])
    return 100 * np.percentile(diffs, [2.5, 97.5])


def check_phase3b(n_boot, rng):
    z = np.load(PHASE3B, allow_pickle=True)
    y = z["true_labels"]
    assert len(set(z["patient_ids"])) == len(y), "duplicate patient IDs"
    print(f"Phase 3B: N={len(y)}, class counts={np.bincount(y).tolist()}")
    scores = {}
    for m in ["distill128", "hard128", "smooth128", "hard64", "tabpfn"]:
        preds = z[f"oof_preds_{m}"]
        assert (z[f"oof_probs_{m}"].argmax(-1) == preds).all(), f"{m}: argmax mismatch"
        f = np.array([macro_f1_sklearn(y, s) for s in preds])
        assert np.allclose(f, [macro_f1_numpy(y, s) for s in preds])
        scores[m] = f
        print(f"  {m:11s} Macro F1 {100 * f.mean():.2f} ± {100 * f.std(ddof=1):.2f}  seeds {np.round(100 * f, 2).tolist()}")
    for a, b in [("distill128", "hard128"), ("distill128", "smooth128"), ("tabpfn", "distill128")]:
        ci = paired_blocked_bootstrap(y, z[f"oof_preds_{a}"], z[f"oof_preds_{b}"], n_boot, rng)
        print(f"  {a} − {b}: {100 * (scores[a] - scores[b]).mean():+.2f} pp, 95% CI ≈ [{ci[0]:.2f}, {ci[1]:.2f}]")


def check_phase3c():
    z = np.load(PHASE3C, allow_pickle=True)
    y = z["y_common"]
    print(f"Phase 3C: N={len(y)}, class counts={np.bincount(y).tolist()}")
    for k in ["eval_a_preds_distill", "eval_b_preds_distill", "eval_b_preds_hard128", "eval_b_preds_smooth"]:
        per_class = 100 * f1_score(y, z[k], labels=LABELS, average=None, zero_division=0)
        print(f"  {k:22s} Macro F1 {100 * macro_f1_sklearn(y, z[k]):.2f}  STTC/MI F1 {per_class[3]:.2f}  HYP F1 {per_class[4]:.2f}")


def check_phase3d(n_boot, rng):
    z = np.load(PHASE3D, allow_pickle=True)
    y = z["y_true"]
    assert len(set(z["patient_ids"])) == len(y), "duplicate patient IDs"
    print(f"Phase 3D: N={len(y)}, class counts={np.bincount(y).tolist()}")
    y_bin = (y != 0).astype(int)  # abnormal (any non-NORM class) is the positive class
    for m in ["standalone_tabpfn", "distilled_mlp_128_64", "smooth_mlp_128_64", "hard_mlp_64_32", "hard_mlp_128_64"]:
        p = z[f"{m}_preds"]
        assert (z[f"{m}_probs"].argmax(1) == p).all(), f"{m}: argmax mismatch"
        p_bin = (p != 0).astype(int)
        tp = np.sum((y_bin == 1) & (p_bin == 1)); tn = np.sum((y_bin == 0) & (p_bin == 0))
        fp = np.sum((y_bin == 0) & (p_bin == 1)); fn = np.sum((y_bin == 1) & (p_bin == 0))
        hyp_recall = 100 * recall_score(y, p, labels=[4], average="macro", zero_division=0)
        print(f"  {m:22s} Macro F1 {100 * macro_f1_sklearn(y, p):.2f}  BA {100 * recall_score(y, p, average='macro'):.2f}"
              f"  Acc {100 * np.mean(p == y):.2f}  Sens {100 * tp / (tp + fn):.2f}  Spec {100 * tn / (tn + fp):.2f}"
              f"  HYP recall {hyp_recall:.2f}")
    for a, b in [("distilled_mlp_128_64", "hard_mlp_128_64"), ("distilled_mlp_128_64", "smooth_mlp_128_64")]:
        pa, pb = z[f"{a}_preds"][None], z[f"{b}_preds"][None]
        ci = paired_blocked_bootstrap(y, pa, pb, n_boot, rng)
        diff = 100 * (macro_f1_sklearn(y, pa[0]) - macro_f1_sklearn(y, pb[0]))
        print(f"  {a} − {b}: {diff:+.2f} pp, 95% CI ≈ [{ci[0]:.2f}, {ci[1]:.2f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates (manuscript used 10,000)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    check_phase3b(args.bootstrap, rng)
    check_phase3c()
    check_phase3d(args.bootstrap, rng)


if __name__ == "__main__":
    main()
