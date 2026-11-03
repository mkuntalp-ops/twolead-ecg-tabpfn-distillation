#!/usr/bin/env python3
"""WP4: discrimination, calibration and sex/age subgroup analyses from stored out-of-fold (OOF) predictions.

No model is retrained and nothing is tuned on the evaluation data. All analyses are secondary/exploratory
(see ../ANALIZ_PLANI_v22.md). Inputs are read relative to the project root.

Outputs (revizyon_v22/results/wp4/):
  wp4_3d_metrics.csv          per-model metrics with patient-blocked bootstrap 95% intervals (Phase 3D, N=15,160)
  wp4_3d_paired_diffs.csv     paired differences between models (same bootstrap index vectors)
  wp4_3d_classwise.csv        per-class AUROC, AUPRC and one-vs-rest ECE
  wp4_3b_seed_metrics.csv     per-seed metrics for the five-partition quota-selected evaluation (Phase 3B)
  wp4_subgroups.csv           sex and age (<65 / >=65) subgroup metrics with within-subgroup bootstrap intervals
  wp4_reliability_3d.pdf/png  top-label reliability diagram (Phase 3D)
  wp4_run_manifest.json       inputs (SHA-256), seeds, package versions, self-checks

Usage (from the project root):  python revizyon_v22/analysis/wp4_calibration_discrimination_subgroups.py
       FEAT_50 v2 re-runs:       WP4_FEATURES=v2 python revizyon_v22/analysis/wp4_calibration_discrimination_subgroups.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
# WP4_FEATURES=v2 reads the FEAT_50 v2 re-runs (revizyon_v22/runs/*_v2) and writes to results/wp4_v2; default is v1.
FEATURES = os.environ.get("WP4_FEATURES", "v1")
assert FEATURES in ("v1", "v2"), f"WP4_FEATURES must be v1 or v2, got {FEATURES!r}"
OUT = ROOT / "revizyon_v22" / "results" / ("wp4" if FEATURES == "v1" else "wp4_v2")
OUT.mkdir(parents=True, exist_ok=True)

if FEATURES == "v1":
    P3D = ROOT / "experiments_phase3/phase3d_full_cohort/phase3d_full_cohort_oof_predictions.npz"
    P3B = ROOT / "predictions/phase3b_repeated_oof_predictions.npz"
else:
    P3D = ROOT / "revizyon_v22/runs/3d_v2/ws/experiments_phase3/phase3d_full_cohort/phase3d_full_cohort_oof_predictions.npz"
    P3B = ROOT / "revizyon_v22/runs/3b_v2/ws/predictions/phase3b_repeated_oof_predictions.npz"
MANIFEST = ROOT / "experiments_phase3/phase3d_full_cohort/phase3d_eligible_cohort_15160_manifest.csv"

CLASSES = ["NORM", "AFIB/AFL", "CD", "STTC/MI", "HYP"]
K = len(CLASSES)
MODELS_3D = {"TabPFN teacher": "standalone_tabpfn", "Distilled MLP 128-64": "distilled_mlp_128_64",
             "Smoothed MLP 128-64": "smooth_mlp_128_64", "Hard-label MLP 128-64": "hard_mlp_128_64",
             "Hard-label MLP 64-32": "hard_mlp_64_32"}
MODELS_3B = {"TabPFN teacher": "tabpfn", "Distilled MLP 128-64": "distill128", "Smoothed MLP 128-64": "smooth128",
             "Hard-label MLP 128-64": "hard128", "Hard-label MLP 64-32": "hard64"}
CONTRASTS = [("Distilled MLP 128-64", "Hard-label MLP 128-64"), ("Distilled MLP 128-64", "Smoothed MLP 128-64"),
             ("Smoothed MLP 128-64", "Hard-label MLP 128-64"), ("TabPFN teacher", "Distilled MLP 128-64")]
N_BINS = 15
B = 2000
SEED = 2026
EPS = 1e-12


# --------------------------------------------------------------------------- metric definitions
def top_label_ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Top-label ECE: sum_b (n_b/N) |accuracy_b - mean confidence_b|, equal-width bins on (0, 1]."""
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    n_b = np.bincount(idx, minlength=n_bins)
    acc_b = np.bincount(idx, weights=correct, minlength=n_bins)
    conf_b = np.bincount(idx, weights=conf, minlength=n_bins)
    nz = n_b > 0
    return float(np.sum(np.abs(acc_b[nz] - conf_b[nz])) / len(y))


def classwise_ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """One-vs-rest ECE per class: sum_b (n_b/N) |observed frequency_b - mean predicted probability_b|."""
    out = np.empty(p.shape[1])
    for c in range(p.shape[1]):
        q = p[:, c]
        obs = (y == c).astype(float)
        idx = np.minimum((q * n_bins).astype(int), n_bins - 1)
        n_b = np.bincount(idx, minlength=n_bins)
        o_b = np.bincount(idx, weights=obs, minlength=n_bins)
        q_b = np.bincount(idx, weights=q, minlength=n_bins)
        nz = n_b > 0
        out[c] = np.sum(np.abs(o_b[nz] - q_b[nz])) / len(y)
    return out


def brier(y: np.ndarray, p: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def nll(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def macro_auroc(y, p):
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(K))))


def macro_auprc(y, p):
    onehot = np.eye(K)[y]
    return float(np.mean([average_precision_score(onehot[:, c], p[:, c]) for c in range(K)]))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = p.argmax(1)
    yb, pb = (y != 0).astype(int), (pred != 0).astype(int)
    tp = np.sum((yb == 1) & (pb == 1)); tn = np.sum((yb == 0) & (pb == 0))
    fp = np.sum((yb == 0) & (pb == 1)); fn = np.sum((yb == 1) & (pb == 0))
    return {
        "macro_F1": 100 * f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0),
        "macro_AUROC": 100 * macro_auroc(y, p),
        "macro_AUPRC": 100 * macro_auprc(y, p),
        "binary_AUROC_abnormal": 100 * roc_auc_score(yb, 1 - p[:, 0]),
        "NLL": nll(y, p),
        "Brier": brier(y, p),
        "ECE_top_label_15bin": 100 * top_label_ece(y, p),
        "ECE_classwise_15bin": 100 * float(np.mean(classwise_ece(y, p))),
        "binary_sensitivity": 100 * tp / max(tp + fn, 1),
        "binary_specificity": 100 * tn / max(tn + fp, 1),
        "HYP_recall": 100 * np.sum((y == 4) & (pred == 4)) / max(np.sum(y == 4), 1),
    }


# --------------------------------------------------------------------------- self-checks (second implementations)
def auroc_mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC as P(score_pos > score_neg) + 0.5 P(tie), via ranks (independent of scikit-learn)."""
    scores = np.concatenate([pos, neg])
    ranks = pd.Series(scores).rank(method="average").to_numpy()   # average ranks for ties
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def self_checks(y, p) -> dict:
    checks = {}
    ovr = [auroc_mann_whitney(p[y == c, c], p[y != c, c]) for c in range(K)]
    checks["auroc_mann_whitney_vs_sklearn_max_abs_diff"] = float(abs(np.mean(ovr) - macro_auroc(y, p)))
    checks["nll_vs_sklearn_log_loss_abs_diff"] = float(abs(nll(y, p) - log_loss(y, p.astype(np.float64), labels=list(range(K)))))
    rng = np.random.default_rng(0)                             # synthetic two-class data whose top label is always class 0
    n = 200_000
    conf = rng.uniform(0.55, 0.85, n)
    yy = np.where(rng.uniform(size=n) < conf, 0, 1)            # P(correct) = stated confidence -> calibrated
    checks["ece_on_calibrated_synthetic_(expected_~0)"] = top_label_ece(yy, np.c_[conf, 1 - conf])
    checks["ece_on_overconfident_by_0.10_synthetic_(expected_~0.10)"] = top_label_ece(yy, np.c_[conf + 0.1, 0.9 - conf])
    return checks


# --------------------------------------------------------------------------- bootstrap helpers
def bootstrap_ci(y, probs: dict[str, np.ndarray], idx_list, metric_names):
    """Per-model metric percentiles and paired-difference percentiles using the same index vectors."""
    draws = {m: {k: [] for k in metric_names} for m in probs}
    for idx in idx_list:
        yy = y[idx]
        for m, p in probs.items():
            vals = metrics(yy, p[idx])
            for k in metric_names: draws[m][k].append(vals[k])
    return {m: {k: np.asarray(v) for k, v in d.items()} for m, d in draws.items()}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    z = np.load(P3D, allow_pickle=True)
    y = z["y_true"].astype(int)
    probs = {name: z[f"{key}_probs"].astype(np.float64) for name, key in MODELS_3D.items()}
    for name, p in probs.items():
        assert p.shape == (len(y), K), name
        assert np.allclose(p.sum(1), 1.0, atol=1e-4), f"{name}: rows do not sum to 1"
        assert (p.argmax(1) == z[f"{MODELS_3D[name]}_preds"]).all(), f"{name}: argmax mismatch"
    assert np.array_equal(np.bincount(y, minlength=K), [8052, 1155, 1647, 3486, 820])

    checks = self_checks(y, probs["Distilled MLP 128-64"])

    # ---- Phase 3D point estimates + bootstrap
    names = list(metrics(y, probs["TabPFN teacher"]).keys())
    rng = np.random.default_rng(SEED)
    idx_list = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    draws = bootstrap_ci(y, probs, idx_list, names)
    rows = []
    for m, p in probs.items():
        pt = metrics(y, p)
        for k in names:
            lo, hi = np.percentile(draws[m][k], [2.5, 97.5])
            rows.append({"model": m, "metric": k, "estimate": pt[k], "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(rows).to_csv(OUT / "wp4_3d_metrics.csv", index=False)
    drows = []
    for a, b in CONTRASTS:
        pa, pb = metrics(y, probs[a]), metrics(y, probs[b])
        for k in names:
            d = draws[a][k] - draws[b][k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            n_opp = min(np.sum(d <= 0), np.sum(d >= 0))
            drows.append({"contrast": f"{a} minus {b}", "metric": k, "difference": pa[k] - pb[k], "ci95_low": lo,
                          "ci95_high": hi, "bootstrap_tail_two_sided": min(1.0, 2 * (n_opp + 1) / (B + 1))})
    pd.DataFrame(drows).to_csv(OUT / "wp4_3d_paired_diffs.csv", index=False)

    # ---- class-wise
    crow = []
    onehot = np.eye(K)[y]
    for m, p in probs.items():
        ce = classwise_ece(y, p)
        for c in range(K):
            crow.append({"model": m, "class": CLASSES[c], "n": int(onehot[:, c].sum()),
                         "AUROC": 100 * roc_auc_score(onehot[:, c], p[:, c]),
                         "AUPRC": 100 * average_precision_score(onehot[:, c], p[:, c]),
                         "prevalence_percent": 100 * onehot[:, c].mean(), "ECE_ovr_15bin": 100 * ce[c]})
    pd.DataFrame(crow).to_csv(OUT / "wp4_3d_classwise.csv", index=False)

    # ---- Phase 3B per seed
    zb = np.load(P3B, allow_pickle=True)
    yb = zb["true_labels"].astype(int)
    srows = []
    for name, key in MODELS_3B.items():
        for s_i, seed in enumerate(zb["seeds"]):
            p = zb[f"oof_probs_{key}"][s_i].astype(np.float64)
            assert (p.argmax(1) == zb[f"oof_preds_{key}"][s_i]).all()
            srows.append({"model": name, "seed": int(seed), **metrics(yb, p)})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(OUT / "wp4_3b_seed_metrics.csv", index=False)

    # ---- subgroups (Phase 3D)
    man = pd.read_csv(MANIFEST).set_index("patient_id_str")
    meta = man.loc[z["patient_ids"]]
    assert (meta["ecg_id"].to_numpy() == z["ecg_ids"]).all()
    groups = {"sex=male (code 0)": meta["sex"].to_numpy() == 0, "sex=female (code 1)": meta["sex"].to_numpy() == 1,
              "age<65": meta["age"].to_numpy() < 65, "age>=65": meta["age"].to_numpy() >= 65}
    sub_names = ["macro_F1", "macro_AUROC", "binary_sensitivity", "binary_specificity", "HYP_recall", "ECE_top_label_15bin"]
    grows = []
    for g, mask in groups.items():
        gi = np.where(mask)[0]; yg = y[gi]
        rng_g = np.random.default_rng(SEED + 1)
        gidx = [rng_g.integers(0, len(gi), len(gi)) for _ in range(B)]
        gprobs = {m: probs[m][gi] for m in ["TabPFN teacher", "Distilled MLP 128-64", "Smoothed MLP 128-64", "Hard-label MLP 128-64"]}
        gd = bootstrap_ci(yg, gprobs, gidx, sub_names)
        for m, p in gprobs.items():
            pt = metrics(yg, p)
            for k in sub_names:
                lo, hi = np.percentile(gd[m][k], [2.5, 97.5])
                grows.append({"subgroup": g, "n": len(gi), "n_per_class": "/".join(map(str, np.bincount(yg, minlength=K))),
                              "model": m, "metric": k, "estimate": pt[k], "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(grows).to_csv(OUT / "wp4_subgroups.csv", index=False)

    # ---- reliability diagram (top-label)
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    style = {"TabPFN teacher": ("#6b6a66", "D"), "Distilled MLP 128-64": ("#2a78d6", "o"),
             "Smoothed MLP 128-64": ("#eb6834", "s"), "Hard-label MLP 128-64": ("#1baf7a", "^")}
    for m, (col, mk) in style.items():
        p = probs[m]; conf = p.max(1); corr = (p.argmax(1) == y)
        bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
        xs, ys = [], []
        for b_ in range(N_BINS):
            sel = bins == b_
            if sel.sum() >= 30: xs.append(conf[sel].mean()); ys.append(corr[sel].mean())
        ax.plot(xs, ys, marker=mk, color=col, lw=1.6, ms=5, label=f"{m} (ECE {100 * top_label_ece(y, p):.1f}%)")
    ax.plot([0, 1], [0, 1], color="#52514e", lw=0.8, ls="--")
    ax.set_xlabel("Mean top-label confidence (15 equal-width bins, n ≥ 30)"); ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0.2, 1.0); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Top-label reliability, eligibility-filtered PTB-XL cohort (N = 15,160)", fontsize=9, loc="left")
    for ext in ("pdf", "png"): fig.savefig(OUT / f"wp4_reliability_3d.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    json.dump({"inputs": {str(p.relative_to(ROOT)): sha(p) for p in (P3D, P3B, MANIFEST)},
               "bootstrap_B": B, "bootstrap_seed": SEED, "subgroup_seed": SEED + 1, "n_bins": N_BINS,
               "sex_coding": "PTB-XL sex 0 = male, 1 = female (inferred from PTB-XL documentation: 52% male / 48% female; database codes 0: 52.1%, 1: 47.9%)",
               "self_checks": checks, "python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
              open(OUT / "wp4_run_manifest.json", "w"), indent=2)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
