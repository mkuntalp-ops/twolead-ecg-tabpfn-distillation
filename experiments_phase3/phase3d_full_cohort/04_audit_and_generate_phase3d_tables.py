#!/usr/bin/env python3
"""
experiments_phase3/phase3d_full_cohort/04_audit_and_generate_phase3d_tables.py

Validates and generates the final, authoritative Phase 3D metrics, tables,
and statistical tests from the frozen OOF predictions NPZ.

Enforces all user-specified identities and assertions:
  - TP + FN = 7108 (Abnormal)
  - TN + FP = 8052 (Normal)
  - binary_accuracy = (TP + TN) / 15160
  - binary_accuracy >= multiclass_accuracy
  - Distinctly computes and labels:
      * Abnormal F1 (pos_label=1)
      * Normal F1 (pos_label=0)
      * Binary Macro F1 ((F1_Norm + F1_Abn) / 2)
  - Explicitly separates Confirmatory Core Family (m=2) and Exploratory Family (m=5).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import hashlib

OUT_DIR = Path(__file__).resolve().parent
NPZ_PATH = OUT_DIR / "phase3d_full_cohort_oof_predictions.npz"

NUM_CLASSES = 5
CLASS_NAMES = ["NORM", "AFIB/AFL", "CD", "STTC/MI", "HYP"]
MODELS = [
    "distilled_mlp_128_64",
    "smooth_mlp_128_64",
    "hard_mlp_128_64",
    "hard_mlp_64_32",
    "standalone_tabpfn"
]

MODEL_DISPLAY_NAMES = {
    "distilled_mlp_128_64": "Teacher-Distilled MLP (128, 64)",
    "smooth_mlp_128_64": "Uniform-Smoothed MLP (128, 64)",
    "hard_mlp_128_64": "Hard-Label MLP (128, 64)",
    "hard_mlp_64_32": "Hard Baseline MLP (64, 32)",
    "standalone_tabpfn": "Standalone TabPFN"
}

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def canonical_bootstrap_paired(y_true, p1, p2, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    N = len(y_true)
    f1_1_orig = f1_score(y_true, p1, average="macro", zero_division=0)
    f1_2_orig = f1_score(y_true, p2, average="macro", zero_division=0)
    delta_orig = (f1_1_orig - f1_2_orig) * 100.0

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, N, size=N)
        yt_b = y_true[idx]
        f1_1_b = f1_score(yt_b, p1[idx], average="macro", zero_division=0)
        f1_2_b = f1_score(yt_b, p2[idx], average="macro", zero_division=0)
        deltas[b] = (f1_1_b - f1_2_b) * 100.0

    ci_low = np.percentile(deltas, 2.5)
    ci_high = np.percentile(deltas, 97.5)
    boot_mean = np.mean(deltas)

    n_le_zero = int(np.sum(deltas <= 0.0))
    n_ge_zero = int(np.sum(deltas >= 0.0))
    n_opp = min(n_le_zero, n_ge_zero)
    p_value = (2.0 * (n_opp + 1.0)) / (n_boot + 1.0)
    p_value = min(p_value, 1.0)

    return {
        "point_diff": float(delta_orig),
        "boot_mean": float(boot_mean),
        "ci_95": [float(ci_low), float(ci_high)],
        "n_opp": n_opp,
        "n_le_zero": n_le_zero,
        "n_ge_zero": n_ge_zero,
        "p_value": float(p_value),
    }

def main():
    print("=" * 80)
    print("PHASE 3D: RIGOROUS METRIC RE-AUDIT & CANONICAL CLOSING")
    print("=" * 80)

    data = np.load(NPZ_PATH)
    y_true = data["y_true"]
    N = len(y_true)
    assert N == 15160, f"Expected 15160, got {N}"

    y_bin_true = (y_true != 0).astype(int)
    n_norm_true = int(np.sum(y_bin_true == 0))
    n_abn_true = int(np.sum(y_bin_true == 1))
    assert n_norm_true == 8052, f"Normal count {n_norm_true} != 8052"
    assert n_abn_true == 7108, f"Abnormal count {n_abn_true} != 7108"

    mc_rows = []
    bin_rows = []
    cm_dict = {}

    for m in MODELS:
        preds = data[f"{m}_preds"]
        probs = data[f"{m}_probs"]

        # Multiclass metrics
        mc_f1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
        bal_acc = balanced_accuracy_score(y_true, preds) * 100.0
        mc_acc = accuracy_score(y_true, preds) * 100.0

        per_f1 = f1_score(y_true, preds, average=None, zero_division=0) * 100.0
        per_prec = precision_score(y_true, preds, average=None, zero_division=0) * 100.0
        per_rec = recall_score(y_true, preds, average=None, zero_division=0) * 100.0

        cm_mc = confusion_matrix(y_true, preds, labels=np.arange(NUM_CLASSES))

        mc_rows.append({
            "model_id": m,
            "model_name": MODEL_DISPLAY_NAMES[m],
            "macro_f1": mc_f1,
            "balanced_accuracy": bal_acc,
            "multiclass_accuracy": mc_acc,
            "norm_f1": per_f1[0],
            "afib_f1": per_f1[1],
            "cd_f1": per_f1[2],
            "sttc_f1": per_f1[3],
            "hyp_f1": per_f1[4],
            "hyp_recall": per_rec[4],
            "hyp_precision": per_prec[4],
        })

        # Binary mapping: 0 -> Normal (0), 1..4 -> Abnormal (1)
        y_bin_pred = (preds != 0).astype(int)
        cm_bin = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1])
        tn, fp, fn, tp = cm_bin.ravel()

        # Strict Assertions
        assert tp + fn == 7108, f"{m}: TP + FN != 7108"
        assert tn + fp == 8052, f"{m}: TN + FP != 8052"
        bin_acc = (tp + tn) / N * 100.0
        assert abs(bin_acc - (accuracy_score(y_bin_true, y_bin_pred) * 100.0)) < 1e-7, "bin_acc formula mismatch"
        assert bin_acc >= mc_acc, f"{m}: bin_acc ({bin_acc}) < mc_acc ({mc_acc})"

        sens = tp / (tp + fn) * 100.0
        spec = tn / (tn + fp) * 100.0
        ppv = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0

        f1_abn = f1_score(y_bin_true, y_bin_pred, pos_label=1, zero_division=0) * 100.0
        f1_norm = f1_score(y_bin_true, y_bin_pred, pos_label=0, zero_division=0) * 100.0
        bin_macro_f1 = (f1_abn + f1_norm) / 2.0

        bin_rows.append({
            "model_id": m,
            "model_name": MODEL_DISPLAY_NAMES[m],
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "multiclass_accuracy": mc_acc,
            "binary_accuracy": bin_acc,
            "accuracy_gain_binary_minus_mc": bin_acc - mc_acc,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "f1_abnormal_positive": f1_abn,
            "f1_normal": f1_norm,
            "binary_macro_f1": bin_macro_f1,
        })

        cm_dict[m] = {
            "multiclass_5x5": cm_mc.tolist(),
            "binary_2x2": {
                "layout": "[[TN, FP], [FN, TP]]",
                "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
            }
        }

    df_mc = pd.DataFrame(mc_rows)
    df_bin = pd.DataFrame(bin_rows)

    df_mc.to_csv(OUT_DIR / "phase3d_multiclass_metrics.csv", index=False)
    df_bin.to_csv(OUT_DIR / "phase3d_binary_screening_metrics.csv", index=False)

    with open(OUT_DIR / "phase3d_confusion_matrices.json", "w") as f:
        json.dump(cm_dict, f, indent=2)

    # Statistical Comparison Tables
    comparisons = [
        # Core Confirmatory Family (m = 2)
        {
            "family": "Core Confirmatory (m=2)",
            "family_size": 2,
            "comparison": "Distilled vs. Hard-Label (128, 64)",
            "m1": "distilled_mlp_128_64",
            "m2": "hard_mlp_128_64",
        },
        {
            "family": "Core Confirmatory (m=2)",
            "family_size": 2,
            "comparison": "Distilled vs. Uniform-Smoothed (128, 64)",
            "m1": "distilled_mlp_128_64",
            "m2": "smooth_mlp_128_64",
        },
        # Exploratory Family (m = 5)
        {
            "family": "Exploratory All-Pairs (m=5)",
            "family_size": 5,
            "comparison": "Standalone TabPFN vs. Distilled MLP",
            "m1": "standalone_tabpfn",
            "m2": "distilled_mlp_128_64",
        },
        {
            "family": "Exploratory All-Pairs (m=5)",
            "family_size": 5,
            "comparison": "Distilled vs. Hard-Label (128, 64)",
            "m1": "distilled_mlp_128_64",
            "m2": "hard_mlp_128_64",
        },
        {
            "family": "Exploratory All-Pairs (m=5)",
            "family_size": 5,
            "comparison": "Uniform-Smoothed vs. Hard-Label (128, 64)",
            "m1": "smooth_mlp_128_64",
            "m2": "hard_mlp_128_64",
        },
        {
            "family": "Exploratory All-Pairs (m=5)",
            "family_size": 5,
            "comparison": "Hard-Label (128, 64) vs. Baseline (64, 32)",
            "m1": "hard_mlp_128_64",
            "m2": "hard_mlp_64_32",
        },
        {
            "family": "Exploratory All-Pairs (m=5)",
            "family_size": 5,
            "comparison": "Distilled vs. Uniform-Smoothed (128, 64)",
            "m1": "distilled_mlp_128_64",
            "m2": "smooth_mlp_128_64",
        },
    ]

    stat_rows = []
    # Calculate bootstrap for all unique pairs
    boot_cache = {}
    for item in comparisons:
        pair_key = (item["m1"], item["m2"])
        if pair_key not in boot_cache:
            p1 = data[f"{item['m1']}_preds"]
            p2 = data[f"{item['m2']}_preds"]
            boot_cache[pair_key] = canonical_bootstrap_paired(y_true, p1, p2, n_boot=10000, seed=42)

    # Core Family Adjustment (m = 2)
    core_items = [c for c in comparisons if c["family_size"] == 2]
    for c in core_items:
        c["res"] = boot_cache[(c["m1"], c["m2"])]
    core_items.sort(key=lambda x: x["res"]["p_value"])

    # Holm-Bonferroni for Core (m=2)
    # rank 0: p * 2
    # rank 1: max(p_adj_prev, p * 1)
    p0 = core_items[0]["res"]["p_value"]
    p1 = core_items[1]["res"]["p_value"]
    core_items[0]["holm_p"] = min(1.0, p0 * 2.0)
    core_items[0]["bonf_p"] = min(1.0, p0 * 2.0)
    core_items[1]["holm_p"] = min(1.0, max(core_items[0]["holm_p"], p1 * 1.0))
    core_items[1]["bonf_p"] = min(1.0, p1 * 2.0)

    # Exploratory Family Adjustment (m = 5)
    exp_items = [c for c in comparisons if c["family_size"] == 5]
    for c in exp_items:
        c["res"] = boot_cache[(c["m1"], c["m2"])]
    exp_items.sort(key=lambda x: x["res"]["p_value"])

    running_max = 0.0
    for rank, c in enumerate(exp_items):
        multiplier = 5 - rank
        raw_p = c["res"]["p_value"]
        holm_p = min(1.0, max(running_max, raw_p * multiplier))
        running_max = holm_p
        bonf_p = min(1.0, raw_p * 5.0)
        c["holm_p"] = holm_p
        c["bonf_p"] = bonf_p

    # Combine into table
    for c in core_items + exp_items:
        r = c["res"]
        stat_rows.append({
            "family_name": c["family"],
            "family_size_m": c["family_size"],
            "comparison": c["comparison"],
            "point_delta_pp": r["point_diff"],
            "ci_95_low": r["ci_95"][0],
            "ci_95_high": r["ci_95"][1],
            "n_opp": r["n_opp"],
            "raw_p_value": r["p_value"],
            "holm_fwer_p": c["holm_p"],
            "bonferroni_p": c["bonf_p"],
            "significant_at_005": "YES" if c["holm_p"] < 0.05 else "NO",
        })

    df_stat = pd.DataFrame(stat_rows)
    df_stat.to_csv(OUT_DIR / "phase3d_statistical_tests_table.csv", index=False)

    print("\nMULTICLASS METRICS (PTB-XL Full Cohort N = 15,160):")
    print(df_mc[["model_name", "macro_f1", "balanced_accuracy", "multiclass_accuracy", "norm_f1", "hyp_f1", "hyp_recall"]].to_string(index=False))

    print("\nBINARY SCREENING METRICS (TN+FP=8052, TP+FN=7108):")
    print(df_bin[["model_name", "TN", "FP", "FN", "TP", "multiclass_accuracy", "binary_accuracy", "sensitivity", "specificity", "ppv", "npv", "f1_abnormal_positive", "binary_macro_f1"]].to_string(index=False))

    print("\nSTATISTICAL COMPARISON TABLE:")
    print(df_stat[["family_name", "comparison", "point_delta_pp", "ci_95_low", "ci_95_high", "raw_p_value", "holm_fwer_p", "bonferroni_p", "significant_at_005"]].to_string(index=False))

    # Save corrected JSON
    results_json_corrected = {
        "status": "VERIFIED_AND_CLOSED",
        "cohort_name": "eligible_cohort_without_quota_subsampling",
        "total_records": N,
        "n_normal": n_norm_true,
        "n_abnormal": n_abn_true,
        "multiclass_metrics": mc_rows,
        "binary_screening_metrics": bin_rows,
        "confusion_matrices": cm_dict,
        "statistical_tests": stat_rows,
    }
    with open(OUT_DIR / "phase3d_full_cohort_results_corrected.json", "w") as f:
        json.dump(results_json_corrected, f, indent=2)

    print("\nGenerated files:")
    print("  -", OUT_DIR / "phase3d_multiclass_metrics.csv", f"(SHA-256: {sha256_file(OUT_DIR / 'phase3d_multiclass_metrics.csv')})")
    print("  -", OUT_DIR / "phase3d_binary_screening_metrics.csv", f"(SHA-256: {sha256_file(OUT_DIR / 'phase3d_binary_screening_metrics.csv')})")
    print("  -", OUT_DIR / "phase3d_confusion_matrices.json", f"(SHA-256: {sha256_file(OUT_DIR / 'phase3d_confusion_matrices.json')})")
    print("  -", OUT_DIR / "phase3d_statistical_tests_table.csv", f"(SHA-256: {sha256_file(OUT_DIR / 'phase3d_statistical_tests_table.csv')})")
    print("  -", OUT_DIR / "phase3d_full_cohort_results_corrected.json", f"(SHA-256: {sha256_file(OUT_DIR / 'phase3d_full_cohort_results_corrected.json')})")

if __name__ == "__main__":
    main()
