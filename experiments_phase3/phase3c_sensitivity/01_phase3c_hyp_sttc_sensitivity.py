#!/usr/bin/env python3
"""
Phase 3C: HYP + STTC Diagnostic Ambiguity & Alternative Exclusion Sensitivity Analysis
=====================================================================================
Research Question:
Does excluding records with co-occurring HYP and STTC diagnoses (instead of assigning
them to HYP under the hierarchical priority rule) alter the teacher-distillation findings
observed under the procedure-matched setup?

Methodological Framework:
- Reviewer-driven exploratory sensitivity analysis.
- Seed: 42 (matching Phase 3B's seed=42 fold structure).
- Total records excluded: 496 patients (co-occurring HYP and STTC in ptbxl_54features.csv).
- Remaining cohort: 4,504 unique patients (1,045 NORM, 1,045 AFIB/AFL, 1,045 CD, 1,045 STTC/MI, 324 HYP).

Fair Comparison Decomposition:
A. Sample Effect (Subpopulation Shift):
   Evaluate Phase 3B seed=42's saved predictions on the 4,504 common patients without retraining.
B. Training & Diagnostic Ambiguity Sensitivity:
   Retrain models from identical Phase 3B initial weights on the pruned training folds
   with fresh inner TabPFN teacher soft targets, evaluated on the 4,504 common patients.
=====================================================================================
"""

import ast
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path("/Volumes/KIOXIA/Projects/gemma3-lora-other")
DATA_DIR = BASE_DIR / "data"
EXP_DIR = BASE_DIR / "experiments_phase3"
SENS_DIR = EXP_DIR / "phase3c_sensitivity"
WEIGHTS_DIR = EXP_DIR / "phase3b_weights"
PRED_DIR = BASE_DIR / "predictions"
RESULTS_DIR = BASE_DIR / "results"

for d in [SENS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PTBXL_CSV = Path("/Volumes/KIOXIA/Datasets/PTB-XL/ptbxl_database.csv")
PTBXL_SCP = Path("/Volumes/KIOXIA/Datasets/PTB-XL/scp_statements.csv")
TABPFN_MULTI_CKPT = "/Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"

LEAD_SUFFIXES = [
    "pr_int", "qrs_dur", "qt_int", "qtc_frid", "tpte_int",
    "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
    "j_amp", "st60_amp", "st_slope",
    "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_ent",
    "pow_05_5hz", "pow_5_15hz", "pow_15_40hz", "spec_ent", "zcr", "rms", "kurtosis",
]
FEAT_50 = [f"L1_{s}" for s in LEAD_SUFFIXES] + [f"L2_{s}" for s in LEAD_SUFFIXES]
CLASS_NAMES = ["NORM", "AFIB/AFL", "CD", "STTC/MI", "HYP"]
NUM_CLASSES = 5
SEED = 42
N_BOOT = 10000


# =============================================================================
# Architecture & Bootstrap Utilities
# =============================================================================
class PyTorchMLP(nn.Module):
    def __init__(self, in_features=50, h1=128, h2=64, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def soft_target_cross_entropy(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -torch.mean(torch.sum(targets * log_probs, dim=1))


def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece * 100.0)


def paired_patient_bootstrap_canonical(y_true, preds_a, preds_b, n_boot=10000, seed=42):
    """Canonical paired patient-level bootstrap with exact p = 2 * (n_opp + 1) / (B + 1)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        f1_a = f1_score(yt, preds_a[idx], average="macro", zero_division=0) * 100.0
        f1_b = f1_score(yt, preds_b[idx], average="macro", zero_division=0) * 100.0
        diffs[b] = f1_a - f1_b

    point_delta = (
        f1_score(y_true, preds_a, average="macro", zero_division=0)
        - f1_score(y_true, preds_b, average="macro", zero_division=0)
    ) * 100.0

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))

    n_lt_zero = int(np.sum(diffs < 0.0))
    n_eq_zero = int(np.sum(diffs == 0.0))
    n_gt_zero = int(np.sum(diffs > 0.0))
    n_le_zero = int(np.sum(diffs <= 0.0))
    n_ge_zero = int(np.sum(diffs >= 0.0))
    n_opp = min(n_le_zero, n_ge_zero)

    p_canon = min(1.0, 2.0 * (n_opp + 1.0) / (n_boot + 1.0))

    return {
        "point_delta_pp": round(float(point_delta), 2),
        "mean_boot_delta_pp": round(float(np.mean(diffs)), 2),
        "ci_95": [round(ci_low, 2), round(ci_high, 2)],
        "n_boot": n_boot,
        "n_lt_zero": n_lt_zero,
        "n_eq_zero": n_eq_zero,
        "n_gt_zero": n_gt_zero,
        "n_le_zero": n_le_zero,
        "n_ge_zero": n_ge_zero,
        "n_opp": n_opp,
        "p_canonical_nominal": round(float(p_canon), 6),
        "p_canonical_nominal_rounded4": round(float(p_canon), 4),
        "significant_nominal_005": bool(p_canon < 0.05),
    }


def train_single_model_phase3c(
    h1: int,
    h2: int,
    init_weights_path: Path,
    X_tr: np.ndarray,
    Y_targets: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    fold: int,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0.001,
    max_epochs: int = 200,
    patience: int = 15,
):
    model = PyTorchMLP(in_features=50, h1=h1, h2=h2, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(init_weights_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr_t = torch.tensor(Y_targets, dtype=torch.float32)
    dataset = TensorDataset(X_tr_t, Y_tr_t)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    t_tr_start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        g = torch.Generator()
        if fold == 1:
            g.manual_seed(42 + epoch)
        else:
            g.manual_seed(420000 + fold * 1000 + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = soft_target_cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation on true hard labels
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = F.cross_entropy(val_logits, y_val_t).item()
            val_probs = F.softmax(val_logits, dim=1).numpy()
            val_preds = np.argmax(val_probs, axis=1)
            val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0) * 100.0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_duration = time.time() - t_tr_start

    # Evaluate on outer test set
    model.load_state_dict(best_state_dict)
    model.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    t_inf0 = time.perf_counter()
    with torch.no_grad():
        test_logits = model(X_te_t)
        test_probs = F.softmax(test_logits, dim=1).numpy()
    test_infer_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(X_te)
    test_preds = np.argmax(test_probs, axis=1)

    return {
        "probs": test_probs,
        "preds": test_preds,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 2),
        "best_val_loss": round(best_val_loss, 4),
        "train_duration_s": round(train_duration, 2),
        "test_infer_ms": round(test_infer_ms, 4),
    }


# =============================================================================
# Main Phase 3C Routine
# =============================================================================
def main():
    print("=" * 80)
    print("PHASE 3C: HYP + STTC ALTERNATIVE EXCLUSION SENSITIVITY ANALYSIS")
    print("Reviewer-Driven Exploratory Sensitivity Analysis (PTB-XL, Seed 42)")
    print("=" * 80)

    # 1. Parse SCP statements and identify co-occurring HYP+STTC records
    assert PTBXL_CSV.exists(), f"PTB-XL CSV missing: {PTBXL_CSV}"
    assert PTBXL_SCP.exists(), f"PTB-XL SCP missing: {PTBXL_SCP}"

    df_raw = pd.read_csv(PTBXL_CSV)
    scp = pd.read_csv(PTBXL_SCP, index_col=0)
    diag_map = scp[scp["diagnostic"] == 1.0]["diagnostic_class"].to_dict()

    def get_superclasses(scp_str):
        codes = ast.literal_eval(scp_str)
        has_af = ("AFIB" in codes) or ("AFLT" in codes)
        superclasses = set()
        for k, v in codes.items():
            if k in diag_map and v >= 50:
                superclasses.add(diag_map[k])
        return has_af, superclasses

    df_raw["has_af"], df_raw["superclasses"] = zip(*df_raw["scp_codes"].apply(get_superclasses))

    def assign_class_orig(row):
        has_af, superclasses = row["has_af"], row["superclasses"]
        if has_af: return "AFIB/AFL"
        if len(superclasses) == 1:
            sc = list(superclasses)[0]
            if sc == "NORM": return "NORM"
            if sc == "CD": return "CD"
            if sc in ["STTC", "MI"]: return "STTC/MI"
            if sc == "HYP": return "HYP"
        elif superclasses == {"STTC", "MI"}: return "STTC/MI"
        elif "HYP" in superclasses and superclasses.issubset({"HYP", "STTC"}): return "HYP"
        return "MULTI/OTHER"

    def assign_class_alt(row):
        has_af, superclasses = row["has_af"], row["superclasses"]
        if has_af: return "AFIB/AFL"
        if len(superclasses) == 1:
            sc = list(superclasses)[0]
            if sc == "NORM": return "NORM"
            if sc == "CD": return "CD"
            if sc in ["STTC", "MI"]: return "STTC/MI"
            if sc == "HYP": return "HYP"
        elif superclasses == {"STTC", "MI"}: return "STTC/MI"
        return "MULTI/OTHER"

    df_raw["cls_orig"] = df_raw.apply(assign_class_orig, axis=1)
    df_raw["cls_alt"] = df_raw.apply(assign_class_alt, axis=1)
    df_raw["is_hyp_sttc_cooccur"] = (df_raw["cls_orig"] == "HYP") & (df_raw["cls_alt"] == "MULTI/OTHER")

    # Audit Affected Patient Counts
    # A. Raw PTB-XL
    raw_hyp_sttc = df_raw["is_hyp_sttc_cooccur"].sum()

    # B. Eligible Pre-quota Adult Pool (N = 15,160)
    df_adult = df_raw[df_raw["age"] >= 18].copy()
    pat_classes = df_adult[df_adult["cls_orig"] != "MULTI/OTHER"].groupby("patient_id")["cls_orig"].nunique()
    single_pats = pat_classes[pat_classes == 1].index
    df_pool = df_adult[df_adult["patient_id"].isin(single_pats) & (df_adult["cls_orig"] != "MULTI/OTHER")].copy()
    df_pool_first = df_pool.sort_values("ecg_id").drop_duplicates(subset=["patient_id"], keep="first")
    pool_hyp_sttc = df_pool_first["is_hyp_sttc_cooccur"].sum()

    # C. Curated 5,000 Cohort (data/ptbxl_54features.csv)
    df_5k = pd.read_csv(DATA_DIR / "ptbxl_54features.csv")
    merged_5k = df_5k.merge(
        df_raw[["ecg_id", "is_hyp_sttc_cooccur", "superclasses", "scp_codes"]], on="ecg_id", how="left"
    )
    cohort_hyp_sttc = merged_5k["is_hyp_sttc_cooccur"].sum()

    print(f"\n1. Audit of Affected Records (Co-occurring HYP + STTC):")
    print(f"  Raw PTB-XL total records: {raw_hyp_sttc}")
    print(f"  Pre-quota adult eligible pool (N = 15,160): {pool_hyp_sttc} unique patients")
    print(f"  Curated 5,000 cohort (data/ptbxl_54features.csv): {cohort_hyp_sttc} unique patients")
    assert cohort_hyp_sttc == 496, f"Expected 496 affected records, got {cohort_hyp_sttc}"

    # Export excluded records list
    excluded_df = merged_5k[merged_5k["is_hyp_sttc_cooccur"]][["ecg_id", "patient_id", "target_s1", "scp_codes", "superclasses"]].copy()
    excluded_csv_path = SENS_DIR / "excluded_records_hyp_sttc.csv"
    excluded_df.to_csv(excluded_csv_path, index=False)
    print(f"  Saved excluded records list ({len(excluded_df)} records) to: {excluded_csv_path.name}")

    excluded_ecg_ids = set(excluded_df["ecg_id"])
    assert len(excluded_ecg_ids) == 496

    # 2. Partition Structure under Seed 42 and Disjointness Check
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_role_manifest = []

    # Prepare containers for remaining cohort (N = 4,504)
    # Map from original 5,000 row index to common subset mask
    common_mask = ~merged_5k["is_hyp_sttc_cooccur"].values
    assert np.sum(common_mask) == 4504

    df_pruned_all = df_5k[common_mask].reset_index(drop=True)
    y_pruned_all = df_pruned_all["target_s1"].values
    patient_pruned_all = df_pruned_all["patient_id"].values
    ecg_id_pruned_all = df_pruned_all["ecg_id"].values

    print(f"\n2. Remaining Cohort Summary (N = {len(df_pruned_all)} unique patients):")
    for c, cname in enumerate(CLASS_NAMES):
        cnt = np.sum(y_pruned_all == c)
        pct = cnt / len(y_pruned_all) * 100.0
        print(f"  {cname:<10} (Class {c}): {cnt:>4} patients ({pct:>5.2f}%)")

    # Folds setup
    fold_splits_data = []
    for fold, (outer_tr_idx, outer_te_idx) in enumerate(sgkf_outer.split(df_5k, df_5k["target_s1"], df_5k["patient_id"]), 1):
        df_outer_tr = df_5k.iloc[outer_tr_idx].reset_index(drop=True)
        df_outer_te = df_5k.iloc[outer_te_idx].reset_index(drop=True)

        sgkf_sub = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        stu_tr_idx, stu_val_idx = list(sgkf_sub.split(df_outer_tr, df_outer_tr["target_s1"], df_outer_tr["patient_id"]))[0]

        df_stu_tr = df_outer_tr.iloc[stu_tr_idx].reset_index(drop=True)
        df_stu_val = df_outer_tr.iloc[stu_val_idx].reset_index(drop=True)

        # Prune excluded records
        tr_pruned = df_stu_tr[~df_stu_tr["ecg_id"].isin(excluded_ecg_ids)].copy()
        val_pruned = df_stu_val[~df_stu_val["ecg_id"].isin(excluded_ecg_ids)].copy()
        te_pruned = df_outer_te[~df_outer_te["ecg_id"].isin(excluded_ecg_ids)].copy()

        # Strict patient disjointness check
        p_tr = set(tr_pruned["patient_id"])
        p_val = set(val_pruned["patient_id"])
        p_te = set(te_pruned["patient_id"])

        assert len(p_tr & p_val) == 0, f"Leakage tr/val in fold {fold}!"
        assert len(p_tr & p_te) == 0, f"Leakage tr/te in fold {fold}!"
        assert len(p_val & p_te) == 0, f"Leakage val/te in fold {fold}!"

        # Identify which indices in df_pruned_all correspond to this test fold
        te_ecgs = set(te_pruned["ecg_id"])
        te_pruned_indices = np.where(np.isin(ecg_id_pruned_all, list(te_ecgs)))[0]
        assert len(te_pruned_indices) == len(te_pruned)

        fold_splits_data.append({
            "fold": fold,
            "stu_tr": tr_pruned,
            "stu_val": val_pruned,
            "outer_te": te_pruned,
            "te_pruned_indices": te_pruned_indices,
            "counts": {
                "stu_tr": len(tr_pruned),
                "stu_val": len(val_pruned),
                "outer_te": len(te_pruned),
                "total": len(tr_pruned) + len(val_pruned) + len(te_pruned),
                "excluded_in_fold": 496 - (len(tr_pruned) + len(val_pruned) + len(te_pruned) - 4504),
            }
        })

        fold_role_manifest.append({
            "fold": fold,
            "stu_tr_patients": len(p_tr),
            "stu_val_patients": len(p_val),
            "outer_te_patients": len(p_te),
            "patient_disjointness_passed": True,
        })
        print(f"  Fold {fold}: Stu Train={len(tr_pruned)}, Stu Val={len(val_pruned)}, Outer Test={len(te_pruned)} | Leakage = 0")

    with open(SENS_DIR / "phase3c_fold_role_manifest.json", "w") as f:
        json.dump(fold_role_manifest, f, indent=2)

    # =========================================================================
    # Evaluation A: Sample Effect (Subpopulation Shift on Existing Phase 3B Seed 42 Models)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION A: SAMPLE EFFECT (UN-RETRAINED PHASE 3B SEED 42 ON COMMON 4,504 PATIENTS)")
    print("=" * 80)

    npz_3b = np.load(PRED_DIR / "phase3b_repeated_oof_predictions.npz", allow_pickle=True)
    # Seed 42 is index 0 in the npz
    old_preds_5k = {
        "hard_mlp_128_64": npz_3b["oof_preds_hard128"][0],
        "smooth_mlp_128_64": npz_3b["oof_preds_smooth128"][0],
        "distill_mlp_128_64": npz_3b["oof_preds_distill128"][0],
        "hard_mlp_64_32": npz_3b["oof_preds_hard64"][0],
        "teacher_tabpfn": npz_3b["oof_preds_tabpfn"][0],
    }
    old_probs_5k = {
        "hard_mlp_128_64": npz_3b["oof_probs_hard128"][0],
        "smooth_mlp_128_64": npz_3b["oof_probs_smooth128"][0],
        "distill_mlp_128_64": npz_3b["oof_probs_distill128"][0],
        "hard_mlp_64_32": npz_3b["oof_probs_hard64"][0],
        "teacher_tabpfn": npz_3b["oof_probs_tabpfn"][0],
    }

    # Filter to common 4,504 patients
    eval_a_metrics = {}
    eval_a_preds = {}
    eval_a_probs = {}

    for m in old_preds_5k:
        p_common = old_preds_5k[m][common_mask]
        prob_common = old_probs_5k[m][common_mask]
        eval_a_preds[m] = p_common
        eval_a_probs[m] = prob_common

        mf1 = float(f1_score(y_pruned_all, p_common, average="macro", zero_division=0) * 100.0)
        acc = float(accuracy_score(y_pruned_all, p_common) * 100.0)
        bal_acc = float(balanced_accuracy_score(y_pruned_all, p_common) * 100.0)
        ece = compute_ece(prob_common, y_pruned_all)
        p_f1 = f1_score(y_pruned_all, p_common, average=None, zero_division=0) * 100.0
        p_rec = recall_score(y_pruned_all, p_common, average=None, zero_division=0) * 100.0
        cm = confusion_matrix(y_pruned_all, p_common).tolist()

        eval_a_metrics[m] = {
            "macro_f1": round(mf1, 2),
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bal_acc, 2),
            "ece": round(ece, 2),
            "per_class_f1": {CLASS_NAMES[c]: round(float(p_f1[c]), 2) for c in range(NUM_CLASSES)},
            "per_class_recall": {CLASS_NAMES[c]: round(float(p_rec[c]), 2) for c in range(NUM_CLASSES)},
            "confusion_matrix": cm,
        }
        print(f"  {m:<25}: Macro F1 = {mf1:.2f}%, Acc = {acc:.2f}%, HYP F1 = {p_f1[4]:.2f}%, STTC F1 = {p_f1[3]:.2f}%")

    # Bootstrap for Evaluation A
    boot_a_dist_vs_smooth = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_a_preds["distill_mlp_128_64"], eval_a_preds["smooth_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )
    boot_a_dist_vs_hard128 = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_a_preds["distill_mlp_128_64"], eval_a_preds["hard_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )
    print(f"  Evaluation A (Sample Effect) Primary Delta (Distill - Smooth): {boot_a_dist_vs_smooth['point_delta_pp']:+.2f} pp (95% CI: {boot_a_dist_vs_smooth['ci_95']}, p = {boot_a_dist_vs_smooth['p_canonical_nominal']})")
    print(f"  Evaluation A (Sample Effect) Secondary Delta (Distill - Hard128): {boot_a_dist_vs_hard128['point_delta_pp']:+.2f} pp (95% CI: {boot_a_dist_vs_hard128['ci_95']}, p = {boot_a_dist_vs_hard128['p_canonical_nominal']})")

    # =========================================================================
    # Evaluation B: Retrained Models with Diagnostic Ambiguity Exclusion
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION B: RETRAINED MODELS UNDER HYP+STTC EXCLUSION (SEED 42, 5 FOLDS)")
    print("=" * 80)

    eval_b_oof_probs = {m: np.zeros((len(df_pruned_all), NUM_CLASSES), dtype=np.float32) for m in old_preds_5k}
    eval_b_oof_preds = {m: np.zeros(len(df_pruned_all), dtype=int) for m in old_preds_5k}
    fold_train_logs = []

    t_b_start = time.time()
    for f_data in fold_splits_data:
        fold = f_data["fold"]
        f_t0 = time.time()
        print(f"\n--- Retraining Fold {fold}/5 (Stu Train={len(f_data['stu_tr'])}, Stu Val={len(f_data['stu_val'])}, Outer Test={len(f_data['outer_te'])}) ---")

        X_tr_raw = f_data["stu_tr"][FEAT_50].values.astype(np.float32)
        y_tr = f_data["stu_tr"]["target_s1"].values.astype(int)
        p_tr = f_data["stu_tr"]["patient_id"].values

        X_val_raw = f_data["stu_val"][FEAT_50].values.astype(np.float32)
        y_val = f_data["stu_val"]["target_s1"].values.astype(int)

        X_te_raw = f_data["outer_te"][FEAT_50].values.astype(np.float32)
        y_te = f_data["outer_te"]["target_s1"].values.astype(int)
        te_indices = f_data["te_pruned_indices"]

        # 1. Fit StandardScaler strictly on pruned Student Train
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_val = scaler.transform(X_val_raw)
        X_te = scaler.transform(X_te_raw)

        # 2. Generate Fresh Inner 3-Fold Teacher Soft Targets on pruned Student Train
        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        teacher_oof_tr = np.zeros((len(y_tr), NUM_CLASSES), dtype=np.float64)

        t_t0 = time.time()
        for in_tr_idx, in_val_idx in inner_sgkf.split(X_tr, y_tr, groups=p_tr):
            assert len(set(p_tr[in_tr_idx]) & set(p_tr[in_val_idx])) == 0, "Inner teacher leakage in Phase 3C!"
            t_inner = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            t_inner.fit(X_tr[in_tr_idx], y_tr[in_tr_idx])
            assert np.array_equal(t_inner.classes_, np.arange(NUM_CLASSES)), f"Class mismatch: {t_inner.classes_}"
            teacher_oof_tr[in_val_idx] = t_inner.predict_proba(X_tr[in_val_idx])

        teacher_gen_time = time.time() - t_t0
        assert np.allclose(teacher_oof_tr.sum(axis=1), 1.0, atol=1e-5), "Teacher soft targets row sums != 1!"

        # 3. Formulate Targets
        Y_tr_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_tr]
        Y_hard = Y_tr_oh.copy()
        Y_smooth = (1.0 - 0.15) * Y_tr_oh + (0.15 / NUM_CLASSES)
        Y_distill = 0.5 * Y_tr_oh + 0.5 * teacher_oof_tr.astype(np.float32)

        # 4. Load Phase 3B Seed 42 Initial Weights
        if fold == 1:
            w128_path = EXP_DIR / "pilot_init_weights_128_64.pt"
            w64_path = EXP_DIR / "pilot_init_weights_64_32.pt"
        else:
            w128_path = WEIGHTS_DIR / f"init_weights_seed42_fold{fold}_128_64.pt"
            w64_path = WEIGHTS_DIR / f"init_weights_seed42_fold{fold}_64_32.pt"

        assert w128_path.exists(), f"Weights missing: {w128_path}"
        assert w64_path.exists(), f"Weights missing: {w64_path}"

        # 5a. Hard-Label MLP (128, 64)
        res_hard = train_single_model_phase3c(
            h1=128, h2=64, init_weights_path=w128_path,
            X_tr=X_tr, Y_targets=Y_hard, X_val=X_val, y_val=y_val, X_te=X_te, fold=fold
        )
        eval_b_oof_probs["hard_mlp_128_64"][te_indices] = res_hard["probs"]
        eval_b_oof_preds["hard_mlp_128_64"][te_indices] = res_hard["preds"]

        # 5b. Uniform Smooth MLP (128, 64)
        res_smooth = train_single_model_phase3c(
            h1=128, h2=64, init_weights_path=w128_path,
            X_tr=X_tr, Y_targets=Y_smooth, X_val=X_val, y_val=y_val, X_te=X_te, fold=fold
        )
        eval_b_oof_probs["smooth_mlp_128_64"][te_indices] = res_smooth["probs"]
        eval_b_oof_preds["smooth_mlp_128_64"][te_indices] = res_smooth["preds"]

        # 5c. Teacher Distilled MLP (128, 64)
        res_distill = train_single_model_phase3c(
            h1=128, h2=64, init_weights_path=w128_path,
            X_tr=X_tr, Y_targets=Y_distill, X_val=X_val, y_val=y_val, X_te=X_te, fold=fold
        )
        eval_b_oof_probs["distill_mlp_128_64"][te_indices] = res_distill["probs"]
        eval_b_oof_preds["distill_mlp_128_64"][te_indices] = res_distill["preds"]

        # 5d. Hard Baseline MLP (64, 32)
        res_hard64 = train_single_model_phase3c(
            h1=64, h2=32, init_weights_path=w64_path,
            X_tr=X_tr, Y_targets=Y_hard, X_val=X_val, y_val=y_val, X_te=X_te, fold=fold
        )
        eval_b_oof_probs["hard_mlp_64_32"][te_indices] = res_hard64["probs"]
        eval_b_oof_preds["hard_mlp_64_32"][te_indices] = res_hard64["preds"]

        # 5e. Standalone TabPFN Reference on pruned outer test
        t0_tab = time.time()
        t_stand = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=8)
        t_stand.fit(X_tr, y_tr)
        t_fit = time.time() - t0_tab

        t_inf0 = time.perf_counter()
        probs_tab = t_stand.predict_proba(X_te)
        infer_tab = (time.perf_counter() - t_inf0) * 1000.0 / len(X_te)
        preds_tab = np.argmax(probs_tab, axis=1)

        eval_b_oof_probs["teacher_tabpfn"][te_indices] = probs_tab
        eval_b_oof_preds["teacher_tabpfn"][te_indices] = preds_tab

        fold_time = time.time() - f_t0
        f1_d = f1_score(y_te, res_distill["preds"], average="macro") * 100.0
        f1_s = f1_score(y_te, res_smooth["preds"], average="macro") * 100.0
        f1_h = f1_score(y_te, res_hard["preds"], average="macro") * 100.0
        print(f"Fold {fold} complete in {fold_time:.1f}s | F1: Distill={f1_d:.2f}%, Smooth={f1_s:.2f}%, Hard={f1_h:.2f}% (Delta Dist-Sm={f1_d - f1_s:+.2f}pp)")

        fold_train_logs.append({
            "fold": fold,
            "duration_s": round(fold_time, 2),
            "teacher_gen_duration_s": round(teacher_gen_time, 2),
            "fold_test_macro_f1": {
                "distill_mlp_128_64": round(f1_d, 2),
                "smooth_mlp_128_64": round(f1_s, 2),
                "hard_mlp_128_64": round(f1_h, 2),
                "hard_mlp_64_32": round(f1_score(y_te, res_hard64["preds"], average="macro") * 100.0, 2),
                "teacher_tabpfn": round(f1_score(y_te, preds_tab, average="macro") * 100.0, 2),
            }
        })

    eval_b_duration = time.time() - t_b_start
    print(f"\nEvaluation B retraining completed in {eval_b_duration:.1f}s.")

    # Evaluation B Metrics
    eval_b_metrics = {}
    for m in old_preds_5k:
        p_b = eval_b_oof_preds[m]
        prob_b = eval_b_oof_probs[m]

        mf1 = float(f1_score(y_pruned_all, p_b, average="macro", zero_division=0) * 100.0)
        acc = float(accuracy_score(y_pruned_all, p_b) * 100.0)
        bal_acc = float(balanced_accuracy_score(y_pruned_all, p_b) * 100.0)
        ece = compute_ece(prob_b, y_pruned_all)
        p_f1 = f1_score(y_pruned_all, p_b, average=None, zero_division=0) * 100.0
        p_rec = recall_score(y_pruned_all, p_b, average=None, zero_division=0) * 100.0
        cm = confusion_matrix(y_pruned_all, p_b).tolist()

        eval_b_metrics[m] = {
            "macro_f1": round(mf1, 2),
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bal_acc, 2),
            "ece": round(ece, 2),
            "per_class_f1": {CLASS_NAMES[c]: round(float(p_f1[c]), 2) for c in range(NUM_CLASSES)},
            "per_class_recall": {CLASS_NAMES[c]: round(float(p_rec[c]), 2) for c in range(NUM_CLASSES)},
            "confusion_matrix": cm,
        }

    # Bootstrap for Evaluation B (Retrained Models on Common 4,504 Patients)
    boot_b_dist_vs_smooth = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["distill_mlp_128_64"], eval_b_oof_preds["smooth_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )
    boot_b_dist_vs_hard128 = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["distill_mlp_128_64"], eval_b_oof_preds["hard_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )
    boot_b_smooth_vs_hard128 = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["smooth_mlp_128_64"], eval_b_oof_preds["hard_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )
    boot_b_hard128_vs_hard64 = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["hard_mlp_128_64"], eval_b_oof_preds["hard_mlp_64_32"], n_boot=N_BOOT, seed=SEED
    )
    boot_b_tabpfn_vs_dist = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["teacher_tabpfn"], eval_b_oof_preds["distill_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )

    # Retraining Sensitivity: Retrained (B) vs Un-retrained (A) on Common Patients
    retrain_delta_dist = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["distill_mlp_128_64"], eval_a_preds["distill_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )
    retrain_delta_smooth = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["smooth_mlp_128_64"], eval_a_preds["smooth_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )
    retrain_delta_hard = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["hard_mlp_128_64"], eval_a_preds["hard_mlp_128_64"], n_boot=N_BOOT, seed=SEED
    )
    retrain_delta_tabpfn = paired_patient_bootstrap_canonical(
        y_pruned_all, eval_b_oof_preds["teacher_tabpfn"], eval_a_preds["teacher_tabpfn"], n_boot=N_BOOT, seed=SEED
    )

    # Save Output NPZ
    npz_out = SENS_DIR / "phase3c_oof_predictions.npz"
    np.savez_compressed(
        npz_out,
        y_common=y_pruned_all,
        patient_ids=patient_pruned_all,
        ecg_ids=ecg_id_pruned_all,
        eval_a_preds_distill=eval_a_preds["distill_mlp_128_64"],
        eval_a_preds_smooth=eval_a_preds["smooth_mlp_128_64"],
        eval_a_preds_hard128=eval_a_preds["hard_mlp_128_64"],
        eval_a_preds_tabpfn=eval_a_preds["teacher_tabpfn"],
        eval_b_preds_distill=eval_b_oof_preds["distill_mlp_128_64"],
        eval_b_preds_smooth=eval_b_oof_preds["smooth_mlp_128_64"],
        eval_b_preds_hard128=eval_b_oof_preds["hard_mlp_128_64"],
        eval_b_preds_tabpfn=eval_b_oof_preds["teacher_tabpfn"],
    )

    # Save Comprehensive Results JSON
    phase3c_results = {
        "metadata": {
            "analysis_name": "Phase 3C: HYP + STTC Diagnostic Ambiguity Sensitivity Analysis",
            "status": "REVIEWER_DRIVEN_EXPLORATORY_SENSITIVITY_ANALYSIS",
            "seed": SEED,
            "raw_ptbxl_affected_records": int(raw_hyp_sttc),
            "prequota_pool_affected_patients": int(pool_hyp_sttc),
            "curated_cohort_affected_patients": int(cohort_hyp_sttc),
            "original_cohort_size": 5000,
            "pruned_cohort_size": len(df_pruned_all),
            "pruned_class_distribution": {
                CLASS_NAMES[c]: int(np.sum(y_pruned_all == c)) for c in range(NUM_CLASSES)
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "evaluation_a_sample_effect_unretrained": {
            "description": "Evaluation A: Phase 3B Seed 42 models evaluated on common 4,504 test patients without retraining",
            "metrics": eval_a_metrics,
            "bootstrap_comparisons": {
                "distill_vs_smooth": boot_a_dist_vs_smooth,
                "distill_vs_hard128": boot_a_dist_vs_hard128,
            }
        },
        "evaluation_b_retrained_sensitivity": {
            "description": "Evaluation B: Models retrained from Phase 3B initial weights on pruned training folds evaluated on common 4,504 test patients",
            "duration_s": round(eval_b_duration, 2),
            "fold_logs": fold_train_logs,
            "metrics": eval_b_metrics,
            "bootstrap_comparisons": {
                "distill_vs_smooth": boot_b_dist_vs_smooth,
                "distill_vs_hard128": boot_b_dist_vs_hard128,
                "smooth_vs_hard128": boot_b_smooth_vs_hard128,
                "capacity_hard128_vs_hard64": boot_b_hard128_vs_hard64,
                "teacher_vs_distill": boot_b_tabpfn_vs_dist,
            }
        },
        "retraining_sensitivity_deltas_b_minus_a": {
            "description": "Performance delta of retrained model (B) minus un-retrained model (A) on common 4,504 test patients",
            "distill_mlp_retrain_delta": retrain_delta_dist,
            "smooth_mlp_retrain_delta": retrain_delta_smooth,
            "hard_mlp_retrain_delta": retrain_delta_hard,
            "tabpfn_retrain_delta": retrain_delta_tabpfn,
        }
    }

    json_out = SENS_DIR / "phase3c_sensitivity_results.json"
    with open(json_out, "w") as f:
        json.dump(phase3c_results, f, indent=2)

    # Compute Hashes
    manifest = {}
    files_to_hash = [
        SENS_DIR / "01_phase3c_hyp_sttc_sensitivity.py",
        excluded_csv_path,
        SENS_DIR / "phase3c_fold_role_manifest.json",
        json_out,
        npz_out,
    ]
    for p in files_to_hash:
        if p.exists():
            with open(p, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            manifest[p.name] = {"path": str(p), "sha256": h, "size_bytes": p.stat().st_size}

    with open(SENS_DIR / "phase3c_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Print Final Summary Table
    print("\n" + "=" * 95)
    print("PHASE 3C RESULTS SUMMARY (COMMON 4,504 PATIENTS, SEED 42)")
    print("=" * 95)
    print(f"{'Model':<25} | {'Original 3B (N=5k)':<18} | {'Eval A (Un-retrained)':<20} | {'Eval B (Retrained)':<18}")
    print("-" * 95)
    # Original 3B seed 42 numbers
    orig_3b = {
        "distill_mlp_128_64": 73.96,
        "smooth_mlp_128_64": 71.40,
        "hard_mlp_128_64": 72.22,
        "hard_mlp_64_32": 71.96,
        "teacher_tabpfn": 75.57,
    }
    for m in old_preds_5k:
        f1_orig = orig_3b[m]
        f1_a = eval_a_metrics[m]["macro_f1"]
        f1_b = eval_b_metrics[m]["macro_f1"]
        print(f"{m:<25} | {f1_orig:>6.2f}%            | {f1_a:>6.2f}%               | {f1_b:>6.2f}%")
    print("=" * 95)

    print("\nPairwise Comparisons on Retrained Models (Evaluation B, N = 4,504):")
    print(f"  Distilled - Uniform Smooth: {boot_b_dist_vs_smooth['point_delta_pp']:+.2f} pp (95% CI: {boot_b_dist_vs_smooth['ci_95']}, p = {boot_b_dist_vs_smooth['p_canonical_nominal']})")
    print(f"  Distilled - Hard-Label:     {boot_b_dist_vs_hard128['point_delta_pp']:+.2f} pp (95% CI: {boot_b_dist_vs_hard128['ci_95']}, p = {boot_b_dist_vs_hard128['p_canonical_nominal']})")
    print(f"  Uniform Smooth - Hard-Label:{boot_b_smooth_vs_hard128['point_delta_pp']:+.2f} pp (95% CI: {boot_b_smooth_vs_hard128['ci_95']}, p = {boot_b_smooth_vs_hard128['p_canonical_nominal']})")
    print(f"  Teacher TabPFN - Distilled: {boot_b_tabpfn_vs_dist['point_delta_pp']:+.2f} pp (95% CI: {boot_b_tabpfn_vs_dist['ci_95']}, p = {boot_b_tabpfn_vs_dist['p_canonical_nominal']})")

    print("\nDetailed Per-Class F1 for HYP and STTC/MI:")
    print("  HYP Class F1:")
    print(f"    Distilled MLP: Orig = {old_preds_5k['distill_mlp_128_64']} | Eval A = {eval_a_metrics['distill_mlp_128_64']['per_class_f1']['HYP']:.2f}% | Eval B = {eval_b_metrics['distill_mlp_128_64']['per_class_f1']['HYP']:.2f}%")
    print(f"    Smooth MLP:    Orig = ... | Eval A = {eval_a_metrics['smooth_mlp_128_64']['per_class_f1']['HYP']:.2f}% | Eval B = {eval_b_metrics['smooth_mlp_128_64']['per_class_f1']['HYP']:.2f}%")
    print(f"    Hard-Label:    Orig = ... | Eval A = {eval_a_metrics['hard_mlp_128_64']['per_class_f1']['HYP']:.2f}% | Eval B = {eval_b_metrics['hard_mlp_128_64']['per_class_f1']['HYP']:.2f}%")
    print("  STTC/MI Class F1:")
    print(f"    Distilled MLP: Eval A = {eval_a_metrics['distill_mlp_128_64']['per_class_f1']['STTC/MI']:.2f}% | Eval B = {eval_b_metrics['distill_mlp_128_64']['per_class_f1']['STTC/MI']:.2f}%")
    print(f"    Smooth MLP:    Eval A = {eval_a_metrics['smooth_mlp_128_64']['per_class_f1']['STTC/MI']:.2f}% | Eval B = {eval_b_metrics['smooth_mlp_128_64']['per_class_f1']['STTC/MI']:.2f}%")
    print(f"    Hard-Label:    Eval A = {eval_a_metrics['hard_mlp_128_64']['per_class_f1']['STTC/MI']:.2f}% | Eval B = {eval_b_metrics['hard_mlp_128_64']['per_class_f1']['STTC/MI']:.2f}%")
    print("=" * 95)


if __name__ == "__main__":
    main()
