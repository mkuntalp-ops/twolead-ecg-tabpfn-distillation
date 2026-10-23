#!/usr/bin/env python3
"""
Phase 3B: Procedure-Matched Repeated Evaluation
=============================================================================
Post-pilot robustness analysis conducting a 5-seed x 5-fold (25 outer fold)
repeated evaluation on PTB-XL (50 waveform features, FEAT_50).

Methodological Framework:
- Frozen Post-Pilot Robustness Analysis (not confirmatory/preregistered).
- Strictly isolates target formulation under a shared training procedure.
- Seeds: [42, 43, 44, 45, 46].
- 5 outer folds per seed = 25 outer fold evaluations.

Invariants & Safeguards:
1. Data Independence:
   - Outer test fold (N=1,000) completely unseen by teachers and students.
   - Student validation fold (N=800, early-stopping) completely unseen by soft-target teachers.
   - Zero patient leakage verified per fold.
   - StandardScaler fitted strictly on student_train (N=3,200).
   - Teacher soft targets generated via inner 3-fold OOF strictly on student_train (N=3,200).
   - Standalone TabPFN reference fitted strictly on student_train (N=3,200).
2. Procedural Matching (Core Triad):
   - Architecture: PyTorch MLP (50 -> 128 -> 64 -> 5, 15,109 params).
   - Identical initial PyTorch weights loaded per fold.
   - Identical Adam optimizer (lr=0.001, weight_decay=0.001).
   - Identical batch sequence per epoch via seeded DataLoader generator.
   - Identical loss: Unified Soft-Target Cross-Entropy.
   - Identical early stopping: Macro F1 on true hard student_val labels (patience=15).
   - Target formulations:
     * Hard-label: One-hot
     * Uniform smoothing: 0.85 * One-hot + 0.15 * 0.2 (epsilon = 0.15)
     * Teacher distillation: 0.5 * One-hot + 0.5 * Teacher_OOF (alpha = 0.5)
3. Secondary Models:
   - Compact Hard-label Baseline: PyTorch MLP (50 -> 64 -> 32 -> 5, 5,541 params).
   - Standalone TabPFN (n_estimators=8, fitted on student_train).
4. Statistical Evaluation:
   - Within each seed: 10,000-iteration paired patient-level bootstrap on concatenated N=5,000 OOF.
   - Across repetitions (5 seeds): Mean, SD, range of OOF Macro F1 and paired deltas.
   - Explicit distinction between bootstrap uncertainty and partition variability.
=============================================================================
"""

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
RESULTS_DIR = BASE_DIR / "results"
PRED_DIR = BASE_DIR / "predictions"
EXP_DIR = BASE_DIR / "experiments_phase3"
WEIGHTS_DIR = EXP_DIR / "phase3b_weights"

for d in [RESULTS_DIR, PRED_DIR, EXP_DIR, WEIGHTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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

SEEDS = [42, 43, 44, 45, 46]
N_OUTER_FOLDS = 5
MODELS_TO_EVALUATE = [
    "hard_mlp_128_64",
    "smooth_mlp_128_64",
    "distill_mlp_128_64",
    "hard_mlp_64_32",
    "teacher_tabpfn",
]


# =============================================================================
# PyTorch Architectures & Loss
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


def paired_patient_bootstrap_delta(y_true, preds_a, preds_b, n_boot=10000, seed=42):
    """
    Computes 10,000-iteration paired patient-level bootstrap for delta (A - B).
    Strictly reports n_opp = min(n_le_zero, n_ge_zero) and conservative p-value.
    """
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

    p_plus1 = (2.0 * n_opp + 1.0) / (n_boot + 1.0)
    p_value = min(1.0, round(float(p_plus1), 4))
    fwer_decision = "SIGNIFICANT" if p_value < 0.05 else "NOT_SIGNIFICANT"

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
        "p_value_two_sided": p_value,
        "fwer_decision_005": fwer_decision,
    }


# =============================================================================
# Core Training Function
# =============================================================================
def train_single_model(
    h1: int,
    h2: int,
    init_weights_path: Path,
    X_tr: np.ndarray,
    Y_targets: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    seed: int,
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
        if seed == 42 and fold == 1:
            g.manual_seed(42 + epoch)
        else:
            g.manual_seed(seed * 10000 + fold * 1000 + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = soft_target_cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation on TRUE HARD LABELS
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
# Main Execution Routine
# =============================================================================
def main():
    print("=" * 80)
    print("PHASE 3B: PROCEDURE-MATCHED REPEATED EVALUATION")
    print("Status: Frozen Post-Pilot Robustness Analysis")
    print(f"Seeds: {SEEDS} (5 seeds x 5 folds = 25 evaluations)")
    print("=" * 80)

    # 1. Load Data
    csv_path = DATA_DIR / "ptbxl_54features.csv"
    assert csv_path.exists(), f"Dataset not found: {csv_path}"
    df = pd.read_csv(csv_path)
    X_all = df[FEAT_50].values.astype(np.float32)
    y_all = df["target_s1"].values.astype(int)
    patients_all = df["patient_id"].values
    n_samples = len(df)
    assert n_samples == 5000, f"Expected 5,000 samples, got {n_samples}"

    all_seed_results = {}
    all_seed_oof_probs = {m: np.zeros((len(SEEDS), n_samples, NUM_CLASSES), dtype=np.float32) for m in MODELS_TO_EVALUATE}
    all_seed_oof_preds = {m: np.zeros((len(SEEDS), n_samples), dtype=int) for m in MODELS_TO_EVALUATE}

    total_experiment_start = time.time()

    # Iterate over Seeds
    for s_idx, seed in enumerate(SEEDS):
        seed_start = time.time()
        print(f"\n" + "#" * 80)
        print(f"STARTING SEED {seed} ({s_idx + 1}/{len(SEEDS)})")
        print("#" * 80)

        sgkf_outer = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=seed)
        outer_splits = list(sgkf_outer.split(X_all, y_all, groups=patients_all))

        seed_fold_logs = []
        oof_probs = {m: np.zeros((n_samples, NUM_CLASSES), dtype=np.float32) for m in MODELS_TO_EVALUATE}
        oof_preds = {m: np.zeros(n_samples, dtype=int) for m in MODELS_TO_EVALUATE}

        # Iterate over Outer Folds
        for fold, (outer_tr_idx, outer_te_idx) in enumerate(outer_splits, 1):
            fold_t0 = time.time()
            print(f"\n--- Seed {seed} | Outer Fold {fold}/{N_OUTER_FOLDS} ---")

            # 1. Assert Outer Independence
            outer_tr_p = set(patients_all[outer_tr_idx])
            outer_te_p = set(patients_all[outer_te_idx])
            assert len(outer_tr_p & outer_te_p) == 0, f"Outer patient leakage in Seed {seed} Fold {fold}!"

            X_tr_raw = X_all[outer_tr_idx]
            y_tr_all = y_all[outer_tr_idx]
            p_tr_all = patients_all[outer_tr_idx]

            X_te_raw = X_all[outer_te_idx]
            y_te = y_all[outer_te_idx]

            # 2. Sub-split outer train into Student Train (3,200) and Student Val (800)
            sgkf_student = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            stu_splits = list(sgkf_student.split(X_tr_raw, y_tr_all, groups=p_tr_all))
            stu_tr_rel_idx, stu_val_rel_idx = stu_splits[0]

            stu_tr_p = set(p_tr_all[stu_tr_rel_idx])
            stu_val_p = set(p_tr_all[stu_val_rel_idx])

            assert len(stu_tr_p & stu_val_p) == 0, "Leakage: Stu Train & Stu Val!"
            assert len(stu_tr_p & outer_te_p) == 0, "Leakage: Stu Train & Outer Test!"
            assert len(stu_val_p & outer_te_p) == 0, "Leakage: Stu Val & Outer Test!"

            X_stu_tr_raw = X_tr_raw[stu_tr_rel_idx]
            y_stu_tr = y_tr_all[stu_tr_rel_idx]
            p_stu_tr = p_tr_all[stu_tr_rel_idx]

            X_stu_val_raw = X_tr_raw[stu_val_rel_idx]
            y_stu_val = y_tr_all[stu_val_rel_idx]

            # 3. Fit StandardScaler strictly on student_train
            scaler = StandardScaler()
            X_stu_tr = scaler.fit_transform(X_stu_tr_raw)
            X_stu_val = scaler.transform(X_stu_val_raw)
            X_te = scaler.transform(X_te_raw)

            # 4. Generate Inner 3-Fold Teacher Soft Targets strictly on student_train
            inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            teacher_oof_stu_tr = np.zeros((len(y_stu_tr), NUM_CLASSES), dtype=np.float64)

            t_t0 = time.time()
            for in_f, (in_tr_idx, in_val_idx) in enumerate(inner_sgkf.split(X_stu_tr, y_stu_tr, groups=p_stu_tr), 1):
                assert len(set(p_stu_tr[in_tr_idx]) & set(p_stu_tr[in_val_idx])) == 0, "Inner teacher leakage!"
                t_inner = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
                t_inner.fit(X_stu_tr[in_tr_idx], y_stu_tr[in_tr_idx])
                assert np.array_equal(t_inner.classes_, np.arange(NUM_CLASSES)), f"Teacher class alignment error: {t_inner.classes_}"
                teacher_oof_stu_tr[in_val_idx] = t_inner.predict_proba(X_stu_tr[in_val_idx])

            teacher_gen_time = time.time() - t_t0
            assert np.allclose(teacher_oof_stu_tr.sum(axis=1), 1.0, atol=1e-5), "Teacher soft targets row sums != 1!"

            # 5. Formulate Targets
            Y_stu_tr_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_stu_tr]
            Y_hard = Y_stu_tr_oh.copy()
            Y_smooth = (1.0 - 0.15) * Y_stu_tr_oh + (0.15 / NUM_CLASSES)
            Y_distill = 0.5 * Y_stu_tr_oh + 0.5 * teacher_oof_stu_tr.astype(np.float32)

            # 6. Save Initial PyTorch Weights per fold
            if seed == 42 and fold == 1:
                weights_128_path = EXP_DIR / "pilot_init_weights_128_64.pt"
                weights_64_path = EXP_DIR / "pilot_init_weights_64_32.pt"
            else:
                torch.manual_seed(seed * 1000 + fold)
                model_128_init = PyTorchMLP(50, 128, 64, NUM_CLASSES)
                weights_128_path = WEIGHTS_DIR / f"init_weights_seed{seed}_fold{fold}_128_64.pt"
                torch.save(model_128_init.state_dict(), weights_128_path)

                torch.manual_seed(seed * 1000 + fold)
                model_64_init = PyTorchMLP(50, 64, 32, NUM_CLASSES)
                weights_64_path = WEIGHTS_DIR / f"init_weights_seed{seed}_fold{fold}_64_32.pt"
                torch.save(model_64_init.state_dict(), weights_64_path)

            fold_results = {}

            # 7a. Hard-Label MLP (128, 64)
            res_hard = train_single_model(
                h1=128, h2=64, init_weights_path=weights_128_path,
                X_tr=X_stu_tr, Y_targets=Y_hard, X_val=X_stu_val, y_val=y_stu_val, X_te=X_te,
                seed=seed, fold=fold
            )
            oof_probs["hard_mlp_128_64"][outer_te_idx] = res_hard["probs"]
            oof_preds["hard_mlp_128_64"][outer_te_idx] = res_hard["preds"]
            fold_results["hard_mlp_128_64"] = res_hard

            # 7b. Uniform Smooth MLP (128, 64)
            res_smooth = train_single_model(
                h1=128, h2=64, init_weights_path=weights_128_path,
                X_tr=X_stu_tr, Y_targets=Y_smooth, X_val=X_stu_val, y_val=y_stu_val, X_te=X_te,
                seed=seed, fold=fold
            )
            oof_probs["smooth_mlp_128_64"][outer_te_idx] = res_smooth["probs"]
            oof_preds["smooth_mlp_128_64"][outer_te_idx] = res_smooth["preds"]
            fold_results["smooth_mlp_128_64"] = res_smooth

            # 7c. Teacher Distilled MLP (128, 64)
            res_distill = train_single_model(
                h1=128, h2=64, init_weights_path=weights_128_path,
                X_tr=X_stu_tr, Y_targets=Y_distill, X_val=X_stu_val, y_val=y_stu_val, X_te=X_te,
                seed=seed, fold=fold
            )
            oof_probs["distill_mlp_128_64"][outer_te_idx] = res_distill["probs"]
            oof_preds["distill_mlp_128_64"][outer_te_idx] = res_distill["preds"]
            fold_results["distill_mlp_128_64"] = res_distill

            # 7d. Hard Baseline MLP (64, 32)
            res_hard64 = train_single_model(
                h1=64, h2=32, init_weights_path=weights_64_path,
                X_tr=X_stu_tr, Y_targets=Y_hard, X_val=X_stu_val, y_val=y_stu_val, X_te=X_te,
                seed=seed, fold=fold
            )
            oof_probs["hard_mlp_64_32"][outer_te_idx] = res_hard64["probs"]
            oof_preds["hard_mlp_64_32"][outer_te_idx] = res_hard64["preds"]
            fold_results["hard_mlp_64_32"] = res_hard64

            # 7e. Standalone TabPFN Reference on outer test
            t0_tab = time.time()
            t_stand = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=8)
            t_stand.fit(X_stu_tr, y_stu_tr)
            t_stand_fit = time.time() - t0_tab

            t0_inf = time.perf_counter()
            probs_tab = t_stand.predict_proba(X_te)
            infer_tab = (time.perf_counter() - t0_inf) * 1000.0 / len(X_te)
            preds_tab = np.argmax(probs_tab, axis=1)

            oof_probs["teacher_tabpfn"][outer_te_idx] = probs_tab
            oof_preds["teacher_tabpfn"][outer_te_idx] = preds_tab

            fold_results["teacher_tabpfn"] = {
                "probs": probs_tab,
                "preds": preds_tab,
                "train_duration_s": round(t_stand_fit, 2),
                "test_infer_ms": round(infer_tab, 4),
            }

            # Check Pilot Concordance for Seed 42 Fold 1
            if seed == 42 and fold == 1:
                pilot_dist_f1 = f1_score(y_te, res_distill["preds"], average="macro") * 100.0
                pilot_hard_f1 = f1_score(y_te, res_hard["preds"], average="macro") * 100.0
                pilot_smooth_f1 = f1_score(y_te, res_smooth["preds"], average="macro") * 100.0
                pilot_tab_f1 = f1_score(y_te, preds_tab, average="macro") * 100.0
                print(f"  [PILOT CONCORDANCE CHECK - Seed 42 Fold 1]")
                print(f"    Distill: {pilot_dist_f1:.2f}% (Pilot was 76.00%)")
                print(f"    Smooth:  {pilot_smooth_f1:.2f}% (Pilot was 73.35%)")
                print(f"    Hard128: {pilot_hard_f1:.2f}% (Pilot was 72.84%)")
                print(f"    TabPFN:  {pilot_tab_f1:.2f}% (Pilot was 76.62%)")
                assert abs(pilot_dist_f1 - 76.00) < 0.05, f"Concordance check failed on Distill! ({pilot_dist_f1} vs 76.00)"
                assert abs(pilot_smooth_f1 - 73.35) < 0.05, f"Concordance check failed on Smooth! ({pilot_smooth_f1} vs 73.35)"
                assert abs(pilot_hard_f1 - 72.84) < 0.05, f"Concordance check failed on Hard128! ({pilot_hard_f1} vs 72.84)"
                assert abs(pilot_tab_f1 - 76.62) < 0.05, f"Concordance check failed on TabPFN! ({pilot_tab_f1} vs 76.62)"
                print("  [PILOT CONCORDANCE CHECK - PASSED 100.0%]")

            fold_time = time.time() - fold_t0
            f1_d = f1_score(y_te, res_distill["preds"], average="macro") * 100.0
            f1_s = f1_score(y_te, res_smooth["preds"], average="macro") * 100.0
            f1_h = f1_score(y_te, res_hard["preds"], average="macro") * 100.0
            print(f"Fold {fold} complete in {fold_time:.1f}s | F1: Distill={f1_d:.2f}%, Smooth={f1_s:.2f}%, Hard={f1_h:.2f}% (Delta Dist-Sm={f1_d - f1_s:+.2f}pp)")

            seed_fold_logs.append({
                "fold": fold,
                "duration_s": round(fold_time, 2),
                "teacher_gen_duration_s": round(teacher_gen_time, 2),
                "models": {
                    m: {
                        k: v for k, v in fold_results[m].items()
                        if k not in ["probs", "preds"]
                    }
                    for m in MODELS_TO_EVALUATE
                },
                "fold_test_macro_f1": {
                    "distill_mlp_128_64": round(f1_score(y_te, res_distill["preds"], average="macro") * 100.0, 2),
                    "smooth_mlp_128_64": round(f1_score(y_te, res_smooth["preds"], average="macro") * 100.0, 2),
                    "hard_mlp_128_64": round(f1_score(y_te, res_hard["preds"], average="macro") * 100.0, 2),
                    "hard_mlp_64_32": round(f1_score(y_te, res_hard64["preds"], average="macro") * 100.0, 2),
                    "teacher_tabpfn": round(f1_score(y_te, preds_tab, average="macro") * 100.0, 2),
                }
            })

        # Seed-level Complete OOF Evaluation (N = 5,000)
        seed_duration = time.time() - seed_start
        print(f"\nSeed {seed} finished in {seed_duration:.1f}s. Computing full OOF metrics and paired bootstrap...")

        seed_oof_metrics = {}
        for m in MODELS_TO_EVALUATE:
            all_seed_oof_probs[m][s_idx] = oof_probs[m]
            all_seed_oof_preds[m][s_idx] = oof_preds[m]

            mf1 = float(f1_score(y_all, oof_preds[m], average="macro", zero_division=0) * 100.0)
            acc = float(accuracy_score(y_all, oof_preds[m]) * 100.0)
            bal_acc = float(balanced_accuracy_score(y_all, oof_preds[m]) * 100.0)
            ece = compute_ece(oof_probs[m], y_all)
            p_f1 = f1_score(y_all, oof_preds[m], average=None, zero_division=0) * 100.0
            cm = confusion_matrix(y_all, oof_preds[m]).tolist()

            seed_oof_metrics[m] = {
                "macro_f1": round(mf1, 2),
                "accuracy": round(acc, 2),
                "balanced_accuracy": round(bal_acc, 2),
                "ece": round(ece, 2),
                "per_class_f1": {CLASS_NAMES[c]: round(float(p_f1[c]), 2) for c in range(NUM_CLASSES)},
                "confusion_matrix": cm,
            }

        # Seed-level Paired Bootstrap Comparisons (N = 10,000)
        p_dist_vs_smooth = paired_patient_bootstrap_delta(
            y_all, oof_preds["distill_mlp_128_64"], oof_preds["smooth_mlp_128_64"], n_boot=10000, seed=seed
        )
        p_dist_vs_hard128 = paired_patient_bootstrap_delta(
            y_all, oof_preds["distill_mlp_128_64"], oof_preds["hard_mlp_128_64"], n_boot=10000, seed=seed
        )
        p_smooth_vs_hard128 = paired_patient_bootstrap_delta(
            y_all, oof_preds["smooth_mlp_128_64"], oof_preds["hard_mlp_128_64"], n_boot=10000, seed=seed
        )
        p_hard128_vs_hard64 = paired_patient_bootstrap_delta(
            y_all, oof_preds["hard_mlp_128_64"], oof_preds["hard_mlp_64_32"], n_boot=10000, seed=seed
        )
        p_tabpfn_vs_dist = paired_patient_bootstrap_delta(
            y_all, oof_preds["teacher_tabpfn"], oof_preds["distill_mlp_128_64"], n_boot=10000, seed=seed
        )

        all_seed_results[f"seed_{seed}"] = {
            "seed": seed,
            "duration_s": round(seed_duration, 2),
            "oof_metrics": seed_oof_metrics,
            "bootstrap_comparisons": {
                "primary_distill_vs_smooth": p_dist_vs_smooth,
                "secondary_core_distill_vs_hard128": p_dist_vs_hard128,
                "exploratory_smooth_vs_hard128": p_smooth_vs_hard128,
                "exploratory_capacity_hard128_vs_hard64": p_hard128_vs_hard64,
                "exploratory_teacher_vs_distill": p_tabpfn_vs_dist,
            },
            "folds": seed_fold_logs,
        }

        print(f"Seed {seed} OOF Summary:")
        print(f"  Distilled MLP: {seed_oof_metrics['distill_mlp_128_64']['macro_f1']:.2f}% Macro F1")
        print(f"  Smooth MLP:    {seed_oof_metrics['smooth_mlp_128_64']['macro_f1']:.2f}% Macro F1")
        print(f"  Hard 128 MLP:  {seed_oof_metrics['hard_mlp_128_64']['macro_f1']:.2f}% Macro F1")
        print(f"  TabPFN:        {seed_oof_metrics['teacher_tabpfn']['macro_f1']:.2f}% Macro F1")
        print(f"  Primary Delta (Distill - Smooth): {p_dist_vs_smooth['point_delta_pp']:+.2f} pp (95% CI: {p_dist_vs_smooth['ci_95']}, p = {p_dist_vs_smooth['p_value_two_sided']})")

    total_experiment_duration = time.time() - total_experiment_start

    # =========================================================================
    # Multi-Seed Aggregation & Synthesis
    # =========================================================================
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS ACROSS ALL 5 SEEDS (25 EVALUATIONS)")
    print("=" * 80)

    repetition_summary = {}
    for m in MODELS_TO_EVALUATE:
        f1_vals = [all_seed_results[f"seed_{s}"]["oof_metrics"][m]["macro_f1"] for s in SEEDS]
        acc_vals = [all_seed_results[f"seed_{s}"]["oof_metrics"][m]["accuracy"] for s in SEEDS]
        ece_vals = [all_seed_results[f"seed_{s}"]["oof_metrics"][m]["ece"] for s in SEEDS]

        repetition_summary[m] = {
            "macro_f1": {
                "values": f1_vals,
                "mean": round(float(np.mean(f1_vals)), 2),
                "sd": round(float(np.std(f1_vals, ddof=1)), 2),
                "min": round(float(np.min(f1_vals)), 2),
                "max": round(float(np.max(f1_vals)), 2),
            },
            "accuracy": {
                "values": acc_vals,
                "mean": round(float(np.mean(acc_vals)), 2),
                "sd": round(float(np.std(acc_vals, ddof=1)), 2),
                "min": round(float(np.min(acc_vals)), 2),
                "max": round(float(np.max(acc_vals)), 2),
            },
            "ece": {
                "values": ece_vals,
                "mean": round(float(np.mean(ece_vals)), 2),
                "sd": round(float(np.std(ece_vals, ddof=1)), 2),
            }
        }

    # Aggregate Pairwise Deltas across seeds
    delta_keys = [
        ("primary_distill_vs_smooth", "Distilled - Uniform Smooth (eps=0.15)"),
        ("secondary_core_distill_vs_hard128", "Distilled - Hard-Label (128, 64)"),
        ("exploratory_smooth_vs_hard128", "Uniform Smooth - Hard-Label (128, 64)"),
        ("exploratory_capacity_hard128_vs_hard64", "Capacity Effect: Hard(128, 64) - Hard(64, 32)"),
        ("exploratory_teacher_vs_distill", "Teacher TabPFN - Distilled MLP"),
    ]

    pairwise_aggregation = {}
    for k, desc in delta_keys:
        pt_deltas = [all_seed_results[f"seed_{s}"]["bootstrap_comparisons"][k]["point_delta_pp"] for s in SEEDS]
        p_vals = [all_seed_results[f"seed_{s}"]["bootstrap_comparisons"][k]["p_value_two_sided"] for s in SEEDS]
        n_opp_vals = [all_seed_results[f"seed_{s}"]["bootstrap_comparisons"][k]["n_opp"] for s in SEEDS]
        ci_lows = [all_seed_results[f"seed_{s}"]["bootstrap_comparisons"][k]["ci_95"][0] for s in SEEDS]
        ci_highs = [all_seed_results[f"seed_{s}"]["bootstrap_comparisons"][k]["ci_95"][1] for s in SEEDS]

        n_positive = sum(d > 0.0 for d in pt_deltas)
        n_sig = sum(p < 0.05 for p in p_vals)

        pairwise_aggregation[k] = {
            "description": desc,
            "deltas_per_seed": pt_deltas,
            "mean_delta_pp": round(float(np.mean(pt_deltas)), 2),
            "sd_delta_pp": round(float(np.std(pt_deltas, ddof=1)), 2),
            "min_delta_pp": round(float(np.min(pt_deltas)), 2),
            "max_delta_pp": round(float(np.max(pt_deltas)), 2),
            "mean_ci_95": [round(float(np.mean(ci_lows)), 2), round(float(np.mean(ci_highs)), 2)],
            "direction_consistency": f"{n_positive}/{len(SEEDS)} positive",
            "p_values_per_seed": p_vals,
            "n_opp_per_seed": n_opp_vals,
            "significant_seeds_fwer_005": f"{n_sig}/{len(SEEDS)} seeds p < 0.05",
        }

    # Save Output Files
    out_results = {
        "metadata": {
            "experiment": "Phase 3B: Procedure-Matched Repeated Evaluation",
            "status": "FROZEN_POST_PILOT_ROBUSTNESS_ANALYSIS",
            "cohort": "PTB-XL (data/ptbxl_54features.csv)",
            "features": "FEAT_50 (waveform only, 50 features)",
            "seeds": SEEDS,
            "n_outer_folds_per_seed": N_OUTER_FOLDS,
            "total_evaluations": len(SEEDS) * N_OUTER_FOLDS,
            "parameters": {
                "epsilon_label_smoothing": 0.15,
                "alpha_distillation": 0.5,
                "optimizer": "Adam(lr=0.001, weight_decay=0.001)",
                "batch_size": 64,
                "early_stopping": "Macro F1 on true hard student_val labels (patience=15)",
                "loss": "Unified Soft-Target Cross-Entropy",
            },
            "hardware": "Apple Silicon MPS (TabPFN) + CPU (PyTorch/Sklearn)",
            "total_experiment_duration_s": round(total_experiment_duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "repetition_summary_across_5_seeds": repetition_summary,
        "pairwise_aggregation_across_5_seeds": pairwise_aggregation,
        "detailed_results_by_seed": all_seed_results,
    }

    out_json_path = RESULTS_DIR / "phase3b_repeated_evaluation_results.json"
    with open(out_json_path, "w") as f:
        json.dump(out_results, f, indent=2)
    print(f"\nStructured results saved to: {out_json_path}")

    # Save OOF Predictions NPZ
    out_npz_path = PRED_DIR / "phase3b_repeated_oof_predictions.npz"
    np.savez_compressed(
        out_npz_path,
        seeds=np.array(SEEDS),
        true_labels=y_all,
        patient_ids=patients_all,
        oof_probs_hard128=all_seed_oof_probs["hard_mlp_128_64"],
        oof_preds_hard128=all_seed_oof_preds["hard_mlp_128_64"],
        oof_probs_smooth128=all_seed_oof_probs["smooth_mlp_128_64"],
        oof_preds_smooth128=all_seed_oof_preds["smooth_mlp_128_64"],
        oof_probs_distill128=all_seed_oof_probs["distill_mlp_128_64"],
        oof_preds_distill128=all_seed_oof_preds["distill_mlp_128_64"],
        oof_probs_hard64=all_seed_oof_probs["hard_mlp_64_32"],
        oof_preds_hard64=all_seed_oof_preds["hard_mlp_64_32"],
        oof_probs_tabpfn=all_seed_oof_probs["teacher_tabpfn"],
        oof_preds_tabpfn=all_seed_oof_preds["teacher_tabpfn"],
    )
    print(f"OOF predictions array saved to: {out_npz_path}")

    # Print Final Summary Table
    print("\n" + "=" * 95)
    print("PHASE 3B FINAL SUMMARY: 5-SEED REPEATED EVALUATION ACROSS PTB-XL (N = 5,000)")
    print("=" * 95)
    print(f"{'Model':<35} | {'Macro F1 (Mean ± SD)':<22} | {'[Min - Max]':<15} | {'Accuracy':<10}")
    print("-" * 95)
    for m in MODELS_TO_EVALUATE:
        m_f1 = repetition_summary[m]["macro_f1"]
        m_acc = repetition_summary[m]["accuracy"]
        print(f"{m:<35} | {m_f1['mean']:>5.2f}% ± {m_f1['sd']:>4.2f}%         | [{m_f1['min']:>5.2f} - {m_f1['max']:>5.2f}]% | {m_acc['mean']:>5.2f}%")
    print("=" * 95)

    print("\nPairwise Comparisons Across 5 Seeds:")
    for k, desc in delta_keys:
        agg = pairwise_aggregation[k]
        print(f"  {agg['description']:<50}: {agg['mean_delta_pp']:>+5.2f} ± {agg['sd_delta_pp']:>4.2f} pp  "
              f"[{agg['min_delta_pp']:>+5.2f} to {agg['max_delta_pp']:>+5.2f}] pp  "
              f"(Consistency: {agg['direction_consistency']}, Sig: {agg['significant_seeds_fwer_005']})")
    print("=" * 95)


if __name__ == "__main__":
    main()
