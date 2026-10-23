#!/usr/bin/env python3
"""
Phase 3A: Procedure-Matched MLP Pilot
=============================================================================
Strictly isolates target formulation from procedural discrepancies on a
single fixed outer fold of PTB-XL (50 waveform features, FEAT_50).

Guarantees & Invariants:
1. Complete Data Independence:
   - Outer test fold (N=1,000) completely unseen by teacher and student.
   - Student validation fold (N=800, early-stopping) completely unseen by teacher.
   - Zero patient overlap across train, val, and test partitions.
2. Complete Procedural Matching:
   - Identical architecture: MLP 50 -> 128 -> 64 -> 5.
   - Identical initial PyTorch weights loaded from disk.
   - Identical StandardScaler fit strictly on student train (N=3,200).
   - Identical optimizer: Adam (lr=1e-3, weight_decay=1e-3).
   - Identical batch sequence generated via seeded generator per epoch.
   - Identical loss formulation: Unified Soft-Target Cross-Entropy.
   - Identical early-stopping criterion: Macro F1 on true hard validation labels (patience=15).
3. Ablation Grid:
   - Hard-label control (one-hot)
   - Uniform label smoothing (epsilon = 0.15)
   - Teacher soft-target distillation (alpha = 0.5)
   - Diagnostic controls: Hard-label baseline (64, 32), Legacy scikit-learn models, Teacher TabPFN.
=============================================================================
"""

import copy
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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Set fixed random seeds for deterministic execution
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path("/Volumes/KIOXIA/Projects/gemma3-lora-other")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
EXP_DIR = BASE_DIR / "experiments_phase3"
for d in [RESULTS_DIR, EXP_DIR]:
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


# =============================================================================
# PyTorch Architecture & Soft-Target Cross-Entropy
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
    """
    Unified Soft-Target Cross-Entropy Loss:
    L = - (1/B) * sum_i sum_c targets[i, c] * log_softmax(logits)[i, c]
    When targets are one-hot, this is mathematically and numerically identical
    to standard categorical cross-entropy.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -torch.mean(torch.sum(targets * log_probs, dim=1))


def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE) for multi-class predictions."""
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


# =============================================================================
# Unit Tests & Pre-flight Verifications
# =============================================================================
def run_preflight_checks():
    print("=" * 80)
    print("RUNNING PRE-FLIGHT VERIFICATION CHECKS")
    print("=" * 80)

    # Check 1: One-hot Soft-CE numerical equivalence
    B, C = 64, 5
    logits = torch.randn(B, C, requires_grad=True)
    labels = torch.randint(0, C, (B,))
    oh = F.one_hot(labels, num_classes=C).float()

    loss_hard = F.cross_entropy(logits, labels)
    loss_soft = soft_target_cross_entropy(logits, oh)
    diff = torch.abs(loss_hard - loss_soft).item()
    print(f"Check 1: Soft-CE vs Hard-CE diff = {diff:.2e}")
    assert diff < 1e-6, f"Soft-CE and Hard-CE differ by {diff}!"

    # Check 2: Checkpoint file exists
    assert os.path.isfile(TABPFN_MULTI_CKPT), f"Checkpoint missing: {TABPFN_MULTI_CKPT}"
    print(f"Check 2: TabPFN checkpoint verified at {TABPFN_MULTI_CKPT}")

    print("Pre-flight verification passed successfully!\n")


# =============================================================================
# Main Pilot Routine
# =============================================================================
def main():
    run_preflight_checks()

    # Load dataset
    csv_path = DATA_DIR / "ptbxl_54features.csv"
    assert csv_path.exists(), f"Dataset not found: {csv_path}"
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path.name} (N = {len(df)} rows, {len(FEAT_50)} features)")
    assert len(FEAT_50) == 50, f"Expected 50 waveform features, got {len(FEAT_50)}"

    X_all = df[FEAT_50].values.astype(np.float32)
    y_all = df["target_s1"].values.astype(int)
    patients_all = df["patient_id"].values

    # Step 1: Fixed Outer Fold 1 Partition (N_train = 4000, N_test = 1000)
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_splits = list(sgkf_outer.split(X_all, y_all, groups=patients_all))
    outer_tr_idx, outer_te_idx = outer_splits[0]  # Fold 1

    outer_tr_patients = set(patients_all[outer_tr_idx])
    outer_te_patients = set(patients_all[outer_te_idx])
    overlap_outer = outer_tr_patients & outer_te_patients
    print(f"\nOuter Fold 1 Partition:")
    print(f"  Outer Train: {len(outer_tr_idx)} records ({len(outer_tr_patients)} unique patients)")
    print(f"  Outer Test:  {len(outer_te_idx)} records ({len(outer_te_patients)} unique patients)")
    print(f"  Outer Patient Overlap: {len(overlap_outer)}")
    assert len(overlap_outer) == 0, f"Patient leakage in outer fold! Overlap: {overlap_outer}"

    X_outer_tr_raw = X_all[outer_tr_idx]
    y_outer_tr = y_all[outer_tr_idx]
    patients_outer_tr = patients_all[outer_tr_idx]

    X_outer_te_raw = X_all[outer_te_idx]
    y_outer_te = y_all[outer_te_idx]

    # Step 2: Student Train vs Student Early-Stopping Validation Partition
    # Outer train (4,000) split into Student Train (3,200) and Student Val (800)
    sgkf_student = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    stu_splits = list(sgkf_student.split(X_outer_tr_raw, y_outer_tr, groups=patients_outer_tr))
    stu_tr_rel_idx, stu_val_rel_idx = stu_splits[0]

    stu_tr_patients = set(patients_outer_tr[stu_tr_rel_idx])
    stu_val_patients = set(patients_outer_tr[stu_val_rel_idx])

    print(f"\nStudent Train / Validation Partition (Within Outer Train):")
    print(f"  Student Train: {len(stu_tr_rel_idx)} records ({len(stu_tr_patients)} unique patients)")
    print(f"  Student Val:   {len(stu_val_rel_idx)} records ({len(stu_val_patients)} unique patients)")
    print(f"  Overlap Stu Train & Stu Val: {len(stu_tr_patients & stu_val_patients)}")
    print(f"  Overlap Stu Train & Outer Test: {len(stu_tr_patients & outer_te_patients)}")
    print(f"  Overlap Stu Val & Outer Test: {len(stu_val_patients & outer_te_patients)}")

    assert len(stu_tr_patients & stu_val_patients) == 0, "Patient leakage between Stu Train and Stu Val!"
    assert len(stu_tr_patients & outer_te_patients) == 0, "Patient leakage between Stu Train and Outer Test!"
    assert len(stu_val_patients & outer_te_patients) == 0, "Patient leakage between Stu Val and Outer Test!"

    X_stu_tr_raw = X_outer_tr_raw[stu_tr_rel_idx]
    y_stu_tr = y_outer_tr[stu_tr_rel_idx]
    patients_stu_tr = patients_outer_tr[stu_tr_rel_idx]

    X_stu_val_raw = X_outer_tr_raw[stu_val_rel_idx]
    y_stu_val = y_outer_tr[stu_val_rel_idx]

    # Step 3: Strictly fit StandardScaler inside Student Train only
    scaler = StandardScaler()
    X_stu_tr = scaler.fit_transform(X_stu_tr_raw)
    X_stu_val = scaler.transform(X_stu_val_raw)
    X_outer_te = scaler.transform(X_outer_te_raw)
    print("\nStandardScaler fitted strictly on Student Train (N=3,200).")

    # Step 4: Generate Teacher Soft Targets strictly inside Student Train (N=3,200)
    # Using 3-Fold StratifiedGroupKFold on Student Train to prevent self-prediction bias
    from tabpfn import TabPFNClassifier

    print("\nGenerating Inner 3-Fold OOF Teacher Soft Targets on Student Train (N=3,200)...")
    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    teacher_oof_stu_tr = np.zeros((len(y_stu_tr), NUM_CLASSES), dtype=np.float64)

    t_start_teacher = time.time()
    for inner_fold, (in_tr_idx, in_val_idx) in enumerate(inner_sgkf.split(X_stu_tr, y_stu_tr, groups=patients_stu_tr), 1):
        t0 = time.time()
        in_tr_p = set(patients_stu_tr[in_tr_idx])
        in_val_p = set(patients_stu_tr[in_val_idx])
        assert len(in_tr_p & in_val_p) == 0, "Leakage in inner teacher fold!"

        t_inner = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
        t_inner.fit(X_stu_tr[in_tr_idx], y_stu_tr[in_tr_idx])
        assert np.array_equal(t_inner.classes_, np.arange(NUM_CLASSES)), f"Class mismatch: {t_inner.classes_}"
        probs_in_val = t_inner.predict_proba(X_stu_tr[in_val_idx])
        teacher_oof_stu_tr[in_val_idx] = probs_in_val
        print(f"  Inner Teacher Fold {inner_fold}/3: {len(in_val_idx)} records in {time.time() - t0:.2f}s")

    teacher_gen_time = time.time() - t_start_teacher
    print(f"Teacher soft targets generated in {teacher_gen_time:.2f}s.")

    # Target verifications
    assert teacher_oof_stu_tr.shape == (3200, 5), f"Shape mismatch: {teacher_oof_stu_tr.shape}"
    assert np.all(teacher_oof_stu_tr >= 0.0) and np.all(teacher_oof_stu_tr <= 1.0), "Probabilities out of bounds!"
    assert np.allclose(teacher_oof_stu_tr.sum(axis=1), 1.0, atol=1e-5), "Teacher probabilities do not sum to 1!"

    # Step 5: Construct Target Formulations
    # One-hot hard targets
    Y_stu_tr_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_stu_tr]
    assert Y_stu_tr_oh.shape == (3200, 5)

    # 1. Hard label target (one-hot)
    Y_target_hard = Y_stu_tr_oh.copy()

    # 2. Uniform label smoothing (epsilon = 0.15)
    eps = 0.15
    Y_target_smooth = (1.0 - eps) * Y_stu_tr_oh + (eps / NUM_CLASSES)
    assert np.allclose(Y_target_smooth.sum(axis=1), 1.0, atol=1e-5)

    # 3. Teacher soft-target distillation (alpha = 0.5)
    alpha = 0.5
    Y_target_distill = (1.0 - alpha) * Y_stu_tr_oh + alpha * teacher_oof_stu_tr.astype(np.float32)
    assert np.allclose(Y_target_distill.sum(axis=1), 1.0, atol=1e-5)

    print("\nTarget Formulations Verified:")
    print(f"  Hard-label:    shape={Y_target_hard.shape}, row_sum_mean={Y_target_hard.sum(axis=1).mean():.4f}")
    print(f"  Label-smooth:  shape={Y_target_smooth.shape}, row_sum_mean={Y_target_smooth.sum(axis=1).mean():.4f}, min={Y_target_smooth.min():.4f}, max={Y_target_smooth.max():.4f}")
    print(f"  Teacher-blend: shape={Y_target_distill.shape}, row_sum_mean={Y_target_distill.sum(axis=1).mean():.4f}, min={Y_target_distill.min():.4f}, max={Y_target_distill.max():.4f}")

    # Step 6: Initialize PyTorch Models with Identical Weights
    init_model = PyTorchMLP(in_features=50, h1=128, h2=64, num_classes=NUM_CLASSES)
    init_weights_path = EXP_DIR / "pilot_init_weights_128_64.pt"
    torch.save(init_model.state_dict(), init_weights_path)
    print(f"\nInitial PyTorch weights saved to {init_weights_path.name}")

    # Baseline 64-32 architecture for reference
    init_model_64 = PyTorchMLP(in_features=50, h1=64, h2=32, num_classes=NUM_CLASSES)
    init_weights_64_path = EXP_DIR / "pilot_init_weights_64_32.pt"
    torch.save(init_model_64.state_dict(), init_weights_64_path)

    # Unified Training Function
    def train_pytorch_student(
        model_name: str,
        h1: int,
        h2: int,
        weights_path: Path,
        Y_targets: np.ndarray,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        max_epochs: int = 200,
        patience: int = 15,
    ):
        print(f"\nTraining {model_name} (Architecture: 50 -> {h1} -> {h2} -> 5)...")

        model = PyTorchMLP(in_features=50, h1=h1, h2=h2, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(weights_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        X_tr_t = torch.tensor(X_stu_tr, dtype=torch.float32)
        Y_tr_t = torch.tensor(Y_targets, dtype=torch.float32)
        dataset = TensorDataset(X_tr_t, Y_tr_t)

        X_val_t = torch.tensor(X_stu_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_stu_val, dtype=torch.long)

        best_val_f1 = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = None
        patience_counter = 0

        history = []
        t_tr_start = time.time()

        for epoch in range(1, max_epochs + 1):
            model.train()
            # Deterministic seeded DataLoader per epoch to enforce identical batch sequence
            g = torch.Generator()
            g.manual_seed(SEED + epoch)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = soft_target_cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(dataset)

            # Evaluate on Student Validation set using TRUE HARD LABELS
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_ce_loss = F.cross_entropy(val_logits, y_val_t).item()
                val_probs = F.softmax(val_logits, dim=1).numpy()
                val_preds = np.argmax(val_probs, axis=1)
                val_f1 = float(f1_score(y_stu_val, val_preds, average="macro", zero_division=0) * 100.0)

            history.append({
                "epoch": epoch,
                "train_loss": round(epoch_loss, 4),
                "val_ce_loss": round(val_ce_loss, 4),
                "val_macro_f1": round(val_f1, 2),
            })

            # Early stopping strictly on Validation Macro F1 evaluated on true hard labels
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_loss = val_ce_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        train_duration = time.time() - t_tr_start
        print(f"  Stopped at epoch {epoch} (Best epoch {best_epoch}: Val Macro F1 = {best_val_f1:.2f}%, Val Loss = {best_val_loss:.4f}) in {train_duration:.2f}s")

        # Load best checkpoint and evaluate on Outer Test Set
        model.load_state_dict(best_state_dict)
        model.eval()
        X_te_t = torch.tensor(X_outer_te, dtype=torch.float32)

        t_inf0 = time.perf_counter()
        with torch.no_grad():
            test_logits = model(X_te_t)
            test_probs = F.softmax(test_logits, dim=1).numpy()
        test_inf_ms_per_rec = (time.perf_counter() - t_inf0) * 1000.0 / len(X_outer_te)
        test_preds = np.argmax(test_probs, axis=1)

        test_macro_f1 = float(f1_score(y_outer_te, test_preds, average="macro", zero_division=0) * 100.0)
        test_acc = float(accuracy_score(y_outer_te, test_preds) * 100.0)
        test_bal_acc = float(balanced_accuracy_score(y_outer_te, test_preds) * 100.0)
        test_ece = compute_ece(test_probs, y_outer_te)

        per_class_f1 = f1_score(y_outer_te, test_preds, average=None, zero_division=0) * 100.0
        cm = confusion_matrix(y_outer_te, test_preds).tolist()

        return {
            "model_name": model_name,
            "architecture": f"50-{h1}-{h2}-5",
            "test_macro_f1": round(test_macro_f1, 2),
            "test_accuracy": round(test_acc, 2),
            "test_balanced_accuracy": round(test_bal_acc, 2),
            "test_ece": round(test_ece, 2),
            "per_class_f1": {CLASS_NAMES[c]: round(float(per_class_f1[c]), 2) for c in range(NUM_CLASSES)},
            "confusion_matrix": cm,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
            "best_val_macro_f1": round(best_val_f1, 2),
            "best_val_loss": round(best_val_loss, 4),
            "train_duration_s": round(train_duration, 2),
            "test_infer_ms_per_rec": round(test_inf_ms_per_rec, 4),
            "history": history,
            "probs": test_probs,
            "preds": test_preds,
        }

    # Step 7: Train the Procedure-Matched Triad
    results = {}

    # 1. Hard-Label Control (128, 64)
    results["hard_mlp_128_64"] = train_pytorch_student(
        model_name="Procedure-Matched Hard-Label MLP (128, 64)",
        h1=128, h2=64,
        weights_path=init_weights_path,
        Y_targets=Y_target_hard,
    )

    # 2. Uniform Label Smoothing (128, 64, eps=0.15)
    results["smooth_mlp_128_64"] = train_pytorch_student(
        model_name="Procedure-Matched Uniform Smooth MLP (128, 64, eps=0.15)",
        h1=128, h2=64,
        weights_path=init_weights_path,
        Y_targets=Y_target_smooth,
    )

    # 3. Teacher Soft-Target Distillation (128, 64, alpha=0.5)
    results["distill_mlp_128_64"] = train_pytorch_student(
        model_name="Procedure-Matched Teacher Distilled MLP (128, 64, alpha=0.5)",
        h1=128, h2=64,
        weights_path=init_weights_path,
        Y_targets=Y_target_distill,
    )

    # 4. Hard-Label Baseline Capacity Control (64, 32)
    results["hard_mlp_64_32"] = train_pytorch_student(
        model_name="Procedure-Matched Hard-Label Baseline MLP (64, 32)",
        h1=64, h2=32,
        weights_path=init_weights_64_path,
        Y_targets=Y_target_hard,
    )

    # Step 8: Legacy Scikit-learn Baseline and Distilled MLP Models (for procedural diagnosis)
    print("\nTraining Legacy Scikit-Learn Models (for procedural discrepancy diagnosis)...")
    # Legacy Baseline MLPClassifier (64, 32)
    t0 = time.time()
    legacy_base_64 = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        learning_rate_init=0.001, alpha=0.01, max_iter=250,
        early_stopping=True, random_state=SEED,
    )
    legacy_base_64.fit(X_stu_tr, y_stu_tr)
    t_inf0 = time.perf_counter()
    l_base_64_probs = legacy_base_64.predict_proba(X_outer_te)
    l_base_64_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(X_outer_te)
    l_base_64_preds = np.argmax(l_base_64_probs, axis=1)

    results["legacy_sklearn_base_mlp_64_32"] = {
        "model_name": "Legacy Sklearn MLPClassifier (64, 32)",
        "architecture": "50-64-32-5",
        "test_macro_f1": round(float(f1_score(y_outer_te, l_base_64_preds, average="macro", zero_division=0) * 100.0), 2),
        "test_accuracy": round(float(accuracy_score(y_outer_te, l_base_64_preds) * 100.0), 2),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_outer_te, l_base_64_preds) * 100.0), 2),
        "test_ece": round(compute_ece(l_base_64_probs, y_outer_te), 2),
        "per_class_f1": {CLASS_NAMES[c]: round(float(f1_score(y_outer_te, l_base_64_preds, average=None, zero_division=0)[c] * 100.0), 2) for c in range(NUM_CLASSES)},
        "train_duration_s": round(time.time() - t0, 2),
        "test_infer_ms_per_rec": round(l_base_64_ms, 4),
        "probs": l_base_64_probs,
        "preds": l_base_64_preds,
    }

    # Legacy Capacity-Matched MLPClassifier (128, 64)
    t0 = time.time()
    legacy_base_128 = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        learning_rate_init=0.001, alpha=0.001, max_iter=350,
        early_stopping=True, random_state=SEED,
    )
    legacy_base_128.fit(X_stu_tr, y_stu_tr)
    t_inf0 = time.perf_counter()
    l_base_128_probs = legacy_base_128.predict_proba(X_outer_te)
    l_base_128_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(X_outer_te)
    l_base_128_preds = np.argmax(l_base_128_probs, axis=1)

    results["legacy_sklearn_base_mlp_128_64"] = {
        "model_name": "Legacy Sklearn MLPClassifier (128, 64)",
        "architecture": "50-128-64-5",
        "test_macro_f1": round(float(f1_score(y_outer_te, l_base_128_preds, average="macro", zero_division=0) * 100.0), 2),
        "test_accuracy": round(float(accuracy_score(y_outer_te, l_base_128_preds) * 100.0), 2),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_outer_te, l_base_128_preds) * 100.0), 2),
        "test_ece": round(compute_ece(l_base_128_probs, y_outer_te), 2),
        "per_class_f1": {CLASS_NAMES[c]: round(float(f1_score(y_outer_te, l_base_128_preds, average=None, zero_division=0)[c] * 100.0), 2) for c in range(NUM_CLASSES)},
        "train_duration_s": round(time.time() - t0, 2),
        "test_infer_ms_per_rec": round(l_base_128_ms, 4),
        "probs": l_base_128_probs,
        "preds": l_base_128_preds,
    }

    # Legacy Distilled MLPRegressor (128, 64) with MSE on Logits
    t0 = time.time()
    Y_dist_logits = np.log(np.clip(Y_target_distill, 1e-4, 1.0))
    legacy_dist_128 = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        learning_rate_init=0.001, alpha=0.001, max_iter=350,
        early_stopping=True, random_state=SEED,
    )
    legacy_dist_128.fit(X_stu_tr, Y_dist_logits)
    t_inf0 = time.perf_counter()
    l_dist_logits = legacy_dist_128.predict(X_outer_te)
    l_dist_exp = np.exp(l_dist_logits - np.max(l_dist_logits, axis=1, keepdims=True))
    l_dist_128_probs = l_dist_exp / np.sum(l_dist_exp, axis=1, keepdims=True)
    l_dist_128_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(X_outer_te)
    l_dist_128_preds = np.argmax(l_dist_128_probs, axis=1)

    results["legacy_sklearn_dist_mlp_128_64"] = {
        "model_name": "Legacy Sklearn Distilled MLPRegressor (128, 64, Logit-MSE)",
        "architecture": "50-128-64-5",
        "test_macro_f1": round(float(f1_score(y_outer_te, l_dist_128_preds, average="macro", zero_division=0) * 100.0), 2),
        "test_accuracy": round(float(accuracy_score(y_outer_te, l_dist_128_preds) * 100.0), 2),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_outer_te, l_dist_128_preds) * 100.0), 2),
        "test_ece": round(compute_ece(l_dist_128_probs, y_outer_te), 2),
        "per_class_f1": {CLASS_NAMES[c]: round(float(f1_score(y_outer_te, l_dist_128_preds, average=None, zero_division=0)[c] * 100.0), 2) for c in range(NUM_CLASSES)},
        "train_duration_s": round(time.time() - t0, 2),
        "test_infer_ms_per_rec": round(l_dist_128_ms, 4),
        "probs": l_dist_128_probs,
        "preds": l_dist_128_preds,
    }

    # Step 9: Teacher TabPFN Reference on Outer Test (Fitted on Student Train)
    print("\nEvaluating Teacher TabPFN Reference on Outer Test (Fitted on Student Train N=3,200)...")
    t0 = time.time()
    t_full = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=8)
    t_full.fit(X_stu_tr, y_stu_tr)
    t_fit_time = time.time() - t0

    t_inf0 = time.perf_counter()
    teacher_probs = t_full.predict_proba(X_outer_te)
    teacher_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(X_outer_te)
    teacher_preds = np.argmax(teacher_probs, axis=1)

    results["teacher_tabpfn"] = {
        "model_name": "Teacher TabPFN (n_estimators=8, fitted on Student Train)",
        "architecture": "Transformer Foundation Model",
        "test_macro_f1": round(float(f1_score(y_outer_te, teacher_preds, average="macro", zero_division=0) * 100.0), 2),
        "test_accuracy": round(float(accuracy_score(y_outer_te, teacher_preds) * 100.0), 2),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_outer_te, teacher_preds) * 100.0), 2),
        "test_ece": round(compute_ece(teacher_probs, y_outer_te), 2),
        "per_class_f1": {CLASS_NAMES[c]: round(float(f1_score(y_outer_te, teacher_preds, average=None, zero_division=0)[c] * 100.0), 2) for c in range(NUM_CLASSES)},
        "train_duration_s": round(t_fit_time, 2),
        "test_infer_ms_per_rec": round(teacher_ms, 4),
        "probs": teacher_probs,
        "preds": teacher_preds,
    }

    # Step 10: Comparative Analysis & Differences
    # Procedurally matched pairwise deltas on this outer test fold
    f1_hard_128 = results["hard_mlp_128_64"]["test_macro_f1"]
    f1_smooth_128 = results["smooth_mlp_128_64"]["test_macro_f1"]
    f1_dist_128 = results["distill_mlp_128_64"]["test_macro_f1"]
    f1_hard_64 = results["hard_mlp_64_32"]["test_macro_f1"]
    f1_teacher = results["teacher_tabpfn"]["test_macro_f1"]

    f1_leg_base_64 = results["legacy_sklearn_base_mlp_64_32"]["test_macro_f1"]
    f1_leg_base_128 = results["legacy_sklearn_base_mlp_128_64"]["test_macro_f1"]
    f1_leg_dist_128 = results["legacy_sklearn_dist_mlp_128_64"]["test_macro_f1"]

    deltas = {
        "matched_delta_distill_vs_hard128": round(f1_dist_128 - f1_hard_128, 2),
        "matched_delta_distill_vs_smooth128": round(f1_dist_128 - f1_smooth_128, 2),
        "matched_delta_smooth128_vs_hard128": round(f1_smooth_128 - f1_hard_128, 2),
        "matched_delta_hard128_vs_hard64_capacity_effect": round(f1_hard_128 - f1_hard_64, 2),
        "legacy_delta_distill_vs_base64_manuscript_confounder": round(f1_leg_dist_128 - f1_leg_base_64, 2),
        "legacy_delta_distill_vs_base128": round(f1_leg_dist_128 - f1_leg_base_128, 2),
        "procedure_effect_hard128_pytorch_vs_sklearn": round(f1_hard_128 - f1_leg_base_128, 2),
        "teacher_gap_vs_distilled": round(f1_teacher - f1_dist_128, 2),
    }

    # Save output to JSON (excluding numpy arrays for serialization)
    json_results = {
        "metadata": {
            "experiment": "Phase 3A: Procedure-Matched MLP Pilot",
            "cohort": "PTB-XL (data/ptbxl_54features.csv)",
            "features": "FEAT_50 (waveform only, 50 features)",
            "outer_fold": "Fold 1 of 5-fold StratifiedGroupKFold (seed=42)",
            "patient_disjointness_verified": True,
            "samples": {
                "total": len(df),
                "outer_train": len(outer_tr_idx),
                "student_train": len(stu_tr_rel_idx),
                "student_val": len(stu_val_rel_idx),
                "outer_test": len(outer_te_idx),
            },
            "parameters": {
                "epsilon_label_smoothing": eps,
                "alpha_distillation": alpha,
                "optimizer": "Adam(lr=0.001, weight_decay=0.001)",
                "batch_size": 64,
                "early_stopping": "Macro F1 on true hard student_val labels (patience=15)",
                "loss": "Unified Soft-Target Cross-Entropy",
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "pairwise_deltas_pp": deltas,
        "models": {k: {m: v for m, v in val.items() if m not in ["probs", "preds", "history"]} for k, val in results.items()},
    }

    out_file = RESULTS_DIR / "phase3a_procedure_matched_pilot_results.json"
    with open(out_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved structured pilot results to: {out_file}")

    # Print Final Summary Table
    print("\n" + "=" * 90)
    print("PHASE 3A PILOT SUMMARY: TEST SET RESULTS (FOLD 1, N = 1,000)")
    print("=" * 90)
    print(f"{'Model Description':<48} | {'Macro F1':<9} | {'Acc':<7} | {'ECE':<6} | {'Infer ms':<8}")
    print("-" * 90)
    for k, res in results.items():
        print(f"{res['model_name']:<48} | {res['test_macro_f1']:>7.2f}% | {res['test_accuracy']:>5.2f}% | {res['test_ece']:>4.2f}% | {res['test_infer_ms_per_rec']:>6.3f} ms")
    print("=" * 90)
    print("\nPairwise Macro F1 Deltas (percentage points):")
    for k, v in deltas.items():
        print(f"  {k:<55}: {v:>+6.2f} pp")
    print("=" * 90)


if __name__ == "__main__":
    main()
