#!/usr/bin/env python3
"""
experiments_phase3/phase3d_full_cohort/03_run_phase3d_full_cohort_evaluation.py

Full Cohort (N = 15,160) Rigorous Cross-Validation and Statistical Evaluation
under the Procedure-Matched Protocol.

Population: "Eligible cohort without quota subsampling" (N = 15,160 unique adult patients)
Classes:
  0: NORM     (N = 8,052, 53.11%)
  1: AFIB/AFL (N = 1,155,  7.62%)
  2: CD       (N = 1,647, 10.86%)
  3: STTC/MI  (N = 3,486, 22.99%)
  4: HYP      (N =   820,  5.41%)

Protocol:
  - 5 Outer Folds: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
  - Outer Train: ~12,128 patients; Outer Test: ~3,032 patients
  - Student Train: ~9,702 patients (80% of outer train); Student Val: ~2,426 patients (20%)
  - Inner Teacher Soft Targets: Generated exclusively inside Student Train via 3-fold OOF TabPFN
  - Standardized Student Models: Distilled MLP (128, 64), Uniform Smooth (128, 64, eps=0.15),
    Hard-Label (128, 64), Hard-Label Baseline (64, 32).
  - Standalone TabPFN Reference: Fit on Student Train (9,702), predicted on Outer Test (3,032).
  - Strict Data Leakage Asserts & Identical Initialization across models per fold.
  - Metrics: Macro F1, Balanced Accuracy, Accuracy, Per-Class F1/P/R, Confusion Matrix.
  - Paired Patient Bootstrap (B = 10,000) for Core Family (Holm & Bonferroni) and Exploratory Family.
  - Binary Diagnostic Metrics (Normal vs. Any Abnormality): Sensitivity, Specificity, PPV, NPV.
"""

import copy
import gc
import hashlib
import json
import os
import resource
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
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
DATA_CSV = OUT_DIR / "ptbxl_eligible_cohort_15160_features.csv"
TABPFN_MULTI_CKPT = "/Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"

WEIGHTS_DIR = OUT_DIR / "phase3d_weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5
CLASS_NAMES = ["NORM", "AFIB/AFL", "CD", "STTC/MI", "HYP"]
FEAT_COUNT = 50

def get_process_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

# PyTorch MLP Architecture
class StandardMLP(nn.Module):
    def __init__(self, in_features=50, hidden_dims=(128, 64), num_classes=5, dropout_rate=0.1):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def soft_cross_entropy(logits, target_probs):
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(target_probs * log_probs, dim=-1).mean()

def train_student_model(
    model,
    X_train_t,
    train_targets_t,
    X_val_t,
    y_val_true_t,
    batch_size=64,
    max_epochs=100,
    patience=15,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(X_train_t, train_targets_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_true_onehot = F.one_hot(y_val_true_t, num_classes=NUM_CLASSES).float().to(device)
    X_val_dev = X_val_t.to(device)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    actual_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = soft_cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation using TRUE labels for all models
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_dev)
            val_loss = soft_cross_entropy(val_logits, val_true_onehot).item()

        actual_epochs = epoch
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    return model, actual_epochs, best_val_loss

def predict_model(model, X_test_t, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return probs, preds

def canonical_bootstrap_paired(y_true, p1, p2, n_boot=10000, seed=42):
    """
    Computes canonical 2-sided paired bootstrap difference for Macro F1.
    Formula: p = 2 * (n_opp + 1) / (B + 1)
    """
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
    print("PHASE 3D: FULL COHORT (N = 15,160) RIGOROUS CROSS-VALIDATION")
    print("=" * 80)

    t_all_start = time.time()

    # 1. Load full cohort data
    print(f"\n[Step 1] Loading full cohort features from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    assert len(df) == 15160, f"Expected 15160, got {len(df)}"

    feat_cols = [c for c in df.columns if c.startswith("L1_") or c.startswith("L2_")]
    assert len(feat_cols) == FEAT_COUNT, f"Expected {FEAT_COUNT} features, got {len(feat_cols)}"

    X = df[feat_cols].values.astype(np.float32)
    y = df["target_class"].values.astype(np.int64)
    patient_ids = df["patient_id"].values
    ecg_ids = df["ecg_id"].values

    print(f"Cohort loaded: {len(df)} patients, {len(feat_cols)} waveform features.")
    print("Full Cohort Class Distribution:")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == c_idx))
        pct = count / len(df) * 100.0
        print(f"  Class {c_idx} ({c_name:10s}): {count:5d} ({pct:5.2f}%)")

    # 2. Outer 5-Fold Stratified Group Split
    print("\n[Step 2] Defining 5-fold StratifiedGroupKFold on patient IDs...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = list(sgkf.split(X, y, groups=patient_ids))

    # Output storage for Out-Of-Fold (OOF) predictions
    model_names = [
        "distilled_mlp_128_64",
        "smooth_mlp_128_64",
        "hard_mlp_128_64",
        "hard_mlp_64_32",
        "standalone_tabpfn",
    ]

    oof_probs = {m: np.zeros((len(df), NUM_CLASSES), dtype=np.float32) for m in model_names}
    oof_preds = {m: np.zeros(len(df), dtype=np.int64) for m in model_names}

    fold_manifests = []
    fold_timing_records = []

    # 3. Iterate over 5 Outer Folds
    for fold, (outer_tr_idx, outer_te_idx) in enumerate(outer_splits, 1):
        print("\n" + "-" * 80)
        print(f"STARTING OUTER FOLD {fold} / 5")
        print("-" * 80)
        t_fold_start = time.time()

        # Strict patient independence assertion
        tr_pats = set(patient_ids[outer_tr_idx])
        te_pats = set(patient_ids[outer_te_idx])
        assert len(tr_pats & te_pats) == 0, f"Outer Fold {fold}: Patient leakage between train and test!"

        X_tr_raw, y_tr_raw = X[outer_tr_idx], y[outer_tr_idx]
        X_te_raw, y_te_raw = X[outer_te_idx], y[outer_te_idx]
        p_tr = patient_ids[outer_tr_idx]

        # Inner Split of Outer Train into Student Train (80%) and Student Val (20%)
        inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        stu_tr_rel, stu_val_rel = next(inner_sgkf.split(X_tr_raw, y_tr_raw, groups=p_tr))

        stu_tr_idx = outer_tr_idx[stu_tr_rel]
        stu_val_idx = outer_tr_idx[stu_val_rel]

        stu_tr_pats = set(patient_ids[stu_tr_idx])
        stu_val_pats = set(patient_ids[stu_val_idx])
        assert len(stu_tr_pats & stu_val_pats) == 0, "Student train/val patient leakage!"
        assert len(stu_tr_pats & te_pats) == 0, "Student train/test patient leakage!"
        assert len(stu_val_pats & te_pats) == 0, "Student val/test patient leakage!"

        X_stu_tr = X[stu_tr_idx]
        y_stu_tr = y[stu_tr_idx]
        p_stu_tr = patient_ids[stu_tr_idx]

        X_stu_val = X[stu_val_idx]
        y_stu_val = y[stu_val_idx]

        X_te = X[outer_te_idx]
        y_te = y[outer_te_idx]

        print(f"Fold {fold} Counts: Outer Test = {len(outer_te_idx)}, Outer Train = {len(outer_tr_idx)}")
        print(f"  -> Student Train = {len(stu_tr_idx)} ({len(stu_tr_idx)/len(outer_tr_idx)*100:.1f}%), Student Val = {len(stu_val_idx)} ({len(stu_val_idx)/len(outer_tr_idx)*100:.1f}%)")

        fold_manifest = {
            "fold": fold,
            "outer_test_count": len(outer_te_idx),
            "outer_train_count": len(outer_tr_idx),
            "student_train_count": len(stu_tr_idx),
            "student_val_count": len(stu_val_idx),
            "student_train_classes": {CLASS_NAMES[c]: int(np.sum(y_stu_tr == c)) for c in range(NUM_CLASSES)},
            "outer_test_classes": {CLASS_NAMES[c]: int(np.sum(y_te == c)) for c in range(NUM_CLASSES)},
        }
        fold_manifests.append(fold_manifest)

        # Standard Scaler strictly on Student Train
        scaler = StandardScaler()
        X_stu_tr_s = scaler.fit_transform(X_stu_tr).astype(np.float32)
        X_stu_val_s = scaler.transform(X_stu_val).astype(np.float32)
        X_te_s = scaler.transform(X_te).astype(np.float32)

        # 3a. Generate Teacher Soft Targets via Inner 3-Fold OOF exclusively on Student Train
        print(f"\n[Fold {fold}] Step A: Generating TabPFN inner-OOF teacher targets on Student Train (N={len(stu_tr_idx)})...")
        teacher_oof_stu_tr = np.zeros((len(stu_tr_idx), NUM_CLASSES), dtype=np.float32)

        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        t_teach_start = time.time()

        for in_fold, (in_tr_idx, in_val_idx) in enumerate(inner_skf.split(X_stu_tr_s, y_stu_tr), 1):
            torch.mps.empty_cache()
            torch.mps.synchronize()
            t_in0 = time.time()

            t_inner = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            t_inner.fit(X_stu_tr_s[in_tr_idx], y_stu_tr[in_tr_idx])
            assert np.array_equal(t_inner.classes_, np.arange(NUM_CLASSES)), f"Class mismatch: {t_inner.classes_}"

            torch.mps.synchronize()
            in_probs = t_inner.predict_proba(X_stu_tr_s[in_val_idx])
            torch.mps.synchronize()

            teacher_oof_stu_tr[in_val_idx] = in_probs.astype(np.float32)
            t_in_elapsed = time.time() - t_in0
            print(f"    Inner Fold {in_fold}/3: fit {len(in_tr_idx)} -> pred {len(in_val_idx)} in {t_in_elapsed:.2f}s")

        t_teach_gen = time.time() - t_teach_start
        print(f"  Inner teacher soft targets generated in {t_teach_gen:.2f}s.")
        assert np.allclose(teacher_oof_stu_tr.sum(axis=1), 1.0, atol=1e-4), "Teacher soft targets do not sum to 1.0!"

        # 3b. Fit Standalone TabPFN on full Student Train and predict Outer Test
        print(f"\n[Fold {fold}] Step B: Fitting Standalone TabPFN reference on Student Train and predicting Outer Test...")
        torch.mps.empty_cache()
        torch.mps.synchronize()
        t_stand_start = time.time()

        t_stand = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=8)
        t_stand.fit(X_stu_tr_s, y_stu_tr)

        torch.mps.synchronize()
        probs_tab = t_stand.predict_proba(X_te_s).astype(np.float32)
        torch.mps.synchronize()
        preds_tab = np.argmax(probs_tab, axis=1)

        t_stand_elapsed = time.time() - t_stand_start
        oof_probs["standalone_tabpfn"][outer_te_idx] = probs_tab
        oof_preds["standalone_tabpfn"][outer_te_idx] = preds_tab
        tab_f1 = f1_score(y_te, preds_tab, average="macro", zero_division=0) * 100.0
        print(f"  Standalone TabPFN completed in {t_stand_elapsed:.2f}s | Outer Test Macro F1: {tab_f1:.2f}%")

        # 3c. Prepare Student Target Distributions
        # One-hot true targets
        y_stu_tr_t = torch.from_numpy(y_stu_tr).long()
        y_stu_tr_onehot = F.one_hot(y_stu_tr_t, num_classes=NUM_CLASSES).float()

        # Uniform-Smoothed targets (eps = 0.15)
        eps = 0.15
        y_stu_tr_smooth = (1.0 - eps) * y_stu_tr_onehot + (eps / NUM_CLASSES)

        # Distilled targets (alpha = 0.5)
        alpha = 0.5
        teach_targets_t = torch.from_numpy(teacher_oof_stu_tr).float()
        y_stu_tr_distill = (1.0 - alpha) * y_stu_tr_onehot + alpha * teach_targets_t

        # Tensors for training
        X_stu_tr_t = torch.from_numpy(X_stu_tr_s).float()
        X_stu_val_t = torch.from_numpy(X_stu_val_s).float()
        y_stu_val_t = torch.from_numpy(y_stu_val).long()
        X_te_t = torch.from_numpy(X_te_s).float()

        # Initialize identical starting weights for all MLP models in this fold
        torch.manual_seed(42 + fold)
        base_model_128 = StandardMLP(in_features=FEAT_COUNT, hidden_dims=(128, 64), num_classes=NUM_CLASSES)
        init_weights_128_path = WEIGHTS_DIR / f"init_weights_fold{fold}_128_64.pt"
        torch.save(base_model_128.state_dict(), init_weights_128_path)

        base_model_64 = StandardMLP(in_features=FEAT_COUNT, hidden_dims=(64, 32), num_classes=NUM_CLASSES)
        init_weights_64_path = WEIGHTS_DIR / f"init_weights_fold{fold}_64_32.pt"
        torch.save(base_model_64.state_dict(), init_weights_64_path)

        # 3d. Train Teacher-Distilled MLP (128, 64)
        print(f"\n[Fold {fold}] Step C: Training Teacher-Distilled MLP (128, 64)...")
        m_distill = StandardMLP(in_features=FEAT_COUNT, hidden_dims=(128, 64), num_classes=NUM_CLASSES)
        m_distill.load_state_dict(torch.load(init_weights_128_path))
        t_d0 = time.time()
        m_distill, ep_d, val_loss_d = train_student_model(
            m_distill, X_stu_tr_t, y_stu_tr_distill, X_stu_val_t, y_stu_val_t, device="cpu"
        )
        probs_d, preds_d = predict_model(m_distill, X_te_t, device="cpu")
        oof_probs["distilled_mlp_128_64"][outer_te_idx] = probs_d
        oof_preds["distilled_mlp_128_64"][outer_te_idx] = preds_d
        f1_d = f1_score(y_te, preds_d, average="macro", zero_division=0) * 100.0
        print(f"  Distilled MLP finished in {time.time()-t_d0:.2f}s ({ep_d} eps, best val loss: {val_loss_d:.4f}) | Test Macro F1: {f1_d:.2f}%")

        # 3e. Train Uniform-Smoothed MLP (128, 64)
        print(f"\n[Fold {fold}] Step D: Training Uniform-Smoothed MLP (128, 64)...")
        m_smooth = StandardMLP(in_features=FEAT_COUNT, hidden_dims=(128, 64), num_classes=NUM_CLASSES)
        m_smooth.load_state_dict(torch.load(init_weights_128_path))
        t_s0 = time.time()
        m_smooth, ep_s, val_loss_s = train_student_model(
            m_smooth, X_stu_tr_t, y_stu_tr_smooth, X_stu_val_t, y_stu_val_t, device="cpu"
        )
        probs_s, preds_s = predict_model(m_smooth, X_te_t, device="cpu")
        oof_probs["smooth_mlp_128_64"][outer_te_idx] = probs_s
        oof_preds["smooth_mlp_128_64"][outer_te_idx] = preds_s
        f1_s = f1_score(y_te, preds_s, average="macro", zero_division=0) * 100.0
        print(f"  Uniform Smooth MLP finished in {time.time()-t_s0:.2f}s ({ep_s} eps, best val loss: {val_loss_s:.4f}) | Test Macro F1: {f1_s:.2f}%")

        # 3f. Train Hard-Label MLP (128, 64)
        print(f"\n[Fold {fold}] Step E: Training Hard-Label Capacity-Matched MLP (128, 64)...")
        m_hard128 = StandardMLP(in_features=FEAT_COUNT, hidden_dims=(128, 64), num_classes=NUM_CLASSES)
        m_hard128.load_state_dict(torch.load(init_weights_128_path))
        t_h128_0 = time.time()
        m_hard128, ep_h128, val_loss_h128 = train_student_model(
            m_hard128, X_stu_tr_t, y_stu_tr_onehot, X_stu_val_t, y_stu_val_t, device="cpu"
        )
        probs_h128, preds_h128 = predict_model(m_hard128, X_te_t, device="cpu")
        oof_probs["hard_mlp_128_64"][outer_te_idx] = probs_h128
        oof_preds["hard_mlp_128_64"][outer_te_idx] = preds_h128
        f1_h128 = f1_score(y_te, preds_h128, average="macro", zero_division=0) * 100.0
        print(f"  Hard-Label (128, 64) finished in {time.time()-t_h128_0:.2f}s ({ep_h128} eps, best val loss: {val_loss_h128:.4f}) | Test Macro F1: {f1_h128:.2f}%")

        # 3g. Train Hard-Label Baseline MLP (64, 32)
        print(f"\n[Fold {fold}] Step F: Training Hard-Label Baseline MLP (64, 32)...")
        m_hard64 = StandardMLP(in_features=FEAT_COUNT, hidden_dims=(64, 32), num_classes=NUM_CLASSES)
        m_hard64.load_state_dict(torch.load(init_weights_64_path))
        t_h64_0 = time.time()
        m_hard64, ep_h64, val_loss_h64 = train_student_model(
            m_hard64, X_stu_tr_t, y_stu_tr_onehot, X_stu_val_t, y_stu_val_t, device="cpu"
        )
        probs_h64, preds_h64 = predict_model(m_hard64, X_te_t, device="cpu")
        oof_probs["hard_mlp_64_32"][outer_te_idx] = probs_h64
        oof_preds["hard_mlp_64_32"][outer_te_idx] = preds_h64
        f1_h64 = f1_score(y_te, preds_h64, average="macro", zero_division=0) * 100.0
        print(f"  Hard-Label (64, 32) finished in {time.time()-t_h64_0:.2f}s ({ep_h64} eps, best val loss: {val_loss_h64:.4f}) | Test Macro F1: {f1_h64:.2f}%")

        t_fold_total = time.time() - t_fold_start
        print(f"\n[Fold {fold} COMPLETE] Total Fold Time: {t_fold_total:.2f}s ({t_fold_total/60:.2f} min)")
        fold_timing_records.append({
            "fold": fold,
            "teacher_gen_sec": t_teach_gen,
            "standalone_tabpfn_sec": t_stand_elapsed,
            "distilled_mlp_sec": time.time() - t_d0,
            "smooth_mlp_sec": time.time() - t_s0,
            "hard_mlp_128_sec": time.time() - t_h128_0,
            "hard_mlp_64_sec": time.time() - t_h64_0,
            "total_fold_sec": t_fold_total
        })

    t_all_total = time.time() - t_all_start
    print("\n" + "=" * 80)
    print(f"ALL 5 FOLDS COMPLETED IN {t_all_total:.2f}s ({t_all_total/60:.2f} minutes)!")
    print("=" * 80)

    # 4. Save Raw OOF Predictions
    print("\n[Step 4] Saving combined Out-Of-Fold predictions across all 15,160 patients...")
    oof_npz_path = OUT_DIR / "phase3d_full_cohort_oof_predictions.npz"
    np.savez_compressed(
        oof_npz_path,
        y_true=y,
        patient_ids=patient_ids,
        ecg_ids=ecg_ids,
        **{f"{m}_probs": oof_probs[m] for m in model_names},
        **{f"{m}_preds": oof_preds[m] for m in model_names},
    )
    oof_npz_hash = sha256_file(oof_npz_path)
    print(f"OOF predictions saved to: {oof_npz_path} (SHA-256: {oof_npz_hash})")

    # 5. Compute Full Cohort Metrics for Each Model
    print("\n[Step 5] Computing full cohort classification metrics...")
    model_metrics = {}

    for m in model_names:
        preds = oof_preds[m]
        probs = oof_probs[m]

        macro_f1 = f1_score(y, preds, average="macro", zero_division=0) * 100.0
        bal_acc = balanced_accuracy_score(y, preds) * 100.0
        acc = accuracy_score(y, preds) * 100.0

        per_class_f1 = f1_score(y, preds, average=None, zero_division=0) * 100.0
        per_class_prec = precision_score(y, preds, average=None, zero_division=0) * 100.0
        per_class_rec = recall_score(y, preds, average=None, zero_division=0) * 100.0
        cm = confusion_matrix(y, preds, labels=np.arange(NUM_CLASSES))

        # Binary Screening Metrics: Class 0 = NORM (0), Classes 1..4 = Abnormal (1)
        y_bin_true = (y != 0).astype(int)
        y_bin_pred = (preds != 0).astype(int)

        bin_cm = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1])
        # bin_cm layout: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = bin_cm.ravel()
        sensitivity = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0
        bin_f1 = f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0
        bin_acc = accuracy_score(y_bin_true, y_bin_pred) * 100.0

        model_metrics[m] = {
            "macro_f1": float(macro_f1),
            "balanced_accuracy": float(bal_acc),
            "accuracy": float(acc),
            "per_class": {
                CLASS_NAMES[c]: {
                    "f1": float(per_class_f1[c]),
                    "precision": float(per_class_prec[c]),
                    "recall": float(per_class_rec[c]),
                    "support": int(np.sum(y == c)),
                }
                for c in range(NUM_CLASSES)
            },
            "confusion_matrix": cm.tolist(),
            "binary_screening": {
                "decision_rule": "NORM (class 0) vs. Abnormal (classes 1, 2, 3, 4) mapped from 5-class argmax",
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "ppv": float(ppv),
                "npv": float(npv),
                "binary_f1": float(bin_f1),
                "binary_accuracy": float(bin_acc),
                "confusion_matrix_2x2": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            },
        }

    # 6. Paired Patient Bootstrap Analysis (B = 10,000)
    print("\n[Step 6] Running paired patient bootstrap (B = 10,000) across model comparisons...")
    comparisons = [
        ("distilled_mlp_128_64", "hard_mlp_128_64", "Distilled vs. Hard-Label (128, 64)"),
        ("distilled_mlp_128_64", "smooth_mlp_128_64", "Distilled vs. Uniform-Smoothed (128, 64)"),
        ("smooth_mlp_128_64", "hard_mlp_128_64", "Uniform-Smoothed vs. Hard-Label (128, 64)"),
        ("hard_mlp_128_64", "hard_mlp_64_32", "Hard-Label (128, 64) vs. Baseline (64, 32)"),
        ("standalone_tabpfn", "distilled_mlp_128_64", "Standalone TabPFN vs. Distilled MLP"),
    ]

    boot_results = {}
    for m1, m2, label in comparisons:
        p1 = oof_preds[m1]
        p2 = oof_preds[m2]
        res = canonical_bootstrap_paired(y, p1, p2, n_boot=10000, seed=42)
        res["comparison"] = label
        boot_results[f"{m1}_vs_{m2}"] = res
        print(f"  {label:45s}: Delta = {res['point_diff']:+.2f} pp | 95% CI: [{res['ci_95'][0]:+.2f}, {res['ci_95'][1]:+.2f}] pp | n_opp = {res['n_opp']} | p = {res['p_value']:.4f}")

    # 7. Multi-Comparison Corrections
    # Core 2-test family: Distilled vs Hard, Distilled vs Smooth
    p_dist_hard = boot_results["distilled_mlp_128_64_vs_hard_mlp_128_64"]["p_value"]
    p_dist_smooth = boot_results["distilled_mlp_128_64_vs_smooth_mlp_128_64"]["p_value"]

    # Sort p-values for core family (m = 2)
    core_tests = [
        ("Distilled vs. Hard-Label", p_dist_hard),
        ("Distilled vs. Uniform-Smoothed", p_dist_smooth)
    ]
    core_tests.sort(key=lambda x: x[1])

    # Holm-Bonferroni (step-down)
    # rank 1: p1 * 2
    # rank 2: max(p_adj1, p2 * 1)
    p_core_holm = {}
    p_core_bonf = {}
    p_core_holm[core_tests[0][0]] = min(1.0, core_tests[0][1] * 2.0)
    p_core_holm[core_tests[1][0]] = min(1.0, max(p_core_holm[core_tests[0][0]], core_tests[1][1] * 1.0))
    p_core_bonf[core_tests[0][0]] = min(1.0, core_tests[0][1] * 2.0)
    p_core_bonf[core_tests[1][0]] = min(1.0, core_tests[1][1] * 2.0)

    # Exploratory 5-test family (m = 5)
    exp_tests = [(k, v["p_value"]) for k, v in boot_results.items()]
    exp_tests.sort(key=lambda x: x[1])
    p_exp_holm = {}
    p_exp_bonf = {}
    running_max = 0.0
    for rank, (k, p_val) in enumerate(exp_tests):
        multiplier = len(exp_tests) - rank
        adj_p = min(1.0, max(running_max, p_val * multiplier))
        running_max = adj_p
        p_exp_holm[k] = adj_p
        p_exp_bonf[k] = min(1.0, p_val * len(exp_tests))

    # Compile Final JSON Results
    final_output = {
        "status": "COMPLETED",
        "cohort_name": "eligible_cohort_without_quota_subsampling",
        "cohort_size": 15160,
        "feature_count": FEAT_COUNT,
        "class_names": CLASS_NAMES,
        "fold_manifests": fold_manifests,
        "timing_records": fold_timing_records,
        "total_runtime_seconds": t_all_total,
        "model_metrics": model_metrics,
        "bootstrap_paired_10000": boot_results,
        "multi_comparison_corrections": {
            "core_2test_family": {
                "tests": core_tests,
                "holm_adjusted_p": p_core_holm,
                "bonferroni_adjusted_p": p_core_bonf,
            },
            "exploratory_5test_family": {
                "tests": exp_tests,
                "holm_adjusted_p": p_exp_holm,
                "bonferroni_adjusted_p": p_exp_bonf,
            }
        },
        "oof_predictions_file": str(oof_npz_path),
        "oof_predictions_sha256": oof_npz_hash,
    }

    results_json_path = OUT_DIR / "phase3d_full_cohort_results.json"
    with open(results_json_path, "w") as f:
        json.dump(final_output, f, indent=2)

    results_json_hash = sha256_file(results_json_path)
    print(f"\nResults successfully written to: {results_json_path} (SHA-256: {results_json_hash})")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 3D (N = 15,160) SUMMARY RESULTS:")
    print("=" * 80)
    print(f"{'Model':30s} | {'Macro F1':10s} | {'Bal Acc':10s} | {'Accuracy':10s} | {'Bin Sens':10s} | {'Bin Spec':10s}")
    print("-" * 80)
    for m in model_names:
        met = model_metrics[m]
        b_met = met["binary_screening"]
        print(f"{m:30s} | {met['macro_f1']:6.2f}%    | {met['balanced_accuracy']:6.2f}%    | {met['accuracy']:6.2f}%    | {b_met['sensitivity']:6.2f}%    | {b_met['specificity']:6.2f}%")

    print("\nPairwise Bootstrap Differences:")
    for k, v in boot_results.items():
        print(f"  {v['comparison']:45s}: Point = {v['point_diff']:+.2f} pp [%95 CI: {v['ci_95'][0]:+.2f}, {v['ci_95'][1]:+.2f}] | p = {v['p_value']:.4f}")

    print("\nCore 2-Test Multi-Comparison Corrections:")
    for test_name, p_val in core_tests:
        print(f"  {test_name:35s}: Nominal p = {p_val:.4f} | Holm p = {p_core_holm[test_name]:.4f} | Bonferroni p = {p_core_bonf[test_name]:.4f}")

if __name__ == "__main__":
    main()
