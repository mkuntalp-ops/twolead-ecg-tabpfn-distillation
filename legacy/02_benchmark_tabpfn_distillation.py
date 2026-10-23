#!/usr/bin/env python3
"""
02_benchmark_tabpfn_distillation.py

Leakage-Free 5-Fold Stratified Group K-Fold Knowledge Distillation Benchmark
on the Curated Adult PTB-XL Ambulatory Two-Lead (Lead I + Lead II) Discovery Cohort (N = 5,000 Patients).

Evaluates:
  - Task 1: 5-Class Differential Diagnosis (NORM, AFIB/AFL, CD, STTC/MI, HYP)
  - Task 2: Binary Clinical Triage (Outpatient Reassurance [NORM] vs. Urgent Cardiology Escalation)
  - Ablation: 50-Feature Pure Waveform Morphology (removing age, sex, irregular_rr_flag, wide_qrs_flag)
  - Statistical Analysis: 10,000-iteration paired bootstrap 95% confidence intervals and p-values vs. Teacher TabPFN
  - Edge vs. Cloud Inference Latency (ms/recording)
"""

import json
import os
import time
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TABPFN_MODEL_CACHE_DIR"] = "/Volumes/KIOXIA/AI Models/tabpfn"

import joblib
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = Path("/Volumes/KIOXIA/Projects/gemma3-lora-other")
DATA_DIR = BASE_DIR / "data"
PRED_DIR = BASE_DIR / "predictions"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
for d in [PRED_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TABPFN_MULTI_CKPT = "/Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
TABPFN_BIN_CKPT = "/Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_20260417_binary.ckpt"

LEAD_SUFFIXES = [
    "pr_int", "qrs_dur", "qt_int", "qtc_frid", "tpte_int",
    "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
    "j_amp", "st60_amp", "st_slope",
    "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_ent",
    "pow_05_5hz", "pow_5_15hz", "pow_15_40hz", "spec_ent", "zcr", "rms", "kurtosis",
]
FEAT_50 = [f"L1_{s}" for s in LEAD_SUFFIXES] + [f"L2_{s}" for s in LEAD_SUFFIXES]
FEAT_54 = FEAT_50 + ["age", "sex", "irregular_rr_flag", "wide_qrs_flag"]

CLASS_NAMES = ["NORM", "AFIB/AFL", "CD", "STTC/MI", "HYP"]
MODEL_KEYS = ["baseline_xgboost", "distilled_xgboost", "baseline_mlp", "distilled_mlp", "teacher_tabpfn"]


def paired_bootstrap_analysis(y_true, preds_dict, probs_dict, is_binary=False, n_boot=10000, seed=42):
    """Computes 10,000-iteration paired patient-level bootstrap 95% CIs and two-sided p-values vs. Teacher TabPFN."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_f1 = {k: np.zeros(n_boot, dtype=np.float64) for k in MODEL_KEYS}

    # Vectorized index generation for 10,000 bootstrap resamples
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        for k in MODEL_KEYS:
            boot_f1[k][b] = f1_score(yt, preds_dict[k][idx], average="macro", zero_division=0) * 100.0

    teacher_scores = boot_f1["teacher_tabpfn"]
    stats_out = {}
    for k in MODEL_KEYS:
        scores = boot_f1[k]
        ci_low = float(np.percentile(scores, 2.5))
        ci_high = float(np.percentile(scores, 97.5))
        if k == "teacher_tabpfn":
            p_val = None
        else:
            diff = teacher_scores - scores
            # Two-sided non-parametric bootstrap p-value
            p_one = float(np.mean(diff <= 0.0))
            p_two = min(1.0, 2.0 * min(p_one, 1.0 - p_one))
            p_val = round(p_two, 4)
        stats_out[k] = {
            "macro_f1_mean_boot": round(float(np.mean(scores)), 2),
            "macro_f1_ci_95": [round(ci_low, 2), round(ci_high, 2)],
            "p_value_vs_tabpfn": p_val,
        }
    return stats_out, boot_f1


def run_task1_5class(df: pd.DataFrame, feat_cols: list[str], tag: str = "54feat", alpha: float = 0.5):
    print("\n" + "#" * 80, flush=True)
    print(f"TASK 1 (5-CLASS DIFFERENTIAL DIAGNOSIS) - [{tag.upper()}] (N = {len(df)} PATIENTS)", flush=True)
    print("#" * 80, flush=True)

    X_raw = df[feat_cols].values.astype(np.float32)
    y = df["target_s1"].values.astype(int)
    groups = df["patient_id"].values
    num_classes = 5

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = {k: np.zeros(len(y), dtype=int) for k in MODEL_KEYS}
    oof_probs = {k: np.zeros((len(y), num_classes), dtype=np.float64) for k in MODEL_KEYS}
    infer_ms_per_rec = {k: [] for k in MODEL_KEYS}

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_raw, y, groups=groups), 1):
        t_fold = time.time()
        assert len(set(groups[tr_idx]) & set(groups[val_idx])) == 0, "Patient leakage detected!"

        # Strictly fit StandardScaler inside training fold only
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[tr_idx])
        X_val = scaler.transform(X_raw[val_idx])
        y_tr, y_val = y[tr_idx], y[val_idx]
        g_tr = groups[tr_idx]

        # 1. Inner 3-Fold StratifiedGroupKFold for genuine OOF Teacher Soft Labels
        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        teacher_oof_tr = np.zeros((len(y_tr), num_classes), dtype=np.float64)
        for _, (i_tr, i_val) in enumerate(inner_sgkf.split(X_tr, y_tr, g_tr)):
            t_in = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            t_in.fit(X_tr[i_tr], y_tr[i_tr])
            teacher_oof_tr[i_val] = t_in.predict_proba(X_tr[i_val])

        # Fit Teacher TabPFN on full training fold and evaluate on validation fold
        t0 = time.perf_counter()
        tabpfn_full = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=8)
        tabpfn_full.fit(X_tr, y_tr)
        t_probs = tabpfn_full.predict_proba(X_val)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / len(X_val)
        t_preds = np.argmax(t_probs, axis=1)
        oof_probs["teacher_tabpfn"][val_idx] = t_probs
        oof_preds["teacher_tabpfn"][val_idx] = t_preds
        infer_ms_per_rec["teacher_tabpfn"].append(dt_ms)

        # Blended Distillation Target
        Y_tr_oh = np.eye(num_classes)[y_tr]
        Y_distill = (1.0 - alpha) * Y_tr_oh + alpha * teacher_oof_tr

        # 2a. Baseline XGBoost (Lightweight Microcontroller Edge Configuration)
        xgb_base = XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss",
            random_state=42, n_jobs=4,
        )
        xgb_base.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        xb_probs = xgb_base.predict_proba(X_val)
        infer_ms_per_rec["baseline_xgboost"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        xb_preds = np.argmax(xb_probs, axis=1)
        oof_probs["baseline_xgboost"][val_idx] = xb_probs
        oof_preds["baseline_xgboost"][val_idx] = xb_preds

        # 2b. Distilled XGBoost (MultiOutputRegressor with reg:logistic on Teacher-Blended Soft Targets)
        xgb_dist = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:logistic", random_state=42, n_jobs=4,
            )
        )
        xgb_dist.fit(X_tr, Y_distill)
        t0 = time.perf_counter()
        xd_raw = np.clip(xgb_dist.predict(X_val), 1e-6, None)
        xd_probs = xd_raw / np.sum(xd_raw, axis=1, keepdims=True)
        infer_ms_per_rec["distilled_xgboost"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        xd_preds = np.argmax(xd_probs, axis=1)
        oof_probs["distilled_xgboost"][val_idx] = xd_probs
        oof_preds["distilled_xgboost"][val_idx] = xd_preds

        # 3a. Baseline MLP
        mlp_base = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            learning_rate_init=0.001, alpha=0.01, max_iter=250,
            early_stopping=True, random_state=42,
        )
        mlp_base.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        mb_probs = mlp_base.predict_proba(X_val)
        infer_ms_per_rec["baseline_mlp"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        mb_preds = np.argmax(mb_probs, axis=1)
        oof_probs["baseline_mlp"][val_idx] = mb_probs
        oof_preds["baseline_mlp"][val_idx] = mb_preds

        # 3b. Distilled MLP (Logit-Space Knowledge Distillation, Hinton et al., 2015)
        Y_dist_logits = np.log(np.clip(Y_distill, 1e-4, 1.0))
        mlp_dist = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            learning_rate_init=0.001, alpha=0.001, max_iter=350,
            early_stopping=True, random_state=42,
        )
        mlp_dist.fit(X_tr, Y_dist_logits)
        t0 = time.perf_counter()
        md_logits = mlp_dist.predict(X_val)
        md_exp = np.exp(md_logits - np.max(md_logits, axis=1, keepdims=True))
        md_probs = md_exp / np.sum(md_exp, axis=1, keepdims=True)
        infer_ms_per_rec["distilled_mlp"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        md_preds = np.argmax(md_probs, axis=1)
        oof_probs["distilled_mlp"][val_idx] = md_probs
        oof_preds["distilled_mlp"][val_idx] = md_preds

        print(
            f"  Fold {fold}/5 ({time.time() - t_fold:.1f}s) Macro F1 -> "
            f"TabPFN: {f1_score(y_val, t_preds, average='macro')*100:.2f}% | "
            f"DistXGB: {f1_score(y_val, xd_preds, average='macro')*100:.2f}% (Base: {f1_score(y_val, xb_preds, average='macro')*100:.2f}%) | "
            f"DistMLP: {f1_score(y_val, md_preds, average='macro')*100:.2f}% (Base: {f1_score(y_val, mb_preds, average='macro')*100:.2f}%)",
            flush=True,
        )

    boot_stats, boot_arrays = paired_bootstrap_analysis(y, oof_preds, oof_probs, is_binary=False, n_boot=10000)

    summary = {}
    for k in MODEL_KEYS:
        pr = oof_preds[k]
        pb = oof_probs[k]
        f1_mac = float(f1_score(y, pr, average="macro") * 100.0)
        acc = float(accuracy_score(y, pr) * 100.0)
        bal = float(balanced_accuracy_score(y, pr) * 100.0)
        auc_ovr = float(roc_auc_score(y, pb, multi_class="ovr", average="macro") * 100.0)
        per_cls_f1 = [round(float(x * 100.0), 2) for x in f1_score(y, pr, average=None)]
        cm = confusion_matrix(y, pr).tolist()
        summary[k] = {
            "patient_macro_f1": round(f1_mac, 2),
            "patient_accuracy": round(acc, 2),
            "patient_balanced_accuracy": round(bal, 2),
            "roc_auc_ovr": round(auc_ovr, 2),
            "per_class_f1": dict(zip(CLASS_NAMES, per_cls_f1)),
            "confusion_matrix": cm,
            "latency_ms_per_rec": round(float(np.mean(infer_ms_per_rec[k])), 4),
            "bootstrap_10k": boot_stats[k],
        }
        print(
            f"  * {k:<18}: Macro F1 = {f1_mac:.2f}% {boot_stats[k]['macro_f1_ci_95']} "
            f"(p={boot_stats[k]['p_value_vs_tabpfn']}) | Acc = {acc:.2f}% | BalAcc = {bal:.2f}% | OVR AUC = {auc_ovr:.2f}%",
            flush=True,
        )

    # Save OOF predictions and bootstrap distributions
    oof_df = pd.DataFrame({"patient_id": groups, "true_label": y})
    for k in MODEL_KEYS:
        oof_df[f"pred_{k}"] = oof_preds[k]
        for c in range(num_classes):
            oof_df[f"prob_{k}_c{c}"] = oof_probs[k][:, c]
    oof_df.to_csv(PRED_DIR / f"oof_task1_{tag}.csv", index=False)
    np.savez_compressed(PRED_DIR / f"boot10k_task1_{tag}.npz", **boot_arrays)
    return summary


def run_task2_binary(df: pd.DataFrame, feat_cols: list[str], tag: str = "54feat", alpha: float = 0.5):
    print("\n" + "#" * 80, flush=True)
    print(f"TASK 2 (BINARY CLINICAL TRIAGE) - [{tag.upper()}] (N = {len(df)} PATIENTS)", flush=True)
    print("#" * 80, flush=True)

    X_raw = df[feat_cols].values.astype(np.float32)
    y = df["target_s2"].values.astype(int)
    groups = df["patient_id"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = {k: np.zeros(len(y), dtype=int) for k in MODEL_KEYS}
    oof_probs = {k: np.zeros(len(y), dtype=np.float64) for k in MODEL_KEYS}
    infer_ms_per_rec = {k: [] for k in MODEL_KEYS}

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_raw, y, groups=groups), 1):
        t_fold = time.time()
        assert len(set(groups[tr_idx]) & set(groups[val_idx])) == 0, "Patient leakage detected!"

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[tr_idx])
        X_val = scaler.transform(X_raw[val_idx])
        y_tr, y_val = y[tr_idx], y[val_idx]
        g_tr = groups[tr_idx]

        # 1. Inner 3-Fold OOF Teacher Soft Labels
        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        teacher_oof_tr = np.zeros(len(y_tr), dtype=np.float64)
        for _, (i_tr, i_val) in enumerate(inner_sgkf.split(X_tr, y_tr, g_tr)):
            t_in = TabPFNClassifier(model_path=TABPFN_BIN_CKPT, device="mps", n_estimators=4)
            t_in.fit(X_tr[i_tr], y_tr[i_tr])
            teacher_oof_tr[i_val] = t_in.predict_proba(X_tr[i_val])[:, 1]

        t0 = time.perf_counter()
        tabpfn_full = TabPFNClassifier(model_path=TABPFN_BIN_CKPT, device="mps", n_estimators=8)
        tabpfn_full.fit(X_tr, y_tr)
        t_prob = tabpfn_full.predict_proba(X_val)[:, 1]
        infer_ms_per_rec["teacher_tabpfn"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        t_pred = (t_prob >= 0.5).astype(int)
        oof_probs["teacher_tabpfn"][val_idx] = t_prob
        oof_preds["teacher_tabpfn"][val_idx] = t_pred

        y_distill = (1.0 - alpha) * y_tr + alpha * teacher_oof_tr

        # 2a. Baseline XGBoost
        xgb_base = XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=42, n_jobs=4,
        )
        xgb_base.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        xb_prob = xgb_base.predict_proba(X_val)[:, 1]
        infer_ms_per_rec["baseline_xgboost"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        xb_pred = (xb_prob >= 0.5).astype(int)
        oof_probs["baseline_xgboost"][val_idx] = xb_prob
        oof_preds["baseline_xgboost"][val_idx] = xb_pred

        # 2b. Distilled XGBoost
        xgb_dist = XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.06,
            objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=4,
        )
        xgb_dist.fit(X_tr, y_distill)
        t0 = time.perf_counter()
        xd_prob = np.clip(xgb_dist.predict(X_val), 0.0, 1.0)
        infer_ms_per_rec["distilled_xgboost"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        xd_pred = (xd_prob >= 0.5).astype(int)
        oof_probs["distilled_xgboost"][val_idx] = xd_prob
        oof_preds["distilled_xgboost"][val_idx] = xd_pred

        # 3a. Baseline MLP
        mlp_base = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            learning_rate_init=0.001, alpha=0.001, max_iter=300,
            early_stopping=True, random_state=42,
        )
        mlp_base.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        mb_prob = mlp_base.predict_proba(X_val)[:, 1]
        infer_ms_per_rec["baseline_mlp"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        mb_pred = (mb_prob >= 0.5).astype(int)
        oof_probs["baseline_mlp"][val_idx] = mb_prob
        oof_preds["baseline_mlp"][val_idx] = mb_pred

        # 3b. Distilled MLP
        mlp_dist = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            learning_rate_init=0.001, alpha=0.001, max_iter=350,
            early_stopping=True, random_state=42,
        )
        mlp_dist.fit(X_tr, y_distill)
        t0 = time.perf_counter()
        md_prob = np.clip(mlp_dist.predict(X_val), 0.0, 1.0)
        infer_ms_per_rec["distilled_mlp"].append((time.perf_counter() - t0) * 1000.0 / len(X_val))
        md_pred = (md_prob >= 0.5).astype(int)
        oof_probs["distilled_mlp"][val_idx] = md_prob
        oof_preds["distilled_mlp"][val_idx] = md_pred

        print(
            f"  Fold {fold}/5 ({time.time() - t_fold:.1f}s) Macro F1 -> "
            f"TabPFN: {f1_score(y_val, t_pred, average='macro')*100:.2f}% | "
            f"DistXGB: {f1_score(y_val, xd_pred, average='macro')*100:.2f}% (Base: {f1_score(y_val, xb_pred, average='macro')*100:.2f}%) | "
            f"DistMLP: {f1_score(y_val, md_pred, average='macro')*100:.2f}% (Base: {f1_score(y_val, mb_pred, average='macro')*100:.2f}%)",
            flush=True,
        )

    boot_stats, boot_arrays = paired_bootstrap_analysis(y, oof_preds, oof_probs, is_binary=True, n_boot=10000)

    summary = {}
    for k in MODEL_KEYS:
        pr = oof_preds[k]
        pb = oof_probs[k]
        f1_mac = float(f1_score(y, pr, average="macro") * 100.0)
        acc = float(accuracy_score(y, pr) * 100.0)
        bal = float(balanced_accuracy_score(y, pr) * 100.0)
        auc = float(roc_auc_score(y, pb) * 100.0)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn) * 100.0)
        spec = float(tn / (tn + fp) * 100.0)
        summary[k] = {
            "patient_macro_f1": round(f1_mac, 2),
            "patient_accuracy": round(acc, 2),
            "patient_balanced_accuracy": round(bal, 2),
            "roc_auc": round(auc, 2),
            "sensitivity_urgent": round(sens, 2),
            "specificity_reassure": round(spec, 2),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "latency_ms_per_rec": round(float(np.mean(infer_ms_per_rec[k])), 4),
            "bootstrap_10k": boot_stats[k],
        }
        print(
            f"  * {k:<18}: Macro F1 = {f1_mac:.2f}% {boot_stats[k]['macro_f1_ci_95']} "
            f"(p={boot_stats[k]['p_value_vs_tabpfn']}) | Acc = {acc:.2f}% | Sens = {sens:.2f}% | Spec = {spec:.2f}% | AUC = {auc:.2f}%",
            flush=True,
        )

    oof_df = pd.DataFrame({"patient_id": groups, "true_label": y})
    for k in MODEL_KEYS:
        oof_df[f"pred_{k}"] = oof_preds[k]
        oof_df[f"prob_{k}"] = oof_probs[k]
    oof_df.to_csv(PRED_DIR / f"oof_task2_{tag}.csv", index=False)
    np.savez_compressed(PRED_DIR / f"boot10k_task2_{tag}.npz", **boot_arrays)
    return summary


def main():
    df_ptb = pd.read_csv(DATA_DIR / "ptbxl_54features.csv")
    print(f"Loaded PTB-XL Discovery Cohort: {len(df_ptb)} unique adult patients.", flush=True)

    res_all = {
        "cohort": "PTB-XL Curated Adult Single-Superclass Discovery Cohort (N=5,000 patients)",
        "task1_54feat": run_task1_5class(df_ptb, FEAT_54, tag="54feat", alpha=0.5),
        "task2_54feat": run_task2_binary(df_ptb, FEAT_54, tag="54feat", alpha=0.5),
        "task1_50feat_ablation": run_task1_5class(df_ptb, FEAT_50, tag="50feat", alpha=0.5),
        "task2_50feat_ablation": run_task2_binary(df_ptb, FEAT_50, tag="50feat", alpha=0.5),
    }

    out_file = RESULTS_DIR / "ptbxl_distillation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res_all, f, indent=2, ensure_ascii=False)
    print(f"\nSaved complete PTB-XL distillation benchmark results to {out_file}!", flush=True)


if __name__ == "__main__":
    main()
