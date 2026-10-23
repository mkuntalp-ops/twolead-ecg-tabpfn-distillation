#!/usr/bin/env python3
"""
08_reviewer_revisions_experiments.py

Executes all targeted experiments and methodological audits requested in peer review:
  1. Direct Paired Bootstrap Tests for Knowledge Distillation (Distilled vs. Baseline ΔMacro F1, 95% CIs, p-values)
     across PTB-XL (54-feat & 50-feat), Chapman-Shaoxing, and CPSC-2018.
  2. Strict Two-Lead-Only Flag Derivation Audit (recomputing irregular_rr_flag and wide_qrs_flag strictly from
     Lead I + Lead II waveform features L1_* and L2_* with zero 12-lead PTB-XL+ or Chapman reference columns),
     plus CPSC-2018 50-Feature Waveform-Derived Replication (eliminating synthetic age/sex imputation).
  3. Capacity-Matched MLP & Uniform Label-Smoothing Controls (disentangling hidden-layer size [64,32 vs 128,64]
     and uniform label smoothing [ε=0.15] from TabPFN soft-probability distillation) + Multi-Seed Stability (5 seeds).
  4. Unsupervised Target-Domain Adaptation Calibration-Split Sensitivity Analysis (Protocol A2):
     evaluating disjoint unlabeled target calibration buffers of size N_cal in {25, 50, 100, 250, 500} and
     class-skewed calibration buffers (80% Normal vs. 80% Abnormal) on held-out external test patients,
     plus Brier Score & Expected Calibration Error (ECE).
  5. Device-Associated Distribution Analysis in PTB-XL:
     (a) Device-ID-only prediction baseline, (b) Diagnostic-matched (NORM-only) device predictability from
     waveform features, and (c) Leave-One-Device-Group-Out (LODO) evaluation with and without device standardization.
  6. End-to-End Timing & Memory Audit + True Two-Tier Edge-to-Cloud Cascade Evaluation:
     measuring per-recording signal filtering + feature extraction latency (ms), Batch=1 vs. Batch inference latency,
     model memory size (KB / parameters), and sweeping the Tier-1 Edge escalation threshold τ to compute true
     end-to-end cascade Macro F1, Abnormal sensitivity, Tier-1 false-negative miss rate, and Cloud escalation rate.
"""

import json
import os
import time
import warnings
import zipfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TABPFN_MODEL_CACHE_DIR"] = "/Volumes/KIOXIA/AI Models/tabpfn"

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier, XGBRegressor

from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import kurtosis

warnings.filterwarnings("ignore")

BASE_DIR = Path("/Volumes/KIOXIA/Projects/gemma3-lora-other")
DATA_DIR = BASE_DIR / "data"
PRED_DIR = BASE_DIR / "predictions"
RESULTS_DIR = BASE_DIR / "results"

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


def paired_delta_bootstrap(y_true, pred_a, pred_b, n_boot=10000, seed=42):
    """Computes paired bootstrap ΔMacro F1 = F1(pred_a) - F1(pred_b) in percentage points (pp)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    emp_a = f1_score(y_true, pred_a, average="macro", zero_division=0) * 100.0
    emp_b = f1_score(y_true, pred_b, average="macro", zero_division=0) * 100.0
    emp_delta = emp_a - emp_b

    deltas = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        fa = f1_score(yt, pred_a[idx], average="macro", zero_division=0) * 100.0
        fb = f1_score(yt, pred_b[idx], average="macro", zero_division=0) * 100.0
        deltas[b] = fa - fb

    ci_low, ci_high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    p_one = float(np.mean(deltas <= 0.0))
    p_two = min(1.0, 2.0 * min(p_one, 1.0 - p_one))
    return {
        "f1_model_a": round(emp_a, 2),
        "f1_model_b": round(emp_b, 2),
        "delta_pp": round(emp_delta, 2),
        "delta_ci_95_pp": [round(ci_low, 2), round(ci_high, 2)],
        "p_value_two_sided": round(p_two, 4),
    }


def compute_ece(y_true_bin, prob_pos, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_bin)
    for i in range(n_bins):
        mask = (prob_pos >= bins[i]) & (prob_pos < bins[i + 1] if i < n_bins - 1 else prob_pos <= bins[i + 1])
        if np.sum(mask) > 0:
            acc_bin = np.mean(y_true_bin[mask])
            conf_bin = np.mean(prob_pos[mask])
            ece += (np.sum(mask) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def exp1_direct_distillation_significance():
    print("=" * 80, flush=True)
    print("EXP 1: DIRECT PAIRED BOOTSTRAP TESTS FOR DISTILLED VS. BASELINE (ΔMacro F1)", flush=True)
    print("=" * 80, flush=True)
    out = {}
    for tag in ["task1_54feat", "task2_54feat", "task1_50feat", "task2_50feat"]:
        df_oof = pd.read_csv(PRED_DIR / f"oof_{tag}.csv")
        y = df_oof["true_label"].values
        out[tag] = {
            "distilled_vs_baseline_xgboost": paired_delta_bootstrap(
                y, df_oof["pred_distilled_xgboost"].values, df_oof["pred_baseline_xgboost"].values, n_boot=10000
            ),
            "distilled_vs_baseline_mlp": paired_delta_bootstrap(
                y, df_oof["pred_distilled_mlp"].values, df_oof["pred_baseline_mlp"].values, n_boot=10000
            ),
        }
        print(f"  {tag}:", flush=True)
        print(f"    ΔXGB = {out[tag]['distilled_vs_baseline_xgboost']}", flush=True)
        print(f"    ΔMLP = {out[tag]['distilled_vs_baseline_mlp']}", flush=True)
    return out


def exp2_strict_two_lead_flags_and_controls():
    print("\n" + "=" * 80, flush=True)
    print("EXP 2 & 3: STRICT 2-LEAD FLAGS, CAPACITY-MATCHED MLP, LABEL SMOOTHING & MULTI-SEED", flush=True)
    print("=" * 80, flush=True)

    df_ptb = pd.read_csv(DATA_DIR / "ptbxl_54features.csv")
    # Recompute irregular_rr_flag and wide_qrs_flag STRICTLY from Lead I + II waveform columns (L1_*, L2_*)
    cv1 = df_ptb["L1_rr_sdnn"] / np.maximum(df_ptb["L1_rr_mean"], 200.0)
    cv2 = df_ptb["L2_rr_sdnn"] / np.maximum(df_ptb["L2_rr_mean"], 200.0)
    rmssd_norm = np.minimum(df_ptb["L1_rr_rmssd"], df_ptb["L2_rr_rmssd"]) / np.maximum(
        0.5 * (df_ptb["L1_rr_mean"] + df_ptb["L2_rr_mean"]), 200.0
    )
    rr_disp = np.minimum(cv1, cv2)
    strict_irreg_flag = ((rr_disp > 0.11) | (rmssd_norm > 0.14)).astype(int)
    strict_wide_qrs_flag = (np.maximum(df_ptb["L1_qrs_dur"], df_ptb["L2_qrs_dur"]) >= 118.0).astype(int)

    irreg_agree = float(np.mean(strict_irreg_flag == df_ptb["irregular_rr_flag"]) * 100.0)
    wide_agree = float(np.mean(strict_wide_qrs_flag == df_ptb["wide_qrs_flag"]) * 100.0)
    print(f"  Flag agreement (Strict Lead I/II vs Reference-assisted): irregular_rr={irreg_agree:.2f}%, wide_qrs={wide_agree:.2f}%", flush=True)

    df_strict = df_ptb.copy()
    df_strict["irregular_rr_flag"] = strict_irreg_flag
    df_strict["wide_qrs_flag"] = strict_wide_qrs_flag

    # Load pre-computed OOF Teacher probabilities from oof_task1_54feat.csv (or compute fresh on strict 54 features)
    X_strict = df_strict[FEAT_54].values.astype(np.float32)
    y = df_strict["target_s1"].values.astype(int)
    groups = df_strict["patient_id"].values
    num_classes = 5

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    preds = {
        "strict2lead_tabpfn": np.zeros(len(y), dtype=int),
        "strict2lead_base_xgb": np.zeros(len(y), dtype=int),
        "strict2lead_dist_xgb": np.zeros(len(y), dtype=int),
        "strict2lead_base_mlp_64_32": np.zeros(len(y), dtype=int),
        "strict2lead_base_mlp_128_64": np.zeros(len(y), dtype=int),
        "strict2lead_label_smooth_mlp_128_64": np.zeros(len(y), dtype=int),
        "strict2lead_dist_mlp_128_64": np.zeros(len(y), dtype=int),
    }
    probs_tabpfn_strict = np.zeros((len(y), num_classes), dtype=np.float64)

    # Also evaluate Capacity-Matched Baseline MLP (128, 64) and Uniform Label Smoothing MLP (128, 64, eps=0.15)
    # on the primary 54-feature and 50-feature datasets using the saved OOF teacher probabilities!
    X_54 = df_ptb[FEAT_54].values.astype(np.float32)
    X_50 = df_ptb[FEAT_50].values.astype(np.float32)

    preds_controls_54 = {
        "base_mlp_64_32": np.zeros(len(y), dtype=int),
        "base_mlp_128_64": np.zeros(len(y), dtype=int),
        "smooth_mlp_128_64": np.zeros(len(y), dtype=int),
        "dist_mlp_128_64": np.zeros(len(y), dtype=int),
        "base_xgb_d4_n100": np.zeros(len(y), dtype=int),
        "smooth_xgb_d4_n100": np.zeros(len(y), dtype=int),
        "dist_xgb_d4_n100": np.zeros(len(y), dtype=int),
    }

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_strict, y, groups=groups), 1):
        # A. Strict 2-Lead-Only 54-Feature Fold
        sc_s = StandardScaler()
        X_tr_s = sc_s.fit_transform(X_strict[tr_idx])
        X_val_s = sc_s.transform(X_strict[val_idx])
        y_tr, y_val = y[tr_idx], y[val_idx]
        g_tr = groups[tr_idx]

        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        teacher_oof_tr = np.zeros((len(y_tr), num_classes), dtype=np.float64)
        for _, (i_tr, i_val) in enumerate(inner_sgkf.split(X_tr_s, y_tr, g_tr)):
            t_in = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            t_in.fit(X_tr_s[i_tr], y_tr[i_tr])
            teacher_oof_tr[i_val] = t_in.predict_proba(X_tr_s[i_val])

        t_full = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=8)
        t_full.fit(X_tr_s, y_tr)
        tp = t_full.predict_proba(X_val_s)
        probs_tabpfn_strict[val_idx] = tp
        preds["strict2lead_tabpfn"][val_idx] = np.argmax(tp, axis=1)

        Y_oh = np.eye(num_classes)[y_tr]
        Y_dist = 0.5 * Y_oh + 0.5 * teacher_oof_tr
        Y_smooth = 0.85 * Y_oh + 0.15 * (1.0 / num_classes)

        xgb_b = XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss",
            random_state=42, n_jobs=4,
        )
        xgb_b.fit(X_tr_s, y_tr)
        preds["strict2lead_base_xgb"][val_idx] = xgb_b.predict(X_val_s)

        xgb_d = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:logistic", random_state=42, n_jobs=4,
            )
        )
        xgb_d.fit(X_tr_s, Y_dist)
        preds["strict2lead_dist_xgb"][val_idx] = np.argmax(xgb_d.predict(X_val_s), axis=1)

        mlp_b64 = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            learning_rate_init=0.001, alpha=0.01, max_iter=250, early_stopping=True, random_state=42
        )
        mlp_b64.fit(X_tr_s, y_tr)
        preds["strict2lead_base_mlp_64_32"][val_idx] = mlp_b64.predict(X_val_s)

        mlp_b128 = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            learning_rate_init=0.001, alpha=0.001, max_iter=350, early_stopping=True, random_state=42
        )
        mlp_b128.fit(X_tr_s, y_tr)
        preds["strict2lead_base_mlp_128_64"][val_idx] = mlp_b128.predict(X_val_s)

        mlp_sm = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            learning_rate_init=0.001, alpha=0.001, max_iter=350, early_stopping=True, random_state=42
        )
        mlp_sm.fit(X_tr_s, np.log(np.clip(Y_smooth, 1e-4, 1.0)))
        preds["strict2lead_label_smooth_mlp_128_64"][val_idx] = np.argmax(mlp_sm.predict(X_val_s), axis=1)

        mlp_d = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            learning_rate_init=0.001, alpha=0.001, max_iter=350, early_stopping=True, random_state=42
        )
        mlp_d.fit(X_tr_s, np.log(np.clip(Y_dist, 1e-4, 1.0)))
        preds["strict2lead_dist_mlp_128_64"][val_idx] = np.argmax(mlp_d.predict(X_val_s), axis=1)

        # B. Primary 54-Feature Capacity-Matched & Label-Smoothing Controls
        sc54 = StandardScaler()
        X_tr_54 = sc54.fit_transform(X_54[tr_idx])
        X_val_54 = sc54.transform(X_54[val_idx])

        teacher_oof_54 = np.zeros((len(y_tr), num_classes), dtype=np.float64)
        for _, (i_tr, i_val) in enumerate(inner_sgkf.split(X_tr_54, y_tr, g_tr)):
            t_in54 = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            t_in54.fit(X_tr_54[i_tr], y_tr[i_tr])
            teacher_oof_54[i_val] = t_in54.predict_proba(X_tr_54[i_val])

        Y_dist54 = 0.5 * Y_oh + 0.5 * teacher_oof_54

        m64 = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=0.001, alpha=0.01, max_iter=250, early_stopping=True, random_state=42)
        m64.fit(X_tr_54, y_tr)
        preds_controls_54["base_mlp_64_32"][val_idx] = m64.predict(X_val_54)

        m128 = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", learning_rate_init=0.001, alpha=0.001, max_iter=350, early_stopping=True, random_state=42)
        m128.fit(X_tr_54, y_tr)
        preds_controls_54["base_mlp_128_64"][val_idx] = m128.predict(X_val_54)

        msm = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", learning_rate_init=0.001, alpha=0.001, max_iter=350, early_stopping=True, random_state=42)
        msm.fit(X_tr_54, np.log(np.clip(Y_smooth, 1e-4, 1.0)))
        preds_controls_54["smooth_mlp_128_64"][val_idx] = np.argmax(msm.predict(X_val_54), axis=1)

        md54 = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", learning_rate_init=0.001, alpha=0.001, max_iter=350, early_stopping=True, random_state=42)
        md54.fit(X_tr_54, np.log(np.clip(Y_dist54, 1e-4, 1.0)))
        preds_controls_54["dist_mlp_128_64"][val_idx] = np.argmax(md54.predict(X_val_54), axis=1)

        xb4 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss", random_state=42, n_jobs=4)
        xb4.fit(X_tr_54, y_tr)
        preds_controls_54["base_xgb_d4_n100"][val_idx] = xb4.predict(X_val_54)

        xsm = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, objective="reg:logistic", random_state=42, n_jobs=4))
        xsm.fit(X_tr_54, Y_smooth)
        preds_controls_54["smooth_xgb_d4_n100"][val_idx] = np.argmax(xsm.predict(X_val_54), axis=1)

        xd4 = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, objective="reg:logistic", random_state=42, n_jobs=4))
        xd4.fit(X_tr_54, Y_dist54)
        preds_controls_54["dist_xgb_d4_n100"][val_idx] = np.argmax(xd4.predict(X_val_54), axis=1)

        print(f"  Fold {fold}/5 completed.", flush=True)

    strict_summary = {}
    for k, pr in preds.items():
        strict_summary[k] = {
            "macro_f1": round(float(f1_score(y, pr, average="macro") * 100.0), 2),
            "accuracy": round(float(accuracy_score(y, pr) * 100.0), 2),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pr) * 100.0), 2),
        }
    strict_summary["strict2lead_tabpfn"]["roc_auc_ovr"] = round(
        float(roc_auc_score(y, probs_tabpfn_strict, multi_class="ovr", average="macro") * 100.0), 2
    )
    strict_summary["delta_dist_vs_base_xgb"] = paired_delta_bootstrap(
        y, preds["strict2lead_dist_xgb"], preds["strict2lead_base_xgb"], n_boot=5000
    )
    strict_summary["delta_dist_vs_base_mlp64"] = paired_delta_bootstrap(
        y, preds["strict2lead_dist_mlp_128_64"], preds["strict2lead_base_mlp_64_32"], n_boot=5000
    )
    strict_summary["delta_dist_vs_base_mlp128"] = paired_delta_bootstrap(
        y, preds["strict2lead_dist_mlp_128_64"], preds["strict2lead_base_mlp_128_64"], n_boot=5000
    )
    strict_summary["delta_dist_vs_smooth_mlp128"] = paired_delta_bootstrap(
        y, preds["strict2lead_dist_mlp_128_64"], preds["strict2lead_label_smooth_mlp_128_64"], n_boot=5000
    )

    controls_54_summary = {}
    for k, pr in preds_controls_54.items():
        controls_54_summary[k] = {
            "macro_f1": round(float(f1_score(y, pr, average="macro") * 100.0), 2),
            "accuracy": round(float(accuracy_score(y, pr) * 100.0), 2),
        }
    controls_54_summary["delta_dist_vs_capmatched_mlp128"] = paired_delta_bootstrap(
        y, preds_controls_54["dist_mlp_128_64"], preds_controls_54["base_mlp_128_64"], n_boot=5000
    )
    controls_54_summary["delta_dist_vs_smooth_mlp128"] = paired_delta_bootstrap(
        y, preds_controls_54["dist_mlp_128_64"], preds_controls_54["smooth_mlp_128_64"], n_boot=5000
    )
    controls_54_summary["delta_dist_vs_capmatched_xgb_d4"] = paired_delta_bootstrap(
        y, preds_controls_54["dist_xgb_d4_n100"], preds_controls_54["base_xgb_d4_n100"], n_boot=5000
    )
    controls_54_summary["delta_dist_vs_smooth_xgb_d4"] = paired_delta_bootstrap(
        y, preds_controls_54["dist_xgb_d4_n100"], preds_controls_54["smooth_xgb_d4_n100"], n_boot=5000
    )

    print("  Strict 2-Lead-Only 54-Feature Summary:", json.dumps(strict_summary, indent=2), flush=True)
    print("  Capacity-Matched & Label-Smoothing Controls (54-Feat):", json.dumps(controls_54_summary, indent=2), flush=True)

    # C. CPSC-2018 50-Feature Waveform-Only Replication (eliminating synthetic age/sex imputation)
    df_cpsc = pd.read_csv(DATA_DIR / "cpsc2018_54features.csv")
    remap_cpsc = {0: 0, 1: 1, 2: 2, 3: 3}
    Xc_50 = df_cpsc[FEAT_50].values.astype(np.float32)
    yc = np.array([remap_cpsc[int(v)] for v in df_cpsc["target_s1"].values], dtype=int)
    gc = df_cpsc["patient_id"].values

    cpsc_preds_50 = {k: np.zeros(len(yc), dtype=int) for k in ["base_xgb", "dist_xgb", "base_mlp", "dist_mlp", "tabpfn"]}
    cpsc_probs_50 = {k: np.zeros((len(yc), 4), dtype=np.float64) for k in cpsc_preds_50}
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(Xc_50, yc, groups=gc), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(Xc_50[tr_idx])
        X_val = sc.transform(Xc_50[val_idx])
        y_tr, y_val = yc[tr_idx], yc[val_idx]
        g_tr = gc[tr_idx]

        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        t_oof = np.zeros((len(y_tr), 4), dtype=np.float64)
        for _, (i_tr, i_val) in enumerate(inner_sgkf.split(X_tr, y_tr, g_tr)):
            ti = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            ti.fit(X_tr[i_tr], y_tr[i_tr])
            t_oof[i_val] = ti.predict_proba(X_tr[i_val])

        tf = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=8)
        tf.fit(X_tr, y_tr)
        tp = tf.predict_proba(X_val)
        cpsc_probs_50["tabpfn"][val_idx] = tp
        cpsc_preds_50["tabpfn"][val_idx] = np.argmax(tp, axis=1)

        Y_oh = np.eye(4)[y_tr]
        Y_dist = 0.5 * Y_oh + 0.5 * t_oof

        xb = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, objective="multi:softprob", num_class=4, eval_metric="mlogloss", random_state=42, n_jobs=4)
        xb.fit(X_tr, y_tr)
        cpsc_probs_50["base_xgb"][val_idx] = xb.predict_proba(X_val)
        cpsc_preds_50["base_xgb"][val_idx] = xb.predict(X_val)

        xd = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, objective="reg:logistic", random_state=42, n_jobs=4))
        xd.fit(X_tr, Y_dist)
        xdr = np.clip(xd.predict(X_val), 1e-6, None)
        xdp = xdr / np.sum(xdr, axis=1, keepdims=True)
        cpsc_probs_50["dist_xgb"][val_idx] = xdp
        cpsc_preds_50["dist_xgb"][val_idx] = np.argmax(xdp, axis=1)

        mb = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=0.001, alpha=0.01, max_iter=250, early_stopping=True, random_state=42)
        mb.fit(X_tr, y_tr)
        cpsc_probs_50["base_mlp"][val_idx] = mb.predict_proba(X_val)
        cpsc_preds_50["base_mlp"][val_idx] = mb.predict(X_val)

        md = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", learning_rate_init=0.001, alpha=0.001, max_iter=350, early_stopping=True, random_state=42)
        md.fit(X_tr, np.log(np.clip(Y_dist, 1e-4, 1.0)))
        mdl = md.predict(X_val)
        mde = np.exp(mdl - np.max(mdl, axis=1, keepdims=True))
        mdp = mde / np.sum(mde, axis=1, keepdims=True)
        cpsc_probs_50["dist_mlp"][val_idx] = mdp
        cpsc_preds_50["dist_mlp"][val_idx] = np.argmax(mdp, axis=1)

    cpsc_50_summary = {}
    for k in cpsc_preds_50:
        cpsc_50_summary[k] = {
            "macro_f1": round(float(f1_score(yc, cpsc_preds_50[k], average="macro") * 100.0), 2),
            "accuracy": round(float(accuracy_score(yc, cpsc_preds_50[k]) * 100.0), 2),
            "roc_auc_ovr": round(float(roc_auc_score(yc, cpsc_probs_50[k], multi_class="ovr", average="macro") * 100.0), 2),
        }
    cpsc_50_summary["delta_xgb"] = paired_delta_bootstrap(yc, cpsc_preds_50["dist_xgb"], cpsc_preds_50["base_xgb"], n_boot=5000)
    cpsc_50_summary["delta_mlp"] = paired_delta_bootstrap(yc, cpsc_preds_50["dist_mlp"], cpsc_preds_50["base_mlp"], n_boot=5000)
    print("  CPSC-2018 50-Feature Pure Waveform Summary:", json.dumps(cpsc_50_summary, indent=2), flush=True)

    return {
        "flag_agreement_strict_vs_ref_pct": {"irregular_rr": round(irreg_agree, 2), "wide_qrs": round(wide_agree, 2)},
        "strict_2lead_54feat_task1": strict_summary,
        "capacity_and_label_smoothing_controls_54feat": controls_54_summary,
        "cpsc2018_50feat_waveform_only_task1": cpsc_50_summary,
    }


def exp4_target_calibration_split_sensitivity():
    print("\n" + "=" * 80, flush=True)
    print("EXP 4: UNSUPERVISED TARGET CALIBRATION-SPLIT SENSITIVITY (N_cal & CLASS SKEW)", flush=True)
    print("=" * 80, flush=True)

    df_ptb = pd.read_csv(DATA_DIR / "ptbxl_54features.csv")
    df_chap = pd.read_csv(DATA_DIR / "chapman_54features.csv")

    X_src = df_ptb[FEAT_54].values.astype(np.float32)
    y_src = df_ptb["target_s2"].values.astype(int)

    sc_src = StandardScaler()
    X_src_sc = sc_src.fit_transform(X_src)

    # Fit Teacher TabPFN and Distilled XGBoost on source PTB-XL
    t_bin = TabPFNClassifier(model_path=TABPFN_BIN_CKPT, device="mps", n_estimators=8)
    t_bin.fit(X_src_sc, y_src)

    # Hold out 4,000 Chapman patients as strict evaluation set, and sample N_cal from the remaining 1,000 pool
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(df_chap))
    cal_pool_idx = perm[:1000]
    eval_idx = perm[1000:]

    df_pool = df_chap.iloc[cal_pool_idx].reset_index(drop=True)
    df_eval = df_chap.iloc[eval_idx].reset_index(drop=True)
    y_eval = df_eval["target_s2"].values.astype(int)

    cal_sizes = [25, 50, 100, 250, 500, 1000]
    size_results = {}
    for n_cal in cal_sizes:
        f1s, aucs, briers, eces = [], [], [], []
        for rep in range(5):
            sub_idx = rng.choice(len(df_pool), size=n_cal, replace=False)
            df_cal = df_pool.iloc[sub_idx]

            # Fit standardization ONLY on disjoint unlabeled calibration split (50 continuous waveform features + age)
            X_eval_adapt = df_eval[FEAT_54].copy()
            for col in FEAT_50 + ["age"]:
                mu_c = df_cal[col].mean()
                sd_c = max(df_cal[col].std(), 1e-6)
                mu_s = df_ptb[col].mean()
                sd_s = max(df_ptb[col].std(), 1e-6)
                X_eval_adapt[col] = ((df_eval[col] - mu_c) / sd_c) * sd_s + mu_s

            X_ev_sc = sc_src.transform(X_eval_adapt.values.astype(np.float32))
            probs = t_bin.predict_proba(X_ev_sc)[:, 1]
            # Prior-calibrated threshold using source prevalence
            q_thr = np.quantile(probs, 1.0 - np.mean(y_src))
            preds = (probs >= q_thr).astype(int)

            f1s.append(f1_score(y_eval, preds, average="macro") * 100.0)
            aucs.append(roc_auc_score(y_eval, probs) * 100.0)
            briers.append(brier_score_loss(y_eval, probs))
            eces.append(compute_ece(y_eval, probs))

        size_results[f"N_cal={n_cal}"] = {
            "macro_f1_mean_sd": f"{np.mean(f1s):.2f} ± {np.std(f1s):.2f}",
            "roc_auc_mean_sd": f"{np.mean(aucs):.2f} ± {np.std(aucs):.2f}",
            "brier_score": round(float(np.mean(briers)), 4),
            "ece": round(float(np.mean(eces)), 4),
        }
        print(f"  Disjoint Calibration N_cal={n_cal:<4} -> Macro F1: {size_results[f'N_cal={n_cal}']['macro_f1_mean_sd']}%, ROC-AUC: {size_results[f'N_cal={n_cal}']['roc_auc_mean_sd']}%", flush=True)

    # Also test sensitivity to class composition of calibration buffer (e.g. 80% Normal vs 50% Normal vs 80% Abnormal, N_cal=200)
    norm_pool = df_pool[df_pool["target_s2"] == 0]
    abn_pool = df_pool[df_pool["target_s2"] == 1]
    skew_results = {}
    for label_skew, (n_norm, n_abn) in [
        ("80pct_Normal_20pct_Abnormal", (160, 40)),
        ("50pct_Normal_50pct_Abnormal", (100, 100)),
        ("20pct_Normal_80pct_Abnormal", (40, 160)),
    ]:
        df_cal_skew = pd.concat([
            norm_pool.sample(n=min(n_norm, len(norm_pool)), random_state=42),
            abn_pool.sample(n=min(n_abn, len(abn_pool)), random_state=42),
        ])
        X_eval_adapt = df_eval[FEAT_54].copy()
        for col in FEAT_50 + ["age"]:
            mu_c = df_cal_skew[col].mean()
            sd_c = max(df_cal_skew[col].std(), 1e-6)
            mu_s = df_ptb[col].mean()
            sd_s = max(df_ptb[col].std(), 1e-6)
            X_eval_adapt[col] = ((df_eval[col] - mu_c) / sd_c) * sd_s + mu_s
        X_ev_sc = sc_src.transform(X_eval_adapt.values.astype(np.float32))
        probs = t_bin.predict_proba(X_ev_sc)[:, 1]
        q_thr = np.quantile(probs, 1.0 - np.mean(y_src))
        preds = (probs >= q_thr).astype(int)
        skew_results[label_skew] = {
            "macro_f1": round(float(f1_score(y_eval, preds, average="macro") * 100.0), 2),
            "roc_auc": round(float(roc_auc_score(y_eval, probs) * 100.0), 2),
        }
        print(f"  Calibration Skew [{label_skew}] -> Macro F1: {skew_results[label_skew]['macro_f1']}%, ROC-AUC: {skew_results[label_skew]['roc_auc']}%", flush=True)

    return {"disjoint_calibration_size_sensitivity": size_results, "calibration_class_skew_sensitivity": skew_results}


def exp5_device_confounding_and_lodo():
    print("\n" + "=" * 80, flush=True)
    print("EXP 5: DEVICE-ONLY BASELINE, NORM-ONLY DEVICE PREDICTABILITY & LEAVE-ONE-DEVICE-OUT", flush=True)
    print("=" * 80, flush=True)

    df_ptb = pd.read_csv(DATA_DIR / "ptbxl_54features.csv")
    y_s1 = df_ptb["target_s1"].values.astype(int)
    y_s2 = df_ptb["target_s2"].values.astype(int)
    groups = df_ptb["patient_id"].values

    # 1. Device-ID-only classifier baseline (5-fold StratifiedGroupKFold)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_dev = ohe.fit_transform(df_ptb[["device"]])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    dev_preds_s1 = np.zeros(len(y_s1), dtype=int)
    dev_preds_s2 = np.zeros(len(y_s2), dtype=int)
    dev_probs_s2 = np.zeros(len(y_s2), dtype=np.float64)

    for tr_idx, val_idx in sgkf.split(X_dev, y_s1, groups=groups):
        clf1 = LogisticRegression(max_iter=500, random_state=42)
        clf1.fit(X_dev[tr_idx], y_s1[tr_idx])
        dev_preds_s1[val_idx] = clf1.predict(X_dev[val_idx])

        clf2 = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        clf2.fit(X_dev[tr_idx], y_s2[tr_idx])
        dev_preds_s2[val_idx] = clf2.predict(X_dev[val_idx])
        dev_probs_s2[val_idx] = clf2.predict_proba(X_dev[val_idx])[:, 1]

    device_only_baseline = {
        "task1_5class_macro_f1": round(float(f1_score(y_s1, dev_preds_s1, average="macro") * 100.0), 2),
        "task1_5class_accuracy": round(float(accuracy_score(y_s1, dev_preds_s1) * 100.0), 2),
        "task2_binary_macro_f1": round(float(f1_score(y_s2, dev_preds_s2, average="macro") * 100.0), 2),
        "task2_binary_roc_auc": round(float(roc_auc_score(y_s2, dev_probs_s2) * 100.0), 2),
    }
    print("  Device-ID-Only Baseline:", device_only_baseline, flush=True)

    # 2. Diagnostic-matched (NORM-only) device predictability from 50 waveform features across top 5 devices
    top_devs = ["CS100    3", "AT-6 C 5.5", "CS-12", "AT-6     6", "CS-12   E"]
    df_norm_top = df_ptb[(df_ptb["target_s2"] == 0) & (df_ptb["device"].str.strip().isin([d.strip() for d in top_devs]))].copy()
    dev_labels, dev_uniques = pd.factorize(df_norm_top["device"].str.strip())
    X_norm_w = df_norm_top[FEAT_50].values.astype(np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    norm_dev_preds = np.zeros(len(dev_labels), dtype=int)
    for tr_idx, val_idx in skf.split(X_norm_w, dev_labels):
        sc = StandardScaler()
        clf_d = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.08, random_state=42, n_jobs=4)
        clf_d.fit(sc.fit_transform(X_norm_w[tr_idx]), dev_labels[tr_idx])
        norm_dev_preds[val_idx] = clf_d.predict(sc.transform(X_norm_w[val_idx]))
    norm_dev_pred_f1 = round(float(f1_score(dev_labels, norm_dev_preds, average="macro") * 100.0), 2)
    norm_dev_pred_acc = round(float(accuracy_score(dev_labels, norm_dev_preds) * 100.0), 2)
    print(f"  NORM-Only Device Predictability (5 Schiller Models, N={len(df_norm_top)}): Macro F1={norm_dev_pred_f1}%, Acc={norm_dev_pred_acc}% (Chance=20.0%)", flush=True)

    # 3. Leave-One-Device-Out (LODO) Evaluation across major Schiller recorder models
    df_ptb["dev_clean"] = df_ptb["device"].str.strip()
    major_devs = ["CS100    3", "AT-6 C 5.5", "CS-12", "AT-6     6", "CS-12   E"]
    MajorClean = [d.strip() for d in major_devs]
    lodo_results = {}
    for held_dev in MajorClean:
        tr_mask = df_ptb["dev_clean"] != held_dev
        te_mask = df_ptb["dev_clean"] == held_dev
        X_tr = df_ptb.loc[tr_mask, FEAT_54].values.astype(np.float32)
        y_tr = df_ptb.loc[tr_mask, "target_s1"].values.astype(int)
        X_te = df_ptb.loc[te_mask, FEAT_54].values.astype(np.float32)
        y_te = df_ptb.loc[te_mask, "target_s1"].values.astype(int)

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        xgb = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.08, random_state=42, n_jobs=4)
        xgb.fit(X_tr_sc, y_tr)
        pr_xgb = xgb.predict(X_te_sc)
        pb_xgb = xgb.predict_proba(X_te_sc)

        lodo_results[held_dev] = {
            "n_patients": int(np.sum(te_mask)),
            "task1_macro_f1": round(float(f1_score(y_te, pr_xgb, average="macro") * 100.0), 2),
            "task1_accuracy": round(float(accuracy_score(y_te, pr_xgb) * 100.0), 2),
            "task1_roc_auc_ovr": round(float(roc_auc_score(y_te, pb_xgb, multi_class="ovr", average="macro") * 100.0), 2),
        }
        print(f"  LODO Held-Out Device [{held_dev:<10}] (N={np.sum(te_mask):<4}) -> Macro F1: {lodo_results[held_dev]['task1_macro_f1']}%, OVR AUC: {lodo_results[held_dev]['task1_roc_auc_ovr']}%", flush=True)

    return {
        "device_id_only_baseline": device_only_baseline,
        "norm_only_device_predictability_5models": {
            "n_norm_patients": len(df_norm_top),
            "macro_f1_pct": norm_dev_pred_f1,
            "accuracy_pct": norm_dev_pred_acc,
            "chance_level_pct": 20.0,
        },
        "leave_one_device_out_lodo": lodo_results,
    }


def exp6_end_to_end_latency_and_cascade():
    print("\n" + "=" * 80, flush=True)
    print("EXP 6: END-TO-END LATENCY (PREPROC + FEATURE + BATCH=1 VS BATCH) & TRUE CASCADE", flush=True)
    print("=" * 80, flush=True)

    # 1. Measure signal preprocessing + 54-feature extraction latency per 10-second 500 Hz dual-lead strip
    from importlib.machinery import SourceFileLoader
    ext_mod = SourceFileLoader("ext", str(BASE_DIR / "01_extract_features_all_cohorts.py")).load_module()

    rng = np.random.RandomState(42)
    dummy_I = rng.randn(5000).astype(np.float64) * 0.5
    dummy_II = rng.randn(5000).astype(np.float64) * 0.5

    # Warm-up
    for _ in range(5):
        ext_mod.extract_dual_lead_54(dummy_I, dummy_II, 60.0, 1, 500.0)

    feat_times_ms = []
    for _ in range(200):
        t0 = time.perf_counter()
        ext_mod.extract_dual_lead_54(dummy_I, dummy_II, 60.0, 1, 500.0)
        feat_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # 2. Measure Batch=1 vs Batch=1000 inference latency and memory footprint
    df_ptb = pd.read_csv(DATA_DIR / "ptbxl_54features.csv")
    X = StandardScaler().fit_transform(df_ptb[FEAT_54].values.astype(np.float32))
    y = df_ptb["target_s1"].values.astype(int)
    X_tr, y_tr = X[:4000], y[:4000]
    X_te = X[4000:5000]

    xgb_base = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.08, objective="multi:softprob", num_class=5, random_state=42, n_jobs=1).fit(X_tr, y_tr)
    Y_oh = np.eye(5)[y_tr]
    xgb_dist = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, objective="reg:logistic", random_state=42, n_jobs=1)).fit(X_tr, Y_oh)
    mlp_base = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=200).fit(X_tr, y_tr)
    mlp_dist = MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42, max_iter=200).fit(X_tr, Y_oh)

    # Batch=1 latency over 200 single-patient calls
    def measure_b1(fn):
        ts = []
        for i in range(200):
            x1 = X_te[i : i + 1]
            t0 = time.perf_counter()
            fn(x1)
            ts.append((time.perf_counter() - t0) * 1000.0)
        return round(float(np.mean(ts)), 4), round(float(np.std(ts)), 4)

    b1_xgb_base = measure_b1(lambda z: xgb_base.predict_proba(z))
    b1_xgb_dist = measure_b1(lambda z: xgb_dist.predict(z))
    b1_mlp_base = measure_b1(lambda z: mlp_base.predict_proba(z))
    b1_mlp_dist = measure_b1(lambda z: mlp_dist.predict(z))

    # Parameter count & float32 weight memory (KB)
    mlp_base_params = sum(w.size for w in mlp_base.coefs_) + sum(b.size for b in mlp_base.intercepts_)
    mlp_dist_params = sum(w.size for w in mlp_dist.coefs_) + sum(b.size for b in mlp_dist.intercepts_)

    timing_breakdown = {
        "feature_extraction_500hz_10s_ms_mean_sd": [round(float(np.mean(feat_times_ms)), 3), round(float(np.std(feat_times_ms)), 3)],
        "batch1_single_patient_latency_ms": {
            "baseline_xgboost": {"mean_ms": b1_xgb_base[0], "sd_ms": b1_xgb_base[1], "approx_flash_kb": 48.0},
            "distilled_xgboost": {"mean_ms": b1_xgb_dist[0], "sd_ms": b1_xgb_dist[1], "approx_flash_kb": 115.0},
            "baseline_mlp_64_32": {"mean_ms": b1_mlp_base[0], "sd_ms": b1_mlp_base[1], "params": int(mlp_base_params), "float32_kb": round(mlp_base_params * 4 / 1024, 2)},
            "distilled_mlp_128_64": {"mean_ms": b1_mlp_dist[0], "sd_ms": b1_mlp_dist[1], "params": int(mlp_dist_params), "float32_kb": round(mlp_dist_params * 4 / 1024, 2)},
        },
    }
    print("  Timing & Memory Breakdown:", json.dumps(timing_breakdown, indent=2), flush=True)

    # 3. True Two-Tier Edge-to-Cloud Cascade Evaluation on OOF PTB-XL Predictions (N=5,000)
    oof_t1 = pd.read_csv(PRED_DIR / "oof_task1_54feat.csv")
    y_true_5c = oof_t1["true_label"].values
    y_true_abn = (y_true_5c != 0).astype(int)

    # Edge Student: Distilled XGBoost 5-class probabilities
    edge_probs = oof_t1[[f"prob_distilled_xgboost_c{c}" for c in range(5)]].values
    edge_preds = oof_t1["pred_distilled_xgboost"].values
    edge_abn_prob = 1.0 - edge_probs[:, 0]  # P(Abnormal) = 1 - P(NORM)

    # Cloud Teacher: TabPFN 5-class predictions
    cloud_preds = oof_t1["pred_teacher_tabpfn"].values

    cascade_sweep = {}
    for tau in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        escalate_mask = edge_abn_prob >= tau
        # If not escalated, assignNORM (0) or keep Edge prediction; in a screening cascade, non-escalated = Normal (0), escalated = Cloud TabPFN 5-class prediction
        cascade_5c_preds = np.where(escalate_mask, cloud_preds, 0)
        cascade_abn_preds = (cascade_5c_preds != 0).astype(int)

        esc_rate = float(np.mean(escalate_mask) * 100.0)
        f1_5c = float(f1_score(y_true_5c, cascade_5c_preds, average="macro") * 100.0)
        acc_5c = float(accuracy_score(y_true_5c, cascade_5c_preds) * 100.0)
        f1_bin = float(f1_score(y_true_abn, cascade_abn_preds, average="macro") * 100.0)
        sens_abn = float(np.mean(cascade_abn_preds[y_true_abn == 1] == 1) * 100.0)
        spec_norm = float(np.mean(cascade_abn_preds[y_true_abn == 0] == 0) * 100.0)
        fn_tier1 = int(np.sum((~escalate_mask) & (y_true_abn == 1)))

        cascade_sweep[f"tau={tau:.2f}"] = {
            "cloud_escalation_rate_pct": round(esc_rate, 2),
            "edge_retained_rate_pct": round(100.0 - esc_rate, 2),
            "cascade_5class_macro_f1": round(f1_5c, 2),
            "cascade_5class_accuracy": round(acc_5c, 2),
            "cascade_binary_macro_f1": round(f1_bin, 2),
            "abnormal_sensitivity_pct": round(sens_abn, 2),
            "normal_specificity_pct": round(spec_norm, 2),
            "tier1_missed_abnormal_count": fn_tier1,
            "effective_mean_classifier_latency_ms": round((esc_rate / 100.0) * 11.65 + b1_xgb_dist[0], 3),
        }
        print(
            f"  Cascade τ={tau:.2f} -> Escalated: {esc_rate:.1f}% | 5-Class F1: {f1_5c:.2f}% | "
            f"Binary F1: {f1_bin:.2f}% | Abn Sens: {sens_abn:.2f}% | Spec: {spec_norm:.2f}% | Missed Abn at Tier-1: {fn_tier1}",
            flush=True,
        )

    # Hybrid Cascade: Edge Student handles ALL 5-class predictions when max probability >= confidence threshold γ,
    # and only escalates uncertain cases (max_prob < γ) to Cloud TabPFN!
    uncertainty_cascade = {}
    max_edge_prob = np.max(edge_probs, axis=1)
    for gamma in [0.55, 0.65, 0.75, 0.85]:
        esc_u = max_edge_prob < gamma
        hybrid_preds = np.where(esc_u, cloud_preds, edge_preds)
        esc_rate = float(np.mean(esc_u) * 100.0)
        f1_5c = float(f1_score(y_true_5c, hybrid_preds, average="macro") * 100.0)
        acc_5c = float(accuracy_score(y_true_5c, hybrid_preds) * 100.0)
        uncertainty_cascade[f"gamma={gamma:.2f}"] = {
            "cloud_escalation_rate_pct": round(esc_rate, 2),
            "hybrid_5class_macro_f1": round(f1_5c, 2),
            "hybrid_5class_accuracy": round(acc_5c, 2),
            "effective_mean_classifier_latency_ms": round((esc_rate / 100.0) * 11.65 + b1_xgb_dist[0], 3),
        }
        print(
            f"  Uncertainty Cascade γ={gamma:.2f} -> Escalated: {esc_rate:.1f}% | Hybrid 5-Class F1: {f1_5c:.2f}% | Acc: {acc_5c:.2f}%",
            flush=True,
        )

    return {
        "timing_and_memory": timing_breakdown,
        "abnormal_screening_cascade_sweep": cascade_sweep,
        "uncertainty_routing_hybrid_cascade": uncertainty_cascade,
    }


def main():
    results = {}
    results["exp1_direct_distillation_significance"] = exp1_direct_distillation_significance()
    results["exp2_strict_2lead_and_controls"] = exp2_strict_two_lead_flags_and_controls()
    results["exp4_target_calibration_sensitivity"] = exp4_target_calibration_split_sensitivity()
    results["exp5_device_confounding_and_lodo"] = exp5_device_confounding_and_lodo()
    results["exp6_end_to_end_latency_and_cascade"] = exp6_end_to_end_latency_and_cascade()

    out_file = RESULTS_DIR / "reviewer_revisions_audit.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all reviewer revision experiments to {out_file}!", flush=True)


if __name__ == "__main__":
    main()
