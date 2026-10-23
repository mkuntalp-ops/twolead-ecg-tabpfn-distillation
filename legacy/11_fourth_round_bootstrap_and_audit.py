#!/usr/bin/env python3
"""
11_fourth_round_bootstrap_and_audit.py

Executes the fourth-round final verification and statistical unification:
  1. Generates and archives all 30 out-of-fold prediction pairs for Table 5 into
     `predictions/table5_all30_oof_predictions.npz` and computes Monte Carlo estimates
     of two-sided bootstrap tail probabilities across B = 10,000 resamples (seed = 42)
     using the unified +1 formula:
       p = min(1.0, 2 * (min(n_{Delta <= 0}, n_{Delta >= 0}) + 1) / (B + 1))
     storing exact tail counts (n_neg, n_zero, n_pos, n_le_zero, n_ge_zero, n_opp)
     and FWER significance status (alpha_FWER = 0.05 / 30 = 0.001667).
  2. Evaluates Chapman-Shaoxing Within-Cohort 5-Fold Stratified Group K-Fold under
     `Strict-54` (Verified Adult Primary) so Table 7 reports `FEAT_50`, `Strict-54`,
     and `Ref-54` explicitly.
  3. Updates `results/fourth_round_verification.json` and `results/reproducibility_manifest.json`.
"""

import hashlib
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

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
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


def unified_paired_bootstrap(y_true, pred_a, pred_b, n_boot=10000, seed=42):
    """
    Unified paired bootstrap test (B = 10,000, seed = 42) using:
      p = min(1.0, 2 * (min(n_{Delta <= 0}, n_{Delta >= 0}) + 1) / (B + 1))
    Records exact tail counts (n_neg, n_zero, n_pos, n_le_zero, n_ge_zero, n_opp).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    emp_a = f1_score(y_true, pred_a, average="macro", zero_division=0) * 100.0
    emp_b = f1_score(y_true, pred_b, average="macro", zero_division=0) * 100.0
    emp_delta = emp_a - emp_b

    deltas = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        deltas[b] = (
            f1_score(yt, pred_a[idx], average="macro", zero_division=0) * 100.0
            - f1_score(yt, pred_b[idx], average="macro", zero_division=0) * 100.0
        )

    ci_low, ci_high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    n_neg = int(np.sum(deltas < 0.0))
    n_zero = int(np.sum(deltas == 0.0))
    n_pos = int(np.sum(deltas > 0.0))
    n_le_zero = int(np.sum(deltas <= 0.0))
    n_ge_zero = int(np.sum(deltas >= 0.0))
    n_opp = min(n_le_zero, n_ge_zero)
    p_plus1 = min(1.0, 2.0 * (n_opp + 1.0) / (n_boot + 1.0))
    alpha_fwer = 0.05 / 30.0
    fwer_sig = bool(p_plus1 < alpha_fwer)

    return {
        "f1_model_a_unrounded": round(float(emp_a), 4),
        "f1_model_b_unrounded": round(float(emp_b), 4),
        "f1_model_a": round(float(emp_a), 2),
        "f1_model_b": round(float(emp_b), 2),
        "delta_pp_unrounded": round(float(emp_delta), 4),
        "delta_pp": round(float(emp_delta), 2),
        "delta_ci_95_pp": [round(ci_low, 2), round(ci_high, 2)],
        "n_boot": n_boot,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "n_pos": n_pos,
        "n_le_zero": n_le_zero,
        "n_ge_zero": n_ge_zero,
        "n_opp": n_opp,
        "p_value_plus1": round(float(p_plus1), 4),
        "p_value_plus1_exact": round(float(p_plus1), 6),
        "fwer_significant_30tests": fwer_sig,
    }


def make_strict_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df_s = df_in.copy()
    cv1 = df_s["L1_rr_sdnn"] / np.maximum(df_s["L1_rr_mean"], 200.0)
    cv2 = df_s["L2_rr_sdnn"] / np.maximum(df_s["L2_rr_mean"], 200.0)
    rmssd_norm = np.minimum(df_s["L1_rr_rmssd"], df_s["L2_rr_rmssd"]) / np.maximum(
        0.5 * (df_s["L1_rr_mean"] + df_s["L2_rr_mean"]), 200.0
    )
    rr_disp = np.minimum(cv1, cv2)
    df_s["irregular_rr_flag"] = ((rr_disp > 0.11) | (rmssd_norm > 0.14)).astype(int)
    df_s["wide_qrs_flag"] = (np.maximum(df_s["L1_qrs_dur"], df_s["L2_qrs_dur"]) >= 118.0).astype(int)
    return df_s


def main():
    t0 = time.time()
    df_ptb = pd.read_csv(DATA_DIR / "ptbxl_54features.csv")
    df_chap = pd.read_csv(DATA_DIR / "chapman_54features.csv")
    df_cpsc = pd.read_csv(DATA_DIR / "cpsc2018_54features.csv")

    oof_t1_50 = pd.read_csv(PRED_DIR / "oof_task1_50feat.csv")
    oof_t2_50 = pd.read_csv(PRED_DIR / "oof_task2_50feat.csv")
    oof_t1_54 = pd.read_csv(PRED_DIR / "oof_task1_54feat.csv")
    oof_t2_54 = pd.read_csv(PRED_DIR / "oof_task2_54feat.csv")

    y_ptb_t1 = oof_t1_50["true_label"].values.astype(int)
    y_ptb_t2 = oof_t2_50["true_label"].values.astype(int)
    groups_ptb = df_ptb["patient_id"].values
    num_classes = 5

    print("1. Computing PTB-XL Control Models (Cap-Matched & Label-Smooth) on FEAT_50 and Ref-54...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    X_ptb_50 = df_ptb[FEAT_50].values.astype(np.float32)
    X_ptb_ref54 = df_ptb[FEAT_54].values.astype(np.float32)

    df_ptb_strict = make_strict_df(df_ptb)
    X_ptb_strict54 = df_ptb_strict[FEAT_54].values.astype(np.float32)

    pred_cap_xgb_50 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_smooth_xgb_50 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_cap_mlp_50 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_smooth_mlp_50 = np.zeros(len(y_ptb_t1), dtype=int)

    pred_cap_xgb_ref54 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_smooth_xgb_ref54 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_cap_mlp_ref54 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_smooth_mlp_ref54 = np.zeros(len(y_ptb_t1), dtype=int)

    pred_base_xgb_strict54 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_dist_xgb_strict54 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_base_mlp_strict54 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_cap_mlp_strict54 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_smooth_mlp_strict54 = np.zeros(len(y_ptb_t1), dtype=int)
    pred_dist_mlp_strict54 = np.zeros(len(y_ptb_t1), dtype=int)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_ptb_50, y_ptb_t1, groups=groups_ptb), 1):
        y_tr = y_ptb_t1[tr_idx]
        g_tr = groups_ptb[tr_idx]
        Y_oh = np.eye(num_classes)[y_tr]
        Y_smooth = 0.85 * Y_oh + 0.15 * (1.0 / num_classes)
        Y_smooth_log = np.log(np.clip(Y_smooth, 1e-4, 1.0))

        # FEAT_50 controls
        sc50 = StandardScaler()
        X_tr_50 = sc50.fit_transform(X_ptb_50[tr_idx])
        X_val_50 = sc50.transform(X_ptb_50[val_idx])

        xgb_c50 = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="mlogloss", random_state=42, n_jobs=1
        )
        xgb_c50.fit(X_tr_50, y_tr)
        pred_cap_xgb_50[val_idx] = xgb_c50.predict(X_val_50)

        xgb_s50 = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:logistic", random_state=42, n_jobs=1
            )
        )
        xgb_s50.fit(X_tr_50, Y_smooth)
        pred_smooth_xgb_50[val_idx] = np.argmax(xgb_s50.predict(X_val_50), axis=1)

        mlp_c50 = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        mlp_c50.fit(X_tr_50, y_tr)
        pred_cap_mlp_50[val_idx] = mlp_c50.predict(X_val_50)

        mlp_s50 = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        mlp_s50.fit(X_tr_50, Y_smooth_log)
        pred_smooth_mlp_50[val_idx] = np.argmax(mlp_s50.predict(X_val_50), axis=1)

        # Ref-54 controls
        sc54 = StandardScaler()
        X_tr_54 = sc54.fit_transform(X_ptb_ref54[tr_idx])
        X_val_54 = sc54.transform(X_ptb_ref54[val_idx])

        xgb_c54 = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="mlogloss", random_state=42, n_jobs=1
        )
        xgb_c54.fit(X_tr_54, y_tr)
        pred_cap_xgb_ref54[val_idx] = xgb_c54.predict(X_val_54)

        xgb_s54 = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:logistic", random_state=42, n_jobs=1
            )
        )
        xgb_s54.fit(X_tr_54, Y_smooth)
        pred_smooth_xgb_ref54[val_idx] = np.argmax(xgb_s54.predict(X_val_54), axis=1)

        mlp_c54 = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        mlp_c54.fit(X_tr_54, y_tr)
        pred_cap_mlp_ref54[val_idx] = mlp_c54.predict(X_val_54)

        mlp_s54 = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        mlp_s54.fit(X_tr_54, Y_smooth_log)
        pred_smooth_mlp_ref54[val_idx] = np.argmax(mlp_s54.predict(X_val_54), axis=1)

        # Strict-54 models
        sc_str = StandardScaler()
        X_tr_str = sc_str.fit_transform(X_ptb_strict54[tr_idx])
        X_val_str = sc_str.transform(X_ptb_strict54[val_idx])

        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        teacher_oof_tr = np.zeros((len(y_tr), num_classes), dtype=np.float64)
        for _, (i_tr, i_val) in enumerate(inner_sgkf.split(X_tr_str, y_tr, g_tr)):
            t_in = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            t_in.fit(X_tr_str[i_tr], y_tr[i_tr])
            teacher_oof_tr[i_val] = t_in.predict_proba(X_tr_str[i_val])

        Y_dist_str = 0.5 * Y_oh + 0.5 * teacher_oof_tr
        Y_dist_str_log = np.log(np.clip(Y_dist_str, 1e-4, 1.0))

        xgb_b_str = XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="mlogloss", random_state=42, n_jobs=1
        )
        xgb_b_str.fit(X_tr_str, y_tr)
        pred_base_xgb_strict54[val_idx] = xgb_b_str.predict(X_val_str)

        xgb_d_str = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:logistic", random_state=42, n_jobs=1
            )
        )
        xgb_d_str.fit(X_tr_str, Y_dist_str)
        pred_dist_xgb_strict54[val_idx] = np.argmax(xgb_d_str.predict(X_val_str), axis=1)

        mlp_b_str = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.01, learning_rate_init=0.001, max_iter=250,
            early_stopping=True, random_state=42
        )
        mlp_b_str.fit(X_tr_str, y_tr)
        pred_base_mlp_strict54[val_idx] = mlp_b_str.predict(X_val_str)

        mlp_c_str = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        mlp_c_str.fit(X_tr_str, y_tr)
        pred_cap_mlp_strict54[val_idx] = mlp_c_str.predict(X_val_str)

        mlp_s_str = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        mlp_s_str.fit(X_tr_str, Y_smooth_log)
        pred_smooth_mlp_strict54[val_idx] = np.argmax(mlp_s_str.predict(X_val_str), axis=1)

        mlp_d_str = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        mlp_d_str.fit(X_tr_str, Y_dist_str_log)
        pred_dist_mlp_strict54[val_idx] = np.argmax(mlp_d_str.predict(X_val_str), axis=1)

    print("2. Computing Chapman-Shaoxing (FEAT_50 & Strict-54) and CPSC-2018 (FEAT_50) OOF Predictions...", flush=True)
    # Chapman FEAT_50 Task 1
    X_chap_50 = df_chap[FEAT_50].values.astype(np.float32)
    y_chap_t1 = df_chap["target_s1"].values.astype(int)
    y_chap_t2 = df_chap["target_s2"].values.astype(int)
    g_chap = df_chap["patient_id"].values

    pred_chap50_base_xgb = np.zeros(len(y_chap_t1), dtype=int)
    pred_chap50_dist_xgb = np.zeros(len(y_chap_t1), dtype=int)
    pred_chap50_base_mlp = np.zeros(len(y_chap_t1), dtype=int)
    pred_chap50_dist_mlp = np.zeros(len(y_chap_t1), dtype=int)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_chap_50, y_chap_t1, groups=g_chap), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_chap_50[tr_idx])
        X_val = sc.transform(X_chap_50[val_idx])
        y_tr = y_chap_t1[tr_idx]
        g_tr = g_chap[tr_idx]

        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        t_oof = np.zeros((len(y_tr), 5), dtype=np.float64)
        for _, (i_tr, i_val) in enumerate(inner_sgkf.split(X_tr, y_tr, g_tr)):
            t_in = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            t_in.fit(X_tr[i_tr], y_tr[i_tr])
            t_oof[i_val] = t_in.predict_proba(X_tr[i_val])

        Y_oh = np.eye(5)[y_tr]
        Y_dist = 0.5 * Y_oh + 0.5 * t_oof
        Y_dist_log = np.log(np.clip(Y_dist, 1e-4, 1.0))

        xb = XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="mlogloss", random_state=42, n_jobs=1
        )
        xb.fit(X_tr, y_tr)
        pred_chap50_base_xgb[val_idx] = xb.predict(X_val)

        xd = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:logistic", random_state=42, n_jobs=1
            )
        )
        xd.fit(X_tr, Y_dist)
        pred_chap50_dist_xgb[val_idx] = np.argmax(xd.predict(X_val), axis=1)

        mb = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.01, learning_rate_init=0.001, max_iter=250,
            early_stopping=True, random_state=42
        )
        mb.fit(X_tr, y_tr)
        pred_chap50_base_mlp[val_idx] = mb.predict(X_val)

        md = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        md.fit(X_tr, Y_dist_log)
        pred_chap50_dist_mlp[val_idx] = np.argmax(md.predict(X_val), axis=1)

    # Chapman Strict-54 Teacher TabPFN replication (Task 1 & Task 2) for Table 7 completeness
    df_chap_strict = make_strict_df(df_chap)
    X_chap_strict54 = df_chap_strict[FEAT_54].values.astype(np.float32)
    probs_chap_strict_t1 = np.zeros((len(y_chap_t1), 5), dtype=np.float64)
    probs_chap_strict_t2 = np.zeros((len(y_chap_t2), 2), dtype=np.float64)
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_chap_strict54, y_chap_t1, groups=g_chap), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_chap_strict54[tr_idx])
        X_val = sc.transform(X_chap_strict54[val_idx])
        t_m = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=8)
        t_m.fit(X_tr, y_chap_t1[tr_idx])
        probs_chap_strict_t1[val_idx] = t_m.predict_proba(X_val)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_chap_strict54, y_chap_t2, groups=g_chap), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_chap_strict54[tr_idx])
        X_val = sc.transform(X_chap_strict54[val_idx])
        t_b = TabPFNClassifier(model_path=TABPFN_BIN_CKPT, device="mps", n_estimators=8)
        t_b.fit(X_tr, y_chap_t2[tr_idx])
        probs_chap_strict_t2[val_idx] = t_b.predict_proba(X_val)

    chap_strict54_summary = {
        "task1_5class_tabpfn": {
            "macro_f1": round(float(f1_score(y_chap_t1, np.argmax(probs_chap_strict_t1, axis=1), average="macro") * 100.0), 2),
            "accuracy": round(float(accuracy_score(y_chap_t1, np.argmax(probs_chap_strict_t1, axis=1)) * 100.0), 2),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_chap_t1, np.argmax(probs_chap_strict_t1, axis=1)) * 100.0), 2),
            "roc_auc": round(float(roc_auc_score(y_chap_t1, probs_chap_strict_t1, multi_class="ovr", average="macro") * 100.0), 2),
        },
        "task2_binary_tabpfn": {
            "macro_f1": round(float(f1_score(y_chap_t2, np.argmax(probs_chap_strict_t2, axis=1), average="macro") * 100.0), 2),
            "accuracy": round(float(accuracy_score(y_chap_t2, np.argmax(probs_chap_strict_t2, axis=1)) * 100.0), 2),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_chap_t2, np.argmax(probs_chap_strict_t2, axis=1)) * 100.0), 2),
            "roc_auc": round(float(roc_auc_score(y_chap_t2, probs_chap_strict_t2[:, 1]) * 100.0), 2),
        },
    }
    print(f"   Chapman Strict-54 Summary: {chap_strict54_summary}", flush=True)

    # CPSC-2018 FEAT_50 Task 1 (4-Class)
    X_cpsc_50 = df_cpsc[FEAT_50].values.astype(np.float32)
    y_cpsc_t1 = df_cpsc["target_s1"].values.astype(int)
    g_cpsc = df_cpsc["patient_id"].values

    pred_cpsc50_base_xgb = np.zeros(len(y_cpsc_t1), dtype=int)
    pred_cpsc50_dist_xgb = np.zeros(len(y_cpsc_t1), dtype=int)
    pred_cpsc50_base_mlp = np.zeros(len(y_cpsc_t1), dtype=int)
    pred_cpsc50_dist_mlp = np.zeros(len(y_cpsc_t1), dtype=int)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_cpsc_50, y_cpsc_t1, groups=g_cpsc), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_cpsc_50[tr_idx])
        X_val = sc.transform(X_cpsc_50[val_idx])
        y_tr = y_cpsc_t1[tr_idx]
        g_tr = g_cpsc[tr_idx]

        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        t_oof = np.zeros((len(y_tr), 4), dtype=np.float64)
        for _, (i_tr, i_val) in enumerate(inner_sgkf.split(X_tr, y_tr, g_tr)):
            t_in = TabPFNClassifier(model_path=TABPFN_MULTI_CKPT, device="mps", n_estimators=4)
            t_in.fit(X_tr[i_tr], y_tr[i_tr])
            t_oof[i_val] = t_in.predict_proba(X_tr[i_val])

        Y_oh = np.eye(4)[y_tr]
        Y_dist = 0.5 * Y_oh + 0.5 * t_oof
        Y_dist_log = np.log(np.clip(Y_dist, 1e-4, 1.0))

        xb = XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="mlogloss", random_state=42, n_jobs=1
        )
        xb.fit(X_tr, y_tr)
        pred_cpsc50_base_xgb[val_idx] = xb.predict(X_val)

        xd = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:logistic", random_state=42, n_jobs=1
            )
        )
        xd.fit(X_tr, Y_dist)
        pred_cpsc50_dist_xgb[val_idx] = np.argmax(xd.predict(X_val), axis=1)

        mb = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.01, learning_rate_init=0.001, max_iter=250,
            early_stopping=True, random_state=42
        )
        mb.fit(X_tr, y_tr)
        pred_cpsc50_base_mlp[val_idx] = mb.predict(X_val)

        md = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=350,
            early_stopping=True, random_state=42
        )
        md.fit(X_tr, Y_dist_log)
        pred_cpsc50_dist_mlp[val_idx] = np.argmax(md.predict(X_val), axis=1)

    # Save all prediction arrays to disk for complete auditability
    np.savez_compressed(
        PRED_DIR / "table5_all30_oof_predictions.npz",
        y_ptb_t1=y_ptb_t1,
        y_ptb_t2=y_ptb_t2,
        y_chap_t1=y_chap_t1,
        y_cpsc_t1=y_cpsc_t1,
        ptb50_tabpfn_t1=oof_t1_50["pred_teacher_tabpfn"].values,
        ptb50_dist_xgb_t1=oof_t1_50["pred_distilled_xgboost"].values,
        ptb50_base_xgb_t1=oof_t1_50["pred_baseline_xgboost"].values,
        ptb50_dist_mlp_t1=oof_t1_50["pred_distilled_mlp"].values,
        ptb50_base_mlp_t1=oof_t1_50["pred_baseline_mlp"].values,
        ptb50_cap_xgb_t1=pred_cap_xgb_50,
        ptb50_smooth_xgb_t1=pred_smooth_xgb_50,
        ptb50_cap_mlp_t1=pred_cap_mlp_50,
        ptb50_smooth_mlp_t1=pred_smooth_mlp_50,
        ptb50_tabpfn_t2=oof_t2_50["pred_teacher_tabpfn"].values,
        ptb50_dist_xgb_t2=oof_t2_50["pred_distilled_xgboost"].values,
        ptb50_base_xgb_t2=oof_t2_50["pred_baseline_xgboost"].values,
        ptb50_dist_mlp_t2=oof_t2_50["pred_distilled_mlp"].values,
        ptb50_base_mlp_t2=oof_t2_50["pred_baseline_mlp"].values,
        ptbref54_tabpfn_t1=oof_t1_54["pred_teacher_tabpfn"].values,
        ptbref54_dist_xgb_t1=oof_t1_54["pred_distilled_xgboost"].values,
        ptbref54_base_xgb_t1=oof_t1_54["pred_baseline_xgboost"].values,
        ptbref54_dist_mlp_t1=oof_t1_54["pred_distilled_mlp"].values,
        ptbref54_base_mlp_t1=oof_t1_54["pred_baseline_mlp"].values,
        ptbref54_cap_xgb_t1=pred_cap_xgb_ref54,
        ptbref54_smooth_xgb_t1=pred_smooth_xgb_ref54,
        ptbref54_cap_mlp_t1=pred_cap_mlp_ref54,
        ptbref54_smooth_mlp_t1=pred_smooth_mlp_ref54,
        ptbref54_tabpfn_t2=oof_t2_54["pred_teacher_tabpfn"].values,
        ptbref54_dist_xgb_t2=oof_t2_54["pred_distilled_xgboost"].values,
        ptbref54_base_xgb_t2=oof_t2_54["pred_baseline_xgboost"].values,
        ptbref54_dist_mlp_t2=oof_t2_54["pred_distilled_mlp"].values,
        ptbref54_base_mlp_t2=oof_t2_54["pred_baseline_mlp"].values,
        ptbstrict54_base_xgb_t1=pred_base_xgb_strict54,
        ptbstrict54_dist_xgb_t1=pred_dist_xgb_strict54,
        ptbstrict54_base_mlp_t1=pred_base_mlp_strict54,
        ptbstrict54_cap_mlp_t1=pred_cap_mlp_strict54,
        ptbstrict54_smooth_mlp_t1=pred_smooth_mlp_strict54,
        ptbstrict54_dist_mlp_t1=pred_dist_mlp_strict54,
        chap50_base_xgb_t1=pred_chap50_base_xgb,
        chap50_dist_xgb_t1=pred_chap50_dist_xgb,
        chap50_base_mlp_t1=pred_chap50_base_mlp,
        chap50_dist_mlp_t1=pred_chap50_dist_mlp,
        cpsc50_base_xgb_t1=pred_cpsc50_base_xgb,
        cpsc50_dist_xgb_t1=pred_cpsc50_dist_xgb,
        cpsc50_base_mlp_t1=pred_cpsc50_base_mlp,
        cpsc50_dist_mlp_t1=pred_cpsc50_dist_mlp,
    )

    print("3. Running Unified B=10,000 (+1 Adjusted) Bootstrap Across All 30 Comparisons in Table 5...", flush=True)
    comparisons_spec = [
        # Section A (6 tests)
        ("A1_TabPFN_vs_DistXGB_PTB_T1_FEAT50", y_ptb_t1, oof_t1_50["pred_teacher_tabpfn"].values, oof_t1_50["pred_distilled_xgboost"].values),
        ("A2_TabPFN_vs_DistMLP_PTB_T1_FEAT50", y_ptb_t1, oof_t1_50["pred_teacher_tabpfn"].values, oof_t1_50["pred_distilled_mlp"].values),
        ("A3_TabPFN_vs_DistXGB_PTB_T1_Ref54", y_ptb_t1, oof_t1_54["pred_teacher_tabpfn"].values, oof_t1_54["pred_distilled_xgboost"].values),
        ("A4_TabPFN_vs_DistMLP_PTB_T1_Ref54", y_ptb_t1, oof_t1_54["pred_teacher_tabpfn"].values, oof_t1_54["pred_distilled_mlp"].values),
        ("A5_TabPFN_vs_DistXGB_PTB_T2_FEAT50", y_ptb_t2, oof_t2_50["pred_teacher_tabpfn"].values, oof_t2_50["pred_distilled_xgboost"].values),
        ("A6_TabPFN_vs_BaseMLP_PTB_T2_Ref54", y_ptb_t2, oof_t2_54["pred_teacher_tabpfn"].values, oof_t2_54["pred_baseline_mlp"].values),
        # Section B (14 tests)
        ("B1_DistXGB_vs_BaseXGB_PTB_T1_FEAT50", y_ptb_t1, oof_t1_50["pred_distilled_xgboost"].values, oof_t1_50["pred_baseline_xgboost"].values),
        ("B2_DistMLP_vs_BaseMLP_PTB_T1_FEAT50", y_ptb_t1, oof_t1_50["pred_distilled_mlp"].values, oof_t1_50["pred_baseline_mlp"].values),
        ("B3_DistXGB_vs_BaseXGB_PTB_T1_Strict54", y_ptb_t1, pred_dist_xgb_strict54, pred_base_xgb_strict54),
        ("B4_DistMLP_vs_BaseMLP_PTB_T1_Strict54", y_ptb_t1, pred_dist_mlp_strict54, pred_base_mlp_strict54),
        ("B5_DistXGB_vs_BaseXGB_PTB_T1_Ref54", y_ptb_t1, oof_t1_54["pred_distilled_xgboost"].values, oof_t1_54["pred_baseline_xgboost"].values),
        ("B6_DistMLP_vs_BaseMLP_PTB_T1_Ref54", y_ptb_t1, oof_t1_54["pred_distilled_mlp"].values, oof_t1_54["pred_baseline_mlp"].values),
        ("B7_DistXGB_vs_BaseXGB_Chap_T1_FEAT50", y_chap_t1, pred_chap50_dist_xgb, pred_chap50_base_xgb),
        ("B8_DistMLP_vs_BaseMLP_Chap_T1_FEAT50", y_chap_t1, pred_chap50_dist_mlp, pred_chap50_base_mlp),
        ("B9_DistMLP_vs_BaseMLP_CPSC_T1_FEAT50", y_cpsc_t1, pred_cpsc50_dist_mlp, pred_cpsc50_base_mlp),
        ("B10_DistXGB_vs_BaseXGB_CPSC_T1_FEAT50", y_cpsc_t1, pred_cpsc50_dist_xgb, pred_cpsc50_base_xgb),
        ("B11_DistXGB_vs_BaseXGB_PTB_T2_FEAT50", y_ptb_t2, oof_t2_50["pred_distilled_xgboost"].values, oof_t2_50["pred_baseline_xgboost"].values),
        ("B12_DistMLP_vs_BaseMLP_PTB_T2_FEAT50", y_ptb_t2, oof_t2_50["pred_distilled_mlp"].values, oof_t2_50["pred_baseline_mlp"].values),
        ("B13_DistXGB_vs_BaseXGB_PTB_T2_Ref54", y_ptb_t2, oof_t2_54["pred_distilled_xgboost"].values, oof_t2_54["pred_baseline_xgboost"].values),
        ("B14_DistMLP_vs_BaseMLP_PTB_T2_Ref54", y_ptb_t2, oof_t2_54["pred_distilled_mlp"].values, oof_t2_54["pred_baseline_mlp"].values),
        # Section C (10 tests)
        ("C1_DistMLP_vs_CapMLP_PTB_T1_FEAT50", y_ptb_t1, oof_t1_50["pred_distilled_mlp"].values, pred_cap_mlp_50),
        ("C2_DistMLP_vs_SmoothMLP_PTB_T1_FEAT50", y_ptb_t1, oof_t1_50["pred_distilled_mlp"].values, pred_smooth_mlp_50),
        ("C3_DistXGB_vs_CapXGB_PTB_T1_FEAT50", y_ptb_t1, oof_t1_50["pred_distilled_xgboost"].values, pred_cap_xgb_50),
        ("C4_DistXGB_vs_SmoothXGB_PTB_T1_FEAT50", y_ptb_t1, oof_t1_50["pred_distilled_xgboost"].values, pred_smooth_xgb_50),
        ("C5_DistMLP_vs_CapMLP_PTB_T1_Strict54", y_ptb_t1, pred_dist_mlp_strict54, pred_cap_mlp_strict54),
        ("C6_DistMLP_vs_SmoothMLP_PTB_T1_Strict54", y_ptb_t1, pred_dist_mlp_strict54, pred_smooth_mlp_strict54),
        ("C7_DistMLP_vs_CapMLP_PTB_T1_Ref54", y_ptb_t1, oof_t1_54["pred_distilled_mlp"].values, pred_cap_mlp_ref54),
        ("C8_DistMLP_vs_SmoothMLP_PTB_T1_Ref54", y_ptb_t1, oof_t1_54["pred_distilled_mlp"].values, pred_smooth_mlp_ref54),
        ("C9_DistXGB_vs_CapXGB_PTB_T1_Ref54", y_ptb_t1, oof_t1_54["pred_distilled_xgboost"].values, pred_cap_xgb_ref54),
        ("C10_DistXGB_vs_SmoothXGB_PTB_T1_Ref54", y_ptb_t1, oof_t1_54["pred_distilled_xgboost"].values, pred_smooth_xgb_ref54),
    ]

    table5_results = {}
    for name, yt, pa, pb in comparisons_spec:
        res = unified_paired_bootstrap(yt, pa, pb, n_boot=10000, seed=42)
        table5_results[name] = res
        print(
            f"   {name:42s} | A={res['f1_model_a']:5.2f} B={res['f1_model_b']:5.2f} "
            f"| Δ={res['delta_pp']:+6.2f} {res['delta_ci_95_pp']} "
            f"| n_opp={res['n_opp']:4d} (neg={res['n_neg']}, zero={res['n_zero']}, pos={res['n_pos']}) "
            f"| p={res['p_value_plus1']:.4f} | FWER={res['fwer_significant_30tests']}",
            flush=True,
        )

    out_data = {
        "hardware_specs": {
            "soc": "Apple M4 (arm64)",
            "ram_gb": 16,
            "os": "macOS 27.0.1",
        },
        "chapman_strict54_within_cohort": chap_strict54_summary,
        "table5_unified_10000_bootstrap_plus1": table5_results,
    }
    out_path = RESULTS_DIR / "fourth_round_verification.json"
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved fourth_round_verification.json in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
