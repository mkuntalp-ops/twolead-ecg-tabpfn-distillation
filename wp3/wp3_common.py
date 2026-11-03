"""WP3 shared definitions: cohorts, partition protocol, target construction (no torch / xgboost imports).

Design: ../ANALIZ_PLANI_v22.md, "WP3 tasarımı". The TabPFN teacher + MLP student run (wp3_teacher_mlp.py) and the
XGBoost student run (wp3_xgb.py) are separate processes because torch and xgboost load two OpenMP runtimes that
crash together on this machine (macOS); the XGBoost run reads the teacher targets and splits saved by the MLP run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "revizyon_v22/data_v2"
OUT_ROOT = ROOT / "revizyon_v22/results/wp3"
SUFFIXES = ["pr_int", "qrs_dur", "qt_int", "qtc_frid", "tpte_int", "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
            "j_amp", "st60_amp", "st_slope", "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_ent",
            "pow_05_5hz", "pow_5_15hz", "pow_15_40hz", "spec_ent", "zcr", "rms", "kurtosis"]
FEAT_50 = [f"{l}_{s}" for l in ("L1", "L2") for s in SUFFIXES]
K = 5
CLASSES = ["NORM", "AFIB/AFL", "CD", "STTC/MI", "HYP"]
COHORTS = ["ptbxl_quota5000", "ptbxl_natural5000", "ptbxl_natural15160", "chapman_quota5000", "chapman_natural"]
NATURAL5000_SAMPLING_SEED = 2026

EPSILONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHAS = [0.25, 0.50, 0.75, 1.00]
TEMPS = [1.0, 2.0, 4.0]
ARMS = (["H"] + [f"S_e{e:.2f}" for e in EPSILONS] + [f"D_a{a:.2f}_T{t:g}" for a in ALPHAS for t in TEMPS]
        + [f"EM_a{a:.2f}_T{t:g}" for a in ALPHAS for t in TEMPS])


def load_cohort(name: str) -> tuple[pd.DataFrame, list[int]]:
    """Frame with FEAT_50, `y`, `group` (patient), `uid` (record) and the cohort's partition seeds."""
    if name == "ptbxl_quota5000":
        d = pd.read_csv(V2 / "ptbxl_54features_v2.csv").rename(columns={"target_s1": "y"})
        d["uid"] = d["ecg_id"].astype(str); seeds = [42, 43, 44, 45, 46]
    elif name in ("ptbxl_natural15160", "ptbxl_natural5000"):
        d = pd.read_csv(V2 / "ptbxl_eligible_cohort_15160_features_v2.csv").rename(columns={"target_class": "y"})
        d["uid"] = d["ecg_id"].astype(str); seeds = [42]
        if name == "ptbxl_natural5000":  # class-stratified random draw; uses labels only
            keep, _ = train_test_split(np.arange(len(d)), train_size=5000, stratify=d["y"],
                                       random_state=NATURAL5000_SAMPLING_SEED)
            d = d.iloc[np.sort(keep)].reset_index(drop=True); seeds = [42, 43, 44, 45, 46]
    elif name in ("chapman_quota5000", "chapman_natural"):
        d = pd.read_csv(V2 / "chapman_eligible_features_v2.csv").rename(columns={"target_class": "y"})
        d["uid"] = d["record_id"].astype(str); seeds = [42]
        if name == "chapman_quota5000":
            d = d[d["in_quota_5000"] == 1].reset_index(drop=True); seeds = [42, 43, 44, 45, 46]
    else:
        raise ValueError(f"unknown cohort {name!r}; expected one of {COHORTS}")
    d["group"] = d["patient_id"].astype(str)
    assert d["uid"].is_unique, "record identifiers are not unique"
    assert set(d["y"].unique()) == set(range(K)), f"labels must cover 0..{K - 1}"
    assert np.isfinite(d[FEAT_50].to_numpy()).all(), "non-finite features"
    return d, seeds


def partitions(y: np.ndarray, grp: np.ndarray, seed: int):
    """Phase 3B protocol. Yields (fold, student_train_idx, student_val_idx, outer_test_idx, inner_splits)."""
    X0 = np.zeros((len(y), 1))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(outer.split(X0, y, grp), 1):
        a, b = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X0[tr], y[tr], grp[tr]))
        s_tr, s_val = tr[a], tr[b]
        assert not set(grp[tr]) & set(grp[te]), "outer patient leakage"
        assert not set(grp[s_tr]) & set(grp[s_val]), "student train/val leakage"
        inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                     .split(X0[s_tr], y[s_tr], grp[s_tr]))
        for itr, iva in inner:
            assert not set(grp[s_tr][itr]) & set(grp[s_tr][iva]), "inner teacher leakage"
        yield fold, s_tr, s_val, te, inner


def entropy(q: np.ndarray) -> np.ndarray:
    return -(q * np.log(np.clip(q, 1e-12, None))).sum(1)


def temper(p: np.ndarray, t: float) -> np.ndarray:
    """q_T proportional to p^(1/T)."""
    z = np.log(np.clip(p, 1e-12, None)) / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def entropy_matched_uniform(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Per-sample target: c_i on the true class, (1-c_i)/(K-1) on each other class, c_i in [1/K, 1], entropy = h_i."""
    def h_of(c):
        r = (1.0 - c) / (K - 1)
        return -(c * np.log(np.clip(c, 1e-12, None)) + (K - 1) * r * np.log(np.clip(r, 1e-12, None)))
    h = np.clip(h, 0.0, np.log(K))
    lo, hi = np.full(len(y), 1.0 / K), np.ones(len(y))
    for _ in range(60):  # h_of decreases on [1/K, 1]
        mid = 0.5 * (lo + hi)
        too_high = h_of(mid) > h
        lo = np.where(too_high, mid, lo); hi = np.where(too_high, hi, mid)
    c = 0.5 * (lo + hi)
    q = np.repeat(((1.0 - c) / (K - 1))[:, None], K, 1)
    q[np.arange(len(y)), y] = c
    return q


def build_targets(y: np.ndarray, teacher: np.ndarray) -> dict[str, np.ndarray]:
    onehot = np.eye(K)[y]
    tg = {"H": onehot}
    for e in EPSILONS:
        tg[f"S_e{e:.2f}"] = (1 - e) * onehot + e / K
    for a in ALPHAS:
        for t in TEMPS:
            d = (1 - a) * onehot + a * temper(teacher, t)
            tg[f"D_a{a:.2f}_T{t:g}"] = d
            tg[f"EM_a{a:.2f}_T{t:g}"] = entropy_matched_uniform(y, entropy(d))
    for k, v in tg.items():
        assert np.allclose(v.sum(1), 1.0, atol=1e-6) and (v >= 0).all(), f"invalid target {k}"
    return tg
