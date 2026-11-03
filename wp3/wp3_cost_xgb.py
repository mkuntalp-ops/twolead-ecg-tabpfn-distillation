#!/usr/bin/env python3
"""WP3 deployment cost, XGBoost students (separate process from torch; see wp3_common).

1. Tree counts for every cohort/seed/fold/arm from the saved validation tables: trees = (best_iter + 1) * K
   (the early-stopped model; xgboost's predict uses iterations 0..best_iter).
2. Exact size for a reference fold (first seed, fold 1) of every cohort: the H and D_fixed (a0.50_T1) students are
   refitted with the WP3 settings, truncated to best_iter, and their number of split and leaf nodes, stored parameters
   (split: feature + threshold; leaf: value) and saved JSON size are recorded; CPU latency is measured with one thread
   for a single record (median of 1,000 calls) and for a batch of 1,000 records (median of 20 calls).
   The refitted test Macro F1 is compared with the stored WP3 predictions as a reproducibility check.

Usage (project root):  python revizyon_v22/wp3/wp3_cost_xgb.py
Writes revizyon_v22/results/wp3/cost/xgb_tree_counts.csv, xgb_reference_models.csv.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp3_common import COHORTS, FEAT_50, K, OUT_ROOT, build_targets, load_cohort  # noqa: E402
from wp3_xgb import XGB_PARAMS  # noqa: E402

OUT = OUT_ROOT / "cost"


def latency(fn, x, reps):
    fn(x)  # warm-up
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(x); t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t))


def tree_counts() -> pd.DataFrame:
    rows = []
    for c in COHORTS:
        for f in sorted((OUT_ROOT / c).glob("seed*_xgb_val.csv")):
            v = pd.read_csv(f)
            rows.append(v.assign(n_trees=(v["best_iter"] + 1) * K)[["cohort", "seed", "fold", "arm", "best_iter", "n_trees"]])
    return pd.concat(rows, ignore_index=True)


def reference_models() -> pd.DataFrame:
    rows = []
    for c in COHORTS:
        d, seeds = load_cohort(c)
        seed, fold = seeds[0], 1
        X = d[FEAT_50].to_numpy(np.float64); y = d["y"].to_numpy(np.int64)
        z = np.load(OUT_ROOT / c / f"seed{seed}_teacher.npz", allow_pickle=True)
        zx = np.load(OUT_ROOT / c / f"seed{seed}_xgb.npz", allow_pickle=True)
        s_tr, s_val, te = z[f"f{fold}_s_tr"], z[f"f{fold}_s_val"], z[f"f{fold}_te"]
        sc = StandardScaler().fit(X[s_tr])
        Xtr, Xval, Xte = sc.transform(X[s_tr]), sc.transform(X[s_val]), sc.transform(X[te])
        targets = build_targets(y[s_tr], z[f"f{fold}_teacher_inner_oof"].astype(np.float64))
        for arm, name in [("H", "H"), ("D_a0.50_T1", "D_fixed")]:
            n = len(Xtr)
            Xr = np.repeat(Xtr, K, 0); yr = np.tile(np.arange(K), n); wr = targets[arm].reshape(-1); keep = wr > 0
            clf = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed * 1000 + fold)
            clf.fit(Xr[keep], yr[keep], sample_weight=wr[keep], eval_set=[(Xval, y[s_val])], verbose=False)
            best = int(clf.best_iteration)
            pt = clf.predict_proba(Xte)
            f1_refit = 100 * f1_score(y[te], pt.argmax(1), average="macro")
            f1_stored = 100 * f1_score(y[te], zx[f"xgb_{arm}"][te].argmax(1), average="macro")
            booster = clf.get_booster()[: best + 1]
            df = booster.trees_to_dataframe()
            n_leaf = int((df["Feature"] == "Leaf").sum()); n_split = len(df) - n_leaf
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "m.ubj"); booster.save_model(p); size_ubj = os.path.getsize(p)
                p = os.path.join(td, "m.json"); booster.save_model(p); size_json = os.path.getsize(p)
            booster.set_param({"nthread": 1})
            one, batch = np.ascontiguousarray(Xte[:1]), np.ascontiguousarray(Xte[:1000])  # no DMatrix: its predictions are cached
            rows.append({"cohort": c, "seed": seed, "fold": fold, "model": f"xgb_{name}", "best_iter": best,
                         "n_trees": (best + 1) * K, "n_split_nodes": n_split, "n_leaf_nodes": n_leaf,
                         "stored_parameters": 2 * n_split + n_leaf, "size_ubj_bytes": size_ubj,
                         "size_json_bytes": size_json,
                         "latency_1_record_ms": latency(booster.inplace_predict, one, 1000),
                         "latency_1000_records_ms": latency(booster.inplace_predict, batch, 20),
                         "test_F1_refit": f1_refit, "test_F1_stored": f1_stored})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tc = tree_counts(); tc.to_csv(OUT / "xgb_tree_counts.csv", index=False)
    print(tc.groupby(["cohort", "arm"]).n_trees.describe().loc[(slice(None), ["H", "D_a0.50_T1"]), :].round(0))
    reference_models().to_csv(OUT / "xgb_reference_models.csv", index=False)


if __name__ == "__main__":
    main()
