#!/usr/bin/env python3
"""WP3, step 2: XGBoost students for every arm, using the splits and inner-OOF teacher targets saved by
wp3_teacher_mlp.py (so MLP and XGBoost see identical targets). Runs in its own process without torch (see wp3_common).

Soft targets are fitted by replicating each training sample K times (copy k: label k, weight Q[i, k]; zero weights
dropped). Fixed settings: max_depth 4, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, <= 2,000 rounds,
early stopping (50 rounds) on student-val mlogloss with true labels, tree_method "hist".

Usage (project root):  python revizyon_v22/wp3/wp3_xgb.py --cohort ptbxl_quota5000 [--seeds 42 43]
Writes revizyon_v22/results/wp3/<cohort>/seed<k>_xgb.npz, seed<k>_xgb_val.csv, run_manifest_xgb.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp3_common import ARMS, FEAT_50, K, OUT_ROOT, build_targets, entropy, load_cohort  # noqa: E402

XGB_PARAMS = dict(objective="multi:softprob", num_class=K, n_estimators=2000, learning_rate=0.05, max_depth=4,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
                  early_stopping_rounds=50, n_jobs=8)


def train_xgb(X_tr, Q, X_val, y_val, X_te, random_state):
    n = len(X_tr)
    Xr = np.repeat(X_tr, K, 0); yr = np.tile(np.arange(K), n); wr = Q.reshape(-1)
    keep = wr > 0
    clf = xgb.XGBClassifier(**XGB_PARAMS, random_state=random_state)
    clf.fit(Xr[keep], yr[keep], sample_weight=wr[keep], eval_set=[(X_val, y_val)], verbose=False)
    return clf.predict_proba(X_val), clf.predict_proba(X_te), int(clf.best_iteration)


def run_seed(cohort: str, d: pd.DataFrame, seed: int, out_dir: Path) -> None:
    X = d[FEAT_50].to_numpy(np.float64); y = d["y"].to_numpy(np.int64)
    z = np.load(out_dir / f"seed{seed}_teacher.npz", allow_pickle=True)
    assert np.array_equal(z["y"], y) and np.array_equal(z["uid"].astype(str), d["uid"].to_numpy()), "cohort mismatch"
    N = len(d)
    probs = {f"xgb_{a}": np.full((N, K), np.nan, np.float32) for a in ARMS}
    val_rows = []
    for fold in range(1, 6):
        t0 = time.time()
        s_tr, s_val, te = z[f"f{fold}_s_tr"], z[f"f{fold}_s_val"], z[f"f{fold}_te"]
        sc = StandardScaler().fit(X[s_tr])  # as for the MLP (monotone transform; kept for identical inputs)
        Xtr, Xval, Xte = sc.transform(X[s_tr]), sc.transform(X[s_val]), sc.transform(X[te])
        ytr, yval = y[s_tr], y[s_val]
        targets = build_targets(ytr, z[f"f{fold}_teacher_inner_oof"].astype(np.float64))
        for arm in ARMS:
            pv, pt, it = train_xgb(Xtr, targets[arm], Xval, yval, Xte, seed * 1000 + fold)
            probs[f"xgb_{arm}"][te] = pt
            val_rows.append({"cohort": cohort, "seed": seed, "fold": fold, "learner": "xgb", "arm": arm,
                             "val_macro_f1": f1_score(yval, pv.argmax(1), average="macro", zero_division=0),
                             "val_nll": log_loss(yval, np.clip(pv, 1e-12, 1), labels=list(range(K))),
                             "best_iter": it, "target_mean_entropy": float(entropy(targets[arm]).mean()),
                             "target_mean_true_class_prob": float(targets[arm][np.arange(len(ytr)), ytr].mean())})
        f1 = lambda p: 100 * f1_score(y[te], p[te].argmax(1), average="macro")  # noqa: E731
        print(f"[{cohort} seed {seed}] xgb fold {fold}: test F1 H={f1(probs['xgb_H']):.2f} "
              f"S.15={f1(probs['xgb_S_e0.15']):.2f} D.5/1={f1(probs['xgb_D_a0.50_T1']):.2f} | {time.time() - t0:.0f}s",
              flush=True)
    assert all(np.isfinite(v).all() for v in probs.values()), "missing out-of-fold predictions"
    pd.DataFrame(val_rows).to_csv(out_dir / f"seed{seed}_xgb_val.csv", index=False)
    np.savez_compressed(out_dir / f"seed{seed}_xgb.npz", y=y, uid=d["uid"].to_numpy(), arms=np.array(ARMS), **probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    d, seeds = load_cohort(a.cohort)
    seeds = a.seeds or seeds
    out_dir = OUT_ROOT / a.cohort
    json.dump({"cohort": a.cohort, "seeds": seeds, "xgb_params": XGB_PARAMS,
               "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                            "sklearn": sklearn.__version__, "numpy": np.__version__},
               "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name("wp3_common.py"))}},
              open(out_dir / "run_manifest_xgb.json", "w"), indent=2)
    for s in seeds:
        if (out_dir / f"seed{s}_xgb.npz").exists():
            print(f"seed {s} already done, skipped"); continue
        assert (out_dir / f"seed{s}_teacher.npz").exists(), f"run wp3_teacher_mlp.py for seed {s} first"
        run_seed(a.cohort, d, s, out_dir)


if __name__ == "__main__":
    main()
