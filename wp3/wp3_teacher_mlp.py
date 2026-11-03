#!/usr/bin/env python3
"""WP3, step 1: TabPFN teacher (inner 3-fold grouped OOF, n_estimators = 8) and MLP students for every arm.

MLP training is the Phase 3B procedure (`experiments_phase3/03_procedure_matched_repeated_eval.py`): 50-128-64-5
(15,109 parameters), Adam lr 1e-3, weight decay 1e-3, batch 64, <= 200 epochs, early stopping on student-val Macro F1
(patience 15), shuffling generator seeded seed*10000 + fold*1000 + epoch, identical initial weights for all arms of a
fold (torch.manual_seed(seed*1000 + fold)). Only the student-validation set is used for any choice.

Usage (project root):  python revizyon_v22/wp3/wp3_teacher_mlp.py --cohort ptbxl_quota5000 [--seeds 42 43]
Writes revizyon_v22/results/wp3/<cohort>/: seed<k>_mlp.npz (OOF test probabilities of every arm and the reference
teacher), seed<k>_teacher.npz (splits and inner-OOF teacher probabilities, read by wp3_xgb.py), seed<k>_mlp_val.csv,
run_manifest_mlp.json. Finished seeds are skipped.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp3_common import (ARMS, CLASSES, FEAT_50, K, NATURAL5000_SAMPLING_SEED, OUT_ROOT, build_targets,  # noqa: E402
                        entropy, load_cohort, partitions)

TABPFN_CKPT = "/Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
TEACHER_N_ESTIMATORS = 8


def as_t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


class PyTorchMLP(nn.Module):  # Phase 3B architecture
    def __init__(self, in_features=50, h1=128, h2=64, num_classes=K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, num_classes))

    def forward(self, x):
        return self.net(x)


def train_mlp(init_state, X_tr, Q, X_val, y_val, X_te, seed, fold, batch_size=64, lr=1e-3, weight_decay=1e-3,
              max_epochs=200, patience=15):
    model = PyTorchMLP(); model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(as_t(X_tr), as_t(Q))
    Xv = as_t(X_val)
    best_f1, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        g = torch.Generator(); g.manual_seed(seed * 10000 + fold * 1000 + epoch)
        for bx, bq in DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g):
            opt.zero_grad()
            loss = -torch.mean(torch.sum(bq * F.log_softmax(model(bx), 1), 1))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vf1 = f1_score(y_val, model(Xv).argmax(1).numpy(), average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_epoch, wait = vf1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return F.softmax(model(Xv), 1).numpy(), F.softmax(model(as_t(X_te)), 1).numpy(), best_epoch


def tabpfn() -> TabPFNClassifier:
    return TabPFNClassifier(model_path=TABPFN_CKPT, device="mps", n_estimators=TEACHER_N_ESTIMATORS)


def run_seed(cohort: str, d: pd.DataFrame, seed: int, out_dir: Path) -> None:
    X = d[FEAT_50].to_numpy(np.float64); y = d["y"].to_numpy(np.int64); grp = d["group"].to_numpy()
    N = len(d)
    probs = {f"mlp_{a}": np.full((N, K), np.nan, np.float32) for a in ARMS}
    probs["teacher_tabpfn"] = np.full((N, K), np.nan, np.float32)
    fold_of = np.zeros(N, np.int8)
    split_store, val_rows = {}, []
    for fold, s_tr, s_val, te, inner in partitions(y, grp, seed):
        t0 = time.time()
        sc = StandardScaler().fit(X[s_tr])
        Xtr, Xval, Xte = sc.transform(X[s_tr]), sc.transform(X[s_val]), sc.transform(X[te])
        ytr, yval = y[s_tr], y[s_val]
        teacher = np.zeros((len(s_tr), K))
        for itr, iva in inner:
            t = tabpfn().fit(Xtr[itr], ytr[itr])
            assert np.array_equal(t.classes_, np.arange(K)), "teacher class order"
            teacher[iva] = t.predict_proba(Xtr[iva])
        assert np.allclose(teacher.sum(1), 1.0, atol=1e-4), "teacher rows do not sum to 1"
        ref = tabpfn().fit(Xtr, ytr)
        probs["teacher_tabpfn"][te] = ref.predict_proba(Xte)
        split_store.update({f"f{fold}_s_tr": s_tr, f"f{fold}_s_val": s_val, f"f{fold}_te": te,
                            f"f{fold}_teacher_inner_oof": teacher.astype(np.float32)})
        targets = build_targets(ytr, teacher)

        torch.manual_seed(seed * 1000 + fold)
        init_state = {k: v.clone() for k, v in PyTorchMLP().state_dict().items()}
        for arm in ARMS:
            pv, pt, it = train_mlp(init_state, Xtr, targets[arm], Xval, yval, Xte, seed, fold)
            probs[f"mlp_{arm}"][te] = pt
            val_rows.append({"cohort": cohort, "seed": seed, "fold": fold, "learner": "mlp", "arm": arm,
                             "val_macro_f1": f1_score(yval, pv.argmax(1), average="macro", zero_division=0),
                             "val_nll": log_loss(yval, np.clip(pv, 1e-12, 1), labels=list(range(K))),
                             "best_iter": it, "target_mean_entropy": float(entropy(targets[arm]).mean()),
                             "target_mean_true_class_prob": float(targets[arm][np.arange(len(ytr)), ytr].mean())})
        fold_of[te] = fold
        f1 = lambda p: 100 * f1_score(y[te], p[te].argmax(1), average="macro")  # noqa: E731
        print(f"[{cohort} seed {seed}] fold {fold}: n_tr={len(s_tr)} n_val={len(s_val)} n_te={len(te)} | test F1 "
              f"teacher={f1(probs['teacher_tabpfn']):.2f} H={f1(probs['mlp_H']):.2f} "
              f"S.15={f1(probs['mlp_S_e0.15']):.2f} D.5/1={f1(probs['mlp_D_a0.50_T1']):.2f} | {time.time() - t0:.0f}s",
              flush=True)
    assert all(np.isfinite(v).all() for v in probs.values()), "missing out-of-fold predictions"
    pd.DataFrame(val_rows).to_csv(out_dir / f"seed{seed}_mlp_val.csv", index=False)
    np.savez_compressed(out_dir / f"seed{seed}_teacher.npz", y=y, uid=d["uid"].to_numpy(), **split_store)
    np.savez_compressed(out_dir / f"seed{seed}_mlp.npz", y=y, uid=d["uid"].to_numpy(), group=grp, fold=fold_of,
                        arms=np.array(ARMS), **probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    d, seeds = load_cohort(a.cohort)
    seeds = a.seeds or seeds
    out_dir = OUT_ROOT / a.cohort; out_dir.mkdir(parents=True, exist_ok=True)
    d[["uid", "group", "y"]].to_csv(out_dir / "cohort_records.csv", index=False)
    import tabpfn as _tp
    json.dump({"cohort": a.cohort, "n": len(d),
               "class_counts": {CLASSES[k]: int(v) for k, v in d["y"].value_counts().sort_index().items()},
               "seeds": seeds, "arms": ARMS, "teacher_n_estimators": TEACHER_N_ESTIMATORS,
               "natural5000_sampling_seed": NATURAL5000_SAMPLING_SEED,
               "versions": {"python": platform.python_version(), "torch": torch.__version__,
                            "sklearn": sklearn.__version__, "tabpfn": _tp.__version__, "numpy": np.__version__},
               "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name("wp3_common.py"))}},
              open(out_dir / "run_manifest_mlp.json", "w"), indent=2)
    for s in seeds:
        if (out_dir / f"seed{s}_mlp.npz").exists():
            print(f"seed {s} already done, skipped"); continue
        run_seed(a.cohort, d, s, out_dir)


if __name__ == "__main__":
    main()
