#!/usr/bin/env python3
"""WP3 deployment cost, MLP student and TabPFN teacher (separate process from xgboost; see wp3_common).

MLP: parameter count, float32 state-dict size, CPU latency with one thread (random weights; latency does not depend on
the weight values) for a single record (median of 1,000 calls) and a batch of 1,000 records (median of 20 calls).
TabPFN (n_estimators = 8, as in WP3): checkpoint size and wall-clock time to predict the outer-test set of the
reference fold (first seed, fold 1) of every cohort, on MPS and on CPU, including fit (TabPFN's fit stores the
training set; the forward pass over training and test data happens at prediction time).

Usage (project root):  python revizyon_v22/wp3/wp3_cost_nn.py
Writes revizyon_v22/results/wp3/cost/nn_costs.csv.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp3_common import COHORTS, FEAT_50, OUT_ROOT, load_cohort  # noqa: E402
from wp3_teacher_mlp import TABPFN_CKPT, TEACHER_N_ESTIMATORS, PyTorchMLP, as_t  # noqa: E402

OUT = OUT_ROOT / "cost"


def latency(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows = []
    m = PyTorchMLP().eval()
    buf = io.BytesIO(); torch.save(m.state_dict(), buf)
    x1, x1000 = torch.randn(1, 50), torch.randn(1000, 50)
    with torch.no_grad():
        rows.append({"model": "mlp_50_128_64_5", "cohort": "all", "n_parameters": sum(p.numel() for p in m.parameters()),
                     "size_bytes": buf.getbuffer().nbytes,
                     "latency_1_record_ms": latency(lambda: m(x1), 1000),
                     "latency_1000_records_ms": latency(lambda: m(x1000), 20)})
    print(rows[-1], flush=True)
    torch.set_num_threads(os.cpu_count())
    for c in COHORTS:
        d, seeds = load_cohort(c)
        z = np.load(OUT_ROOT / c / f"seed{seeds[0]}_teacher.npz", allow_pickle=True)
        X = d[FEAT_50].to_numpy(np.float64); y = d["y"].to_numpy(np.int64)
        s_tr, te = z["f1_s_tr"], z["f1_te"]
        sc = StandardScaler().fit(X[s_tr]); Xtr, Xte = sc.transform(X[s_tr]), sc.transform(X[te])
        for dev in ("mps", "cpu"):
            if dev == "cpu" and len(s_tr) > 5000:  # tabpfn 9.0.0 refuses CPU runs with > 5,000 context samples
                rows.append({"model": f"tabpfn_v3_n{TEACHER_N_ESTIMATORS}_{dev}", "cohort": c, "n_context": len(s_tr),
                             "n_test": len(te), "note": "not run: tabpfn default CPU limit (> 5,000 samples)"})
                continue

            def run():
                TabPFNClassifier(model_path=TABPFN_CKPT, device=dev, n_estimators=TEACHER_N_ESTIMATORS) \
                    .fit(Xtr, y[s_tr]).predict_proba(Xte)
            run()  # warm-up (weights loaded, kernels compiled)
            t0 = time.perf_counter(); run(); sec = time.perf_counter() - t0
            rows.append({"model": f"tabpfn_v3_n{TEACHER_N_ESTIMATORS}_{dev}", "cohort": c,
                         "n_context": len(s_tr), "n_test": len(te), "size_bytes": os.path.getsize(TABPFN_CKPT),
                         "fit_predict_test_set_s": sec, "per_record_ms": 1e3 * sec / len(te)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "nn_costs.csv", index=False)


if __name__ == "__main__":
    main()
