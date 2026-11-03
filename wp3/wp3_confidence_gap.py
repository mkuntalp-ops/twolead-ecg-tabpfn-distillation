#!/usr/bin/env python3
"""Direction of miscalibration: mean top-label confidence minus accuracy (percentage points) of every reported
model, averaged over partitions (negative = under-confident). Same nested selection as wp3_analyze.py.

Usage (project root):  python revizyon_v22/wp3/wp3_confidence_gap.py
Writes revizyon_v22/results/wp3/summary/wp3_confidence_gap.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp3_analyze as A  # noqa: E402
from wp3_common import COHORTS  # noqa: E402

rows = []
for c in COHORTS:
    cdir = A.OUT_ROOT / c
    for seed in sorted(int(p.stem[4:-4]) for p in cdir.glob("seed*_mlp.npz")):
        z = dict(np.load(cdir / f"seed{seed}_mlp.npz", allow_pickle=True))
        zx = dict(np.load(cdir / f"seed{seed}_xgb.npz", allow_pickle=True))
        z.update({k: v for k, v in zx.items() if k.startswith("xgb_")})
        val = pd.concat([pd.read_csv(cdir / f"seed{seed}_mlp_val.csv"), pd.read_csv(cdir / f"seed{seed}_xgb_val.csv")])
        sel = A.select(val); y = z["y"].astype(int)
        probs = {("teacher_tabpfn", "teacher"): z["teacher_tabpfn"].astype(np.float64)}
        for lrn in ("mlp", "xgb"):
            probs.update({(lrn, n): p for n, p in A.model_probs(z, z["fold"].astype(int), sel, lrn).items()})
        for (lrn, n), p in probs.items():
            rows.append({"cohort": c, "seed": seed, "learner": lrn, "model": n,
                         "mean_confidence": 100 * p.max(1).mean(), "accuracy": 100 * (p.argmax(1) == y).mean(),
                         "confidence_minus_accuracy": 100 * (p.max(1).mean() - (p.argmax(1) == y).mean())})
d = pd.DataFrame(rows)
d.groupby(["cohort", "learner", "model"])[["mean_confidence", "accuracy", "confidence_minus_accuracy"]].mean() \
    .round(3).reset_index().to_csv(A.OUT_ROOT / "summary" / "wp3_confidence_gap.csv", index=False)
