#!/usr/bin/env python3
"""(Usage, project root: python -I revizyon_v22/wp3/wp3_independent_check.py revizyon_v22/results/wp3)
Independent recomputation of WP3 Macro F1 and paired differences from stored OOF predictions.
Own selection rule implementation, sklearn metrics, own bootstrap RNG. Does not import project code."""
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
R = Path(sys.argv[1])
COH = ["ptbxl_quota5000", "ptbxl_natural5000", "ptbxl_natural15160", "chapman_quota5000", "chapman_natural"]
met = pd.read_csv(R / "summary/wp3_metrics_per_seed.csv")
pooled = pd.read_csv(R / "summary/wp3_contrasts_pooled.csv")
perseed = pd.read_csv(R / "summary/wp3_contrasts_per_seed.csv")
fam_of = lambda a: a.split("_")[0]
worst = {"f1": 0, "dseed": 0, "dpool": 0, "ci": 0}; rows = []
CONS = [("D_tuned", "S_tuned"), ("D_tuned", "EM_tuned"), ("D_tuned", "H")]
for c in COH:
    seeds = sorted(int(re.search(r"seed(\d+)_mlp", p).group(1)) for p in glob.glob(str(R / c / "seed*_mlp.npz")))
    per = {}
    for s in seeds:
        z = dict(np.load(R / c / f"seed{s}_mlp.npz", allow_pickle=True))
        val = [pd.read_csv(R / c / f"seed{s}_mlp_val.csv")]
        if (R / c / f"seed{s}_xgb.npz").exists():
            z.update({k: v for k, v in np.load(R / c / f"seed{s}_xgb.npz", allow_pickle=True).items() if k.startswith("xgb_")})
            val.append(pd.read_csv(R / c / f"seed{s}_xgb_val.csv"))
        val = pd.concat(val); y = z["y"].astype(int); fold = z["fold"].astype(int)
        grid = list(z["arms"])
        for lrn in val.learner.unique():
            P = {"H": z[f"{lrn}_H"]}
            for fam in ("S", "D", "EM"):
                out = np.zeros_like(P["H"])
                for f in np.unique(fold):
                    v = val[(val.learner == lrn) & (val.fold == f) & (val.arm.map(fam_of) == fam)].copy()
                    v["o"] = v.arm.map(grid.index)
                    best = v.sort_values(["val_macro_f1", "val_nll", "o"], ascending=[False, True, True]).arm.iloc[0]
                    out[fold == f] = z[f"{lrn}_{best}"][fold == f]
                P[f"{fam}_tuned"] = out
            for m, p in P.items():
                f1 = 100 * f1_score(y, p.argmax(1), average="macro")
                ref = met[(met.cohort == c) & (met.seed == s) & (met.learner == lrn) & (met.model == m)].macro_F1
                worst["f1"] = max(worst["f1"], abs(f1 - ref.iloc[0]))
            per[(s, lrn)] = (y, P)
    for lrn in ("mlp", "xgb"):
        ys = [per[(s, lrn)][0] for s in seeds]; assert all(np.array_equal(ys[0], q) for q in ys)
        y = ys[0]; n = len(y); rng = np.random.default_rng(12345)
        idx = rng.integers(0, n, size=(2000, n))
        for a, b in CONS:
            pa = [per[(s, lrn)][1][a].argmax(1) for s in seeds]; pb = [per[(s, lrn)][1][b].argmax(1) for s in seeds]
            dseed = [100 * (f1_score(y, x, average="macro") - f1_score(y, w, average="macro")) for x, w in zip(pa, pb)]
            for s, d in zip(seeds, dseed):
                r = perseed[(perseed.cohort == c) & (perseed.seed == s) & (perseed.learner == lrn) & (perseed.contrast == f"{a} - {b}")]
                worst["dseed"] = max(worst["dseed"], abs(d - r.d_macro_F1.iloc[0]))
            d = np.mean(dseed)
            r = pooled[(pooled.cohort == c) & (pooled.learner == lrn) & (pooled.contrast == f"{a} - {b}")].iloc[0]
            worst["dpool"] = max(worst["dpool"], abs(d - r.d_macro_F1))
            boot = np.array([np.mean([100 * (f1_score(y[i], x[i], average="macro") - f1_score(y[i], w[i], average="macro"))
                                      for x, w in zip(pa, pb)]) for i in idx[:400]])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            worst["ci"] = max(worst["ci"], abs(lo - r.d_macro_F1_lo), abs(hi - r.d_macro_F1_hi))
            rows.append((c, lrn, f"{a}-{b}", round(d, 3), round(r.d_macro_F1, 3), round(lo, 2), round(r.d_macro_F1_lo, 2), round(hi, 2), round(r.d_macro_F1_hi, 2)))
print(pd.DataFrame(rows, columns=["cohort", "lrn", "con", "d_indep", "d_stored", "lo_indep400", "lo_stored", "hi_indep400", "hi_stored"]).to_string())
print("max abs differences:", worst)
