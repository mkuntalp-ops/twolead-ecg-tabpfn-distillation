#!/usr/bin/env python3
"""WP3 seed-averaged paired bootstrap (added after the WP3 results were seen; it is the summary used for the
submitted v21 manuscript and complements, not replaces, the pre-specified per-seed summaries of wp3_analyze.py).

For a cohort with seeds s = 1..S (S = 5 or 1) and models A, B: D_AB = (1/S) sum_s [M(y, p_sA) - M(y, p_sB)], where
p_s are the pooled out-of-fold predictions of partition seed s. In each of B = 10,000 replicates one patient index
vector (patients drawn with replacement, all their records taken) is applied to every seed and model. The interval is
the 2.5th/97.5th percentile; the two-sided tail estimate is min(1, 2(n_opp + 1)/(B + 1)) with
n_opp = min(#{D_b <= 0}, #{D_b >= 0}); Holm adjustment over the two primary contrasts (D_tuned - S_tuned,
D_tuned - EM_tuned) within each cohort and learner. Metrics: Macro F1 (%), NLL, top-label ECE (%, 15 bins).
Fast implementations are checked against scikit-learn / WP4 on the point estimates.

Usage (project root):  python revizyon_v22/wp3/wp3_pooled.py
Writes revizyon_v22/results/wp3/summary/wp3_contrasts_pooled.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp3_analyze as A  # noqa: E402
from wp3_common import COHORTS, K, OUT_ROOT  # noqa: E402

B, SEED = 10_000, 2026
PRIMARY = [("D_tuned", "S_tuned"), ("D_tuned", "EM_tuned")]


def macro_f1(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return 100 * np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0))


def nll(y, p):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def ece(y, p, n_bins=15):
    conf = p.max(1); corr = p.argmax(1) == y
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    n = np.bincount(b, minlength=n_bins)
    sc = np.bincount(b, conf, n_bins); sa = np.bincount(b, corr.astype(float), n_bins)
    return 100 * float(np.sum(np.abs(sa - sc)) / len(y))


METRICS = {"macro_F1": lambda y, p: macro_f1(y, p.argmax(1)), "NLL": nll, "ECE_top_label_15bin": ece}


def main():
    rows = []
    for cohort in COHORTS:
        cdir = OUT_ROOT / cohort
        seeds = sorted(int(p.stem[4:-4]) for p in cdir.glob("seed*_mlp.npz"))
        per_seed = []
        for seed in seeds:
            z = dict(np.load(cdir / f"seed{seed}_mlp.npz", allow_pickle=True))
            zx = dict(np.load(cdir / f"seed{seed}_xgb.npz", allow_pickle=True))
            assert np.array_equal(zx["uid"], z["uid"])
            z.update({k: v for k, v in zx.items() if k.startswith("xgb_")})
            val = pd.concat([pd.read_csv(cdir / f"seed{seed}_mlp_val.csv"), pd.read_csv(cdir / f"seed{seed}_xgb_val.csv")])
            sel = A.select(val); fold = z["fold"].astype(int)
            per_seed.append({lrn: A.model_probs(z, fold, sel, lrn) for lrn in ("mlp", "xgb")})
            y, group = z["y"].astype(int), z["group"].astype(str)
            if seed == seeds[0]:
                y0, g0 = y, group
            assert np.array_equal(y, y0) and np.array_equal(group, g0), "record order differs between seeds"
            p = per_seed[-1]["mlp"]["D_tuned"]  # implementation checks
            assert abs(macro_f1(y, p.argmax(1)) - 100 * f1_score(y, p.argmax(1), average="macro")) < 1e-9
            assert abs(nll(y, p) - A.wp4.metrics(y, p)["NLL"]) < 1e-9  # WP4 NLL (no renormalization)
            pc = np.clip(p, 1e-12, 1); pc = pc / pc.sum(1, keepdims=True)  # sklearn renormalizes rows; same clipping
            assert abs(nll(y, pc) - log_loss(y, pc, labels=list(range(K)))) < 1e-9
            assert abs(ece(y, p) - A.wp4.metrics(y, p)["ECE_top_label_15bin"]) < 1e-9
        uniq, inv = np.unique(g0, return_inverse=True)
        members = [np.where(inv == i)[0] for i in range(len(uniq))]
        rng = np.random.default_rng(SEED)
        idx = [np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(B)]
        for lrn in ("mlp", "xgb"):
            names = sorted({n for c in A.CONTRASTS for n in c})
            # seed-averaged metric of every model in every replicate (computed once, then differenced)
            # per-record quantities, so that each replicate needs only integer bincounts
            pre = {n: [(s[lrn][n].argmax(1), np.log(np.clip(s[lrn][n][np.arange(len(y0)), y0], 1e-12, 1)),
                        s[lrn][n].max(1), (s[lrn][n].argmax(1) == y0).astype(float)) for s in per_seed] for n in names}

            def stats(n, ix):
                f1, nl, ec = [], [], []
                for pred, lp, conf, corr in pre[n]:
                    yy, pp = y0[ix], pred[ix]
                    f1.append(macro_f1(yy, pp)); nl.append(-lp[ix].mean())
                    c, k = conf[ix], corr[ix]
                    bins = np.minimum((c * 15).astype(int), 14)
                    ec.append(100 * np.abs(np.bincount(bins, k, 15) - np.bincount(bins, c, 15)).sum() / len(ix))
                return np.mean(f1), np.mean(nl), np.mean(ec)

            full = np.arange(len(y0))
            pts = {n: dict(zip(METRICS, stats(n, full))) for n in names}
            for n in names:  # fast statistics equal the reference implementations
                for mm, fn in METRICS.items():
                    assert abs(pts[n][mm] - np.mean([fn(y0, s[lrn][n]) for s in per_seed])) < 1e-9, (n, mm)
            dr = {n: np.array([stats(n, ix) for ix in idx]) for n in names}
            drw = {mm: {n: dr[n][:, j] for n in names} for j, mm in enumerate(METRICS)}
            pts = {mm: {n: pts[n][mm] for n in names} for mm in METRICS}
            for a, b in A.CONTRASTS:
                row = {"cohort": cohort, "learner": lrn, "contrast": f"{a} - {b}", "n_seeds": len(seeds), "n": len(y0),
                       "primary": (a, b) in PRIMARY}
                for m in METRICS:
                    draws = drw[m][a] - drw[m][b]
                    n_opp = min(np.sum(draws <= 0), np.sum(draws >= 0))
                    row.update({f"d_{m}": pts[m][a] - pts[m][b], f"d_{m}_lo": np.percentile(draws, 2.5),
                                f"d_{m}_hi": np.percentile(draws, 97.5),
                                f"p_{m}": min(1.0, 2 * (n_opp + 1) / (B + 1))})
                rows.append(row)
            print(f"{cohort} {lrn} done", flush=True)
    df = pd.DataFrame(rows)
    df["p_macro_F1_holm_primary"] = np.nan
    for (c, l), g in df[df.primary].groupby(["cohort", "learner"]):
        order = g["p_macro_F1"].sort_values().index
        adj, run = [], 0.0
        for i, ix in enumerate(order):
            run = max(run, min(1.0, (len(order) - i) * df.loc[ix, "p_macro_F1"])); adj.append((ix, run))
        for ix, v in adj:
            df.loc[ix, "p_macro_F1_holm_primary"] = v
    df.to_csv(OUT_ROOT / "summary" / "wp3_contrasts_pooled.csv", index=False)
    print(df[["cohort", "learner", "contrast", "d_macro_F1", "d_macro_F1_lo", "d_macro_F1_hi", "p_macro_F1",
              "p_macro_F1_holm_primary"]].round(4).to_string())


if __name__ == "__main__":
    main()
