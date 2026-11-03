#!/usr/bin/env python3
"""WP3 analysis: nested selection on student-validation, test metrics and paired patient-blocked bootstrap.

For every cohort, partition seed, learner (mlp, xgb) and outer fold, the configuration of each arm family (S, D, EM)
with the highest student-val Macro F1 is chosen (ties: lower val NLL, then grid order); the outer-test predictions of
the chosen configuration form the "tuned" out-of-fold predictions. Test data are not used for any choice.
Metric definitions are those of WP4 (`revizyon_v22/analysis/wp4_calibration_discrimination_subgroups.py`).

Usage (project root):  python revizyon_v22/wp3/wp3_analyze.py [--cohorts ...]
Outputs (revizyon_v22/results/wp3/summary/):
  wp3_metrics_per_seed.csv     metrics of every reported model per cohort/seed (with bootstrap 95% intervals)
  wp3_contrasts_per_seed.csv   paired differences per cohort/seed (same bootstrap index vectors)
  wp3_contrasts_summary.csv    mean, SD, min, max of the per-seed differences
  wp3_grid_test_f1.csv         descriptive grid: test Macro F1 of every arm (mean over seeds), with val F1 and entropy
  wp3_selected_configs.csv     how often each configuration was selected
  wp3_analysis_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp3_common import ALPHAS, ARMS, COHORTS, K, OUT_ROOT, TEMPS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("wp4", ROOT / "revizyon_v22/analysis/wp4_calibration_discrimination_subgroups.py")
wp4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(wp4)
B, BOOT_SEED = 2000, 2026
MAIN_METRICS = ["macro_F1", "macro_AUROC", "NLL", "ECE_top_label_15bin"]
FAMILIES = {"S": [a for a in ARMS if a.startswith("S_")], "D": [a for a in ARMS if a.startswith("D_")],
            "EM": [a for a in ARMS if a.startswith("EM_")]}
CONTRASTS = [("D_tuned", "S_tuned"), ("D_tuned", "EM_tuned"), ("D_tuned", "H"), ("S_tuned", "H"),
             ("EM_tuned", "H"), ("D_fixed", "S_fixed"), ("D_fixed", "H")]
OUT = OUT_ROOT / "summary"


def select(val: pd.DataFrame) -> pd.DataFrame:
    """Chosen arm per (learner, fold, family)."""
    rows = []
    order = {a: i for i, a in enumerate(ARMS)}
    for (lrn, fold), g in val.groupby(["learner", "fold"]):
        for fam, arms in FAMILIES.items():
            h = g[g["arm"].isin(arms)].assign(o=lambda t: t["arm"].map(order))
            best = h.sort_values(["val_macro_f1", "val_nll", "o"], ascending=[False, True, True]).iloc[0]
            rows.append({"learner": lrn, "fold": fold, "family": fam, "arm": best["arm"],
                         "val_macro_f1": best["val_macro_f1"]})
    return pd.DataFrame(rows)


def model_probs(z: dict, fold: np.ndarray, sel: pd.DataFrame, lrn: str) -> dict[str, np.ndarray]:
    out = {"H": z[f"{lrn}_H"], "S_fixed": z[f"{lrn}_S_e0.15"], "D_fixed": z[f"{lrn}_D_a0.50_T1"]}
    for fam in FAMILIES:
        p = np.full_like(z[f"{lrn}_H"], np.nan, dtype=np.float64)
        for f in range(1, 6):
            arm = sel.query("learner == @lrn and fold == @f and family == @fam")["arm"].item()
            p[fold == f] = z[f"{lrn}_{arm}"][fold == f]
        out[f"{fam}_tuned"] = p
    return {k: v.astype(np.float64) for k, v in out.items()}


def boot_indices(group: np.ndarray, seed: int) -> list[np.ndarray]:
    """Patient-blocked resampling: draw patients with replacement and take all their records."""
    uniq, inv = np.unique(group, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(B)]


def analyse_cohort(cohort: str):
    cdir = OUT_ROOT / cohort
    seeds = sorted(int(p.stem[4:-4]) for p in cdir.glob("seed*_mlp.npz"))
    met_rows, con_rows, grid_rows, sel_rows = [], [], [], []
    for seed in seeds:
        zm = dict(np.load(cdir / f"seed{seed}_mlp.npz", allow_pickle=True))
        y, fold, group = zm["y"].astype(int), zm["fold"].astype(int), zm["group"].astype(str)
        val = [pd.read_csv(cdir / f"seed{seed}_mlp_val.csv")]
        learners = ["mlp"]
        if (cdir / f"seed{seed}_xgb.npz").exists():
            zx = dict(np.load(cdir / f"seed{seed}_xgb.npz", allow_pickle=True))
            assert np.array_equal(zx["y"], zm["y"]) and np.array_equal(zx["uid"], zm["uid"]), "mlp/xgb record mismatch"
            zm.update({k: v for k, v in zx.items() if k.startswith("xgb_")})
            val.append(pd.read_csv(cdir / f"seed{seed}_xgb_val.csv")); learners.append("xgb")
        val = pd.concat(val, ignore_index=True)
        sel = select(val)
        sel_rows.append(sel.assign(cohort=cohort, seed=seed))
        idx = boot_indices(group, BOOT_SEED + seed)
        models = {"teacher_tabpfn": {"teacher": zm["teacher_tabpfn"].astype(np.float64)}}
        for lrn in learners:
            models[lrn] = model_probs(zm, fold, sel, lrn)
        for lrn, mp in models.items():
            for name, p in mp.items():
                assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-4), f"{lrn} {name}"
            draws = wp4.bootstrap_ci(y, mp, idx, MAIN_METRICS)
            for name, p in mp.items():
                point = wp4.metrics(y, p)
                row = {"cohort": cohort, "seed": seed, "learner": lrn, "model": name, "n": len(y)}
                for m in MAIN_METRICS:
                    row[m] = point[m]
                    row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(draws[name][m], [2.5, 97.5])
                met_rows.append(row)
            if lrn == "teacher_tabpfn":
                continue
            for a, b in CONTRASTS:
                pa, pb = wp4.metrics(y, mp[a]), wp4.metrics(y, mp[b])
                row = {"cohort": cohort, "seed": seed, "learner": lrn, "contrast": f"{a} - {b}"}
                for m in MAIN_METRICS:
                    dd = draws[a][m] - draws[b][m]
                    row[f"d_{m}"] = pa[m] - pb[m]
                    row[f"d_{m}_lo"], row[f"d_{m}_hi"] = np.percentile(dd, [2.5, 97.5])
                con_rows.append(row)
            for arm in ARMS:
                p = zm[f"{lrn}_{arm}"].astype(np.float64)
                v = val.query("learner == @lrn and arm == @arm")
                grid_rows.append({"cohort": cohort, "seed": seed, "learner": lrn, "arm": arm,
                                  "test_macro_F1": wp4.metrics(y, p)["macro_F1"],
                                  "val_macro_F1_mean_folds": 100 * v["val_macro_f1"].mean(),
                                  "target_mean_entropy": v["target_mean_entropy"].mean()})
        print(f"{cohort} seed {seed}: done", flush=True)
    return met_rows, con_rows, grid_rows, sel_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=COHORTS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    met, con, grid, sel = [], [], [], []
    for c in a.cohorts:
        if not list((OUT_ROOT / c).glob("seed*_mlp.npz")):
            print(f"{c}: no results yet, skipped"); continue
        r = analyse_cohort(c)
        met += r[0]; con += r[1]; grid += r[2]; sel += r[3]
    met, con, grid = pd.DataFrame(met), pd.DataFrame(con), pd.DataFrame(grid)
    sel = pd.concat(sel, ignore_index=True)
    met.to_csv(OUT / "wp3_metrics_per_seed.csv", index=False)
    con.to_csv(OUT / "wp3_contrasts_per_seed.csv", index=False)
    dcols = [f"d_{m}" for m in MAIN_METRICS]
    summ = con.groupby(["cohort", "learner", "contrast"])[dcols].agg(["mean", "std", "min", "max", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(OUT / "wp3_contrasts_summary.csv", index=False)
    g = grid.groupby(["cohort", "learner", "arm"])[["test_macro_F1", "val_macro_F1_mean_folds", "target_mean_entropy"]].mean()
    g = g.reset_index()
    em = {(r.cohort, r.learner, r.arm[3:]): r.test_macro_F1 for r in g.itertuples() if r.arm.startswith("EM_")}
    g["test_F1_D_minus_matched_EM"] = [r.test_macro_F1 - em[(r.cohort, r.learner, r.arm[2:])] if r.arm.startswith("D_") else np.nan
                                       for r in g.itertuples()]
    g.to_csv(OUT / "wp3_grid_test_f1.csv", index=False)
    sel.groupby(["cohort", "learner", "family", "arm"]).size().rename("times_selected").reset_index() \
       .to_csv(OUT / "wp3_selected_configs.csv", index=False)
    z = np.load(sorted(p for c in a.cohorts for p in (OUT_ROOT / c).glob("seed*_mlp.npz"))[0], allow_pickle=True)
    y0, p0 = z["y"].astype(int), z["mlp_H"].astype(np.float64)
    json.dump({"bootstrap": {"B": B, "seed": f"{BOOT_SEED} + partition seed", "unit": "patient"},
               "contrasts": [f"{x} - {y}" for x, y in CONTRASTS], "families": FAMILIES,
               "alphas": ALPHAS, "temperatures": TEMPS,
               "self_checks_on_first_result": wp4.self_checks(y0, p0),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              open(OUT / "wp3_analysis_manifest.json", "w"), indent=2)
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
