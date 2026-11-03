#!/usr/bin/env python3
"""Tune FEAT_50 v2 fiducial parameters on a 1,000-record development subset against PTB-XL+ reference measurements.

Diagnostic labels are never read. Reference: Uni-G lead-specific QRS/QT, 12SL and Uni-G global PR.
Outputs (revizyon_v22/results/feat50_v2/): dev_grid.csv, dev_unchanged_features_check.csv, dev_manifest.json
"""
from __future__ import annotations

import itertools
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feat50_v2 import DEFAULT_PARAMS, SUFFIXES, extract_feat50_v2  # noqa: E402

OUT = ROOT / "revizyon_v22/results/feat50_v2"; OUT.mkdir(parents=True, exist_ok=True)
MAN = ROOT / "experiments_phase3/phase3d_full_cohort/phase3d_eligible_cohort_15160_manifest.csv"
V1 = ROOT / "experiments_phase3/phase3d_full_cohort/ptbxl_eligible_cohort_15160_features.csv"
ZIP = Path("/Volumes/KIOXIA/Datasets/ptb-xl/ptb-xl-data.zip")
SEED, N_DEV = 2026, 1000


def load_signals(zf, names, rel):
    path = rel + ".dat"
    if path not in names:
        path = next(n for n in names if n.endswith(path))
    sig = np.frombuffer(zf.read(path), dtype="<i2").reshape(-1, 12) / 1000.0
    return sig[:, 0], sig[:, 1]


def references(ecg_ids):
    u = pd.read_csv(ROOT / "data/ptbxl_plus/unig_features.csv",
                    usecols=["ecg_id", "QRS_Dur_I", "QRS_Dur_II", "QT_Int_I", "QT_Int_II", "PR_Int_Global"])
    s = pd.read_csv(ROOT / "data/ptbxl_plus/12sl_features.csv", usecols=["ecg_id", "PR_Int_Global"]).rename(
        columns={"PR_Int_Global": "PR_12SL"})
    r = u.rename(columns={"PR_Int_Global": "PR_UNIG"}).merge(s, on="ecg_id", how="outer")
    return r[r.ecg_id.isin(ecg_ids)].set_index("ecg_id")


def agreement(feat: pd.DataFrame, ref: pd.DataFrame) -> dict:
    m = feat.join(ref, how="inner")
    out = {}
    for name, a, b in [("PR_L2_vs_12SL", "L2_pr_int", "PR_12SL"), ("PR_L2_vs_UNIG", "L2_pr_int", "PR_UNIG"),
                       ("PR_L1_vs_12SL", "L1_pr_int", "PR_12SL"),
                       ("QRS_L1_vs_UNIG", "L1_qrs_dur", "QRS_Dur_I"), ("QRS_L2_vs_UNIG", "L2_qrs_dur", "QRS_Dur_II"),
                       ("QT_L1_vs_UNIG", "L1_qt_int", "QT_Int_I"), ("QT_L2_vs_UNIG", "L2_qt_int", "QT_Int_II")]:
        d = m[[a, b]].dropna()
        out[f"{name}_r"] = stats.pearsonr(d[a], d[b])[0]
        out[f"{name}_meddiff"] = float(np.median(d[a] - d[b]))
    out["pct_L1_qrs_ge146"] = 100 * float(np.mean(feat["L1_qrs_dur"] >= 146))
    out["pct_L2_pr_at_clip"] = 100 * float(np.mean((feat["L2_pr_int"] <= 80) | (feat["L2_pr_int"] >= 360)))
    return out


def main():
    man = pd.read_csv(MAN)
    dev = man.sample(n=N_DEV, random_state=SEED).sort_values("ecg_id")
    v1 = pd.read_csv(V1).set_index("ecg_id")
    ref = references(dev.ecg_id)
    zf = zipfile.ZipFile(ZIP); names = set(zf.namelist())
    sigs = {int(r.ecg_id): load_signals(zf, names, r.filename_hr) for r in dev.itertuples()}

    rows = []
    v1_dev = v1.loc[dev.ecg_id, [f"{l}_{s}" for l in ("L1", "L2") for s in SUFFIXES]]
    rows.append({"variant": "v1 (stored)", **agreement(v1_dev, ref)})
    grid = {"qrs_thr_frac": [0.05, 0.085, 0.12], "qrs_gap_ms": [10.0, 20.0], "p_thr_frac": [0.10, 0.20, 0.30]}
    feats_default = None
    for vals in itertools.product(*grid.values()):
        prm = dict(zip(grid.keys(), vals))
        fv = pd.DataFrame({e: extract_feat50_v2(*sigs[e], params=prm) for e in sigs}).T
        fv.index.name = "ecg_id"
        rows.append({"variant": "v2 " + json.dumps(prm), **prm, **agreement(fv, ref)})
        if all(DEFAULT_PARAMS[k] == v for k, v in prm.items()): feats_default = fv
    g = pd.DataFrame(rows); g.to_csv(OUT / "dev_grid.csv", index=False)

    # features whose definition is unchanged: RR, spectral/statistical exactly; amplitudes approximately
    chk = []
    for s in SUFFIXES:
        for l in ("L1", "L2"):
            a, b = v1_dev[f"{l}_{s}"].to_numpy(float), feats_default[f"{l}_{s}"].to_numpy(float)
            chk.append({"feature": f"{l}_{s}", "max_abs_diff": float(np.max(np.abs(a - b))),
                        "pearson_r_v1_v2": float(stats.pearsonr(a, b)[0]) if np.std(a) > 0 and np.std(b) > 0 else np.nan})
    pd.DataFrame(chk).to_csv(OUT / "dev_unchanged_features_check.csv", index=False)
    json.dump({"seed": SEED, "n_dev": N_DEV, "dev_ecg_ids": dev.ecg_id.tolist(), "grid": grid,
               "default_params": DEFAULT_PARAMS}, open(OUT / "dev_manifest.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    cols = ["variant", "PR_L2_vs_12SL_r", "PR_L2_vs_UNIG_r", "PR_L2_vs_12SL_meddiff", "QRS_L1_vs_UNIG_r", "QRS_L2_vs_UNIG_r",
            "QRS_L2_vs_UNIG_meddiff", "QT_L2_vs_UNIG_r", "QT_L2_vs_UNIG_meddiff", "pct_L1_qrs_ge146", "pct_L2_pr_at_clip"]
    print(g[cols].round(3).to_string(index=False))
    print(pd.DataFrame(chk).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
