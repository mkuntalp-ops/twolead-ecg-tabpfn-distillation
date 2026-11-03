#!/usr/bin/env python3
"""Re-extract FEAT_50 with the v2 fiducial corrections for the 15,160 eligible PTB-XL patients.

Writes v2 copies of the two feature tables used by the experiments; every non-FEAT_50 column (identifiers, labels,
demographics, legacy flags) is copied unchanged from the v1 table, so the experiment scripts only need a new path.
Outputs (revizyon_v22/data_v2/):
  ptbxl_eligible_cohort_15160_features_v2.csv   (Phase 3D input, same rows/order as v1)
  ptbxl_54features_v2.csv                       (Phase 3B/3C input, same rows/order as data/ptbxl_54features.csv)
  extraction_v2_debug.csv, extraction_v2_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feat50_v2 import DEFAULT_PARAMS, SUFFIXES, extract_feat50_v2  # noqa: E402

OUT = ROOT / "revizyon_v22/data_v2"; OUT.mkdir(parents=True, exist_ok=True)
V1_15160 = ROOT / "experiments_phase3/phase3d_full_cohort/ptbxl_eligible_cohort_15160_features.csv"
V1_5000 = ROOT / "data/ptbxl_54features.csv"
MAN = ROOT / "experiments_phase3/phase3d_full_cohort/phase3d_eligible_cohort_15160_manifest.csv"
ZIP = Path("/Volumes/KIOXIA/Datasets/ptb-xl/ptb-xl-data.zip")
FEAT_50 = [f"{l}_{s}" for l in ("L1", "L2") for s in SUFFIXES]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    v1 = pd.read_csv(V1_15160)
    man = pd.read_csv(MAN).set_index("ecg_id")
    zf = zipfile.ZipFile(ZIP); names = set(zf.namelist())
    feats, dbg = {}, []
    t0 = time.time()
    for i, e in enumerate(v1["ecg_id"]):
        rel = man.loc[e, "filename_hr"] + ".dat"
        if rel not in names: rel = next(n for n in names if n.endswith(rel))
        sig = np.frombuffer(zf.read(rel), dtype="<i2").reshape(-1, 12) / 1000.0
        f, d = extract_feat50_v2(sig[:, 0], sig[:, 1], return_debug=True)
        assert all(np.isfinite(v) for v in f.values()), f"non-finite feature for ecg_id {e}"
        feats[e] = f
        dbg.append({"ecg_id": e, **{f"{l}_{k}": v for l in ("L1", "L2") for k, v in d[l].items()}})
        if (i + 1) % 2000 == 0: print(f"{i + 1} records, {time.time() - t0:.0f} s", flush=True)
    fv = pd.DataFrame.from_dict(feats, orient="index")[FEAT_50]
    v2 = v1.copy()
    v2[FEAT_50] = fv.loc[v1["ecg_id"]].to_numpy()
    out15 = OUT / "ptbxl_eligible_cohort_15160_features_v2.csv"; v2.to_csv(out15, index=False)

    s1 = pd.read_csv(V1_5000)
    assert s1["ecg_id"].isin(v1["ecg_id"]).all()
    s2 = s1.copy(); s2[FEAT_50] = fv.loc[s1["ecg_id"]].to_numpy()
    out5 = OUT / "ptbxl_54features_v2.csv"; s2.to_csv(out5, index=False)

    pd.DataFrame(dbg).to_csv(OUT / "extraction_v2_debug.csv", index=False)
    json.dump({"params": DEFAULT_PARAMS, "n_records": len(v2), "seconds": round(time.time() - t0, 1),
               "inputs": {str(p.relative_to(ROOT)): sha(p) for p in (V1_15160, V1_5000, MAN)},
               "outputs": {p.name: sha(p) for p in (out15, out5)},
               "unchanged_columns_identical_15160": bool((v2.drop(columns=FEAT_50) == v1.drop(columns=FEAT_50)).all().all())},
              open(OUT / "extraction_v2_manifest.json", "w"), indent=2)
    print("done", len(v2), "records")


if __name__ == "__main__":
    main()
