#!/usr/bin/env python3
"""Extract FEAT_50 v2 for the eligible Chapman-Shaoxing cohort (adults with a single superclass label).

Label rules are those of the archived v1 cohort builder (`01_extract_features_all_cohorts.py::build_chapman_cohort`),
re-implemented here unchanged because the mapping is a nested function there. The v1 5,000-record quota sample is
flagged (`in_quota_5000`) and its labels are checked against the v1 table. The GE MUSE QRS duration and QT interval
from Diagnostics.csv are kept for an external measurement-agreement check (the v2 extractor was not tuned on Chapman).
Outputs (revizyon_v22/data_v2/):
  chapman_eligible_features_v2.csv, chapman_extraction_v2_manifest.json
"""
from __future__ import annotations

import hashlib
import io
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

DIAG = ROOT / "data/chapman/Diagnostics.csv"
ZIP = ROOT / "data/chapman/ECGData.zip"
V1_QUOTA = ROOT / "data/chapman_54features.csv"
OUT = ROOT / "revizyon_v22/data_v2"
FEAT_50 = [f"{l}_{s}" for l in ("L1", "L2") for s in SUFFIXES]
CLASS_MAP = {"NORM": 0, "AFIB/AFL": 1, "CD": 2, "STTC/MI": 3, "HYP": 4}

CD_CODES = {"1AVB", "2AVB", "2AVB1", "2AVB2", "3AVB", "AVB", "LBBB", "LBBBB", "LFBBB", "RBBB", "IRBBB", "IVB", "IDC",
            "PRIE", "WPW", "VPE"}
STTC_MI_CODES = {"MI", "MIBW", "MIFW", "MILW", "MISW", "STDD", "STE", "STTC", "STTU", "TWC", "TWO", "AQW", "QTIE"}
HYP_CODES = {"LVH", "LVHV", "RVH", "RAH", "RAHV"}


def map_chapman(row) -> str:
    """v1 superclass rule (unchanged)."""
    rhy = str(row["Rhythm"]).strip()
    beats = set(str(row["Beat"]).strip().split())
    beats.discard("NONE"); beats.discard("nan")
    if rhy in ["AFIB", "AF"]:
        return "AFIB/AFL"
    has_cd, has_sttc, has_hyp = bool(beats & CD_CODES), bool(beats & STTC_MI_CODES), bool(beats & HYP_CODES)
    if rhy in ["SR", "SB", "ST", "SI", "SA"] and len(beats) == 0:
        return "NORM"
    if has_cd and not has_hyp:
        return "CD"
    if has_hyp and not has_cd:
        return "HYP"
    if has_sttc and not has_cd and not has_hyp:
        return "STTC/MI"
    return "MULTI/OTHER"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    diag = pd.read_csv(DIAG)
    assert diag["FileName"].is_unique, "Chapman FileName (one ECG per patient) is not unique"
    diag = diag[diag["PatientAge"] >= 18].copy()
    diag["target_name"] = diag.apply(map_chapman, axis=1)
    elig = diag[diag["target_name"] != "MULTI/OTHER"].reset_index(drop=True)

    v1 = pd.read_csv(V1_QUOTA).set_index("record_id")
    assert v1.index.isin(elig["FileName"]).all(), "v1 quota records missing from the eligible cohort"
    lab_ok = (elig.set_index("FileName").loc[v1.index, "target_name"] == v1["target_name"]).all()
    assert lab_ok, "label rule re-implementation disagrees with the v1 quota table"

    zf = zipfile.ZipFile(ZIP)
    rows, t0 = [], time.time()
    for i, r in elig.iterrows():
        sig = pd.read_csv(io.BytesIO(zf.read(f"ECGData/{r['FileName']}.csv")), usecols=[0, 1]).to_numpy(np.float64) / 1000.0
        f, d = extract_feat50_v2(sig[:, 0], sig[:, 1], return_debug=True)
        assert all(np.isfinite(v) for v in f.values()), f"non-finite feature for {r['FileName']}"
        rows.append({"record_id": r["FileName"], "patient_id": f"CHAP_{r['FileName']}", "target_name": r["target_name"],
                     "target_class": CLASS_MAP[r["target_name"]], "age": float(r["PatientAge"]),
                     "sex_male": int(str(r["Gender"]).strip().upper().startswith("M")),
                     "in_quota_5000": int(r["FileName"] in v1.index),
                     "muse_qrs_ms": r["QRSDuration"], "muse_qt_ms": r["QTInterval"],
                     "L1_qrs_fallback": d["L1"]["qrs_fallback"], "L2_qrs_fallback": d["L2"]["qrs_fallback"],
                     "L1_p_fallback": d["L1"]["p_fallback"], "L2_p_fallback": d["L2"]["p_fallback"], **f})
        if (i + 1) % 2000 == 0: print(f"{i + 1}/{len(elig)} records, {time.time() - t0:.0f} s", flush=True)
    out = pd.DataFrame(rows)
    p = OUT / "chapman_eligible_features_v2.csv"; out.to_csv(p, index=False)
    json.dump({"params": DEFAULT_PARAMS, "n_records": len(out), "n_quota_5000": int(out["in_quota_5000"].sum()),
               "class_counts": out["target_name"].value_counts().to_dict(),
               "class_counts_quota_5000": out.loc[out.in_quota_5000 == 1, "target_name"].value_counts().to_dict(),
               "v1_label_rule_reproduced_on_quota_5000": bool(lab_ok), "seconds": round(time.time() - t0, 1),
               "inputs": {str(q.relative_to(ROOT)): sha(q) for q in (DIAG, ZIP, V1_QUOTA)}, "output_sha256": sha(p)},
              open(OUT / "chapman_extraction_v2_manifest.json", "w"), indent=2)
    print("done", len(out), "records")


if __name__ == "__main__":
    main()
