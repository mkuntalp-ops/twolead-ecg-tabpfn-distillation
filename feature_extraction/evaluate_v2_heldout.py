#!/usr/bin/env python3
"""Evaluate FEAT_50 v2 against the pre-specified acceptance criteria on the 14,160 records not used for tuning.

Criteria (ANALIZ_PLANI_v22.md): PR-reference Pearson r >= 0.6 (non-AF records); QRS-Uni-G r not lower than v1;
no record at the 146 ms ceiling; QT-Uni-G r not lower than v1 and smaller median difference; amplitude agreement
(R, S, T) preserved (r change <= 0.02). Output: revizyon_v22/results/feat50_v2/heldout_evaluation.csv (+ .md table)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revizyon_v22/results/feat50_v2"
dev_ids = set(json.load(open(OUT / "dev_manifest.json"))["dev_ecg_ids"])
v1 = pd.read_csv(ROOT / "experiments_phase3/phase3d_full_cohort/ptbxl_eligible_cohort_15160_features.csv").set_index("ecg_id")
v2 = pd.read_csv(ROOT / "revizyon_v22/data_v2/ptbxl_eligible_cohort_15160_features_v2.csv").set_index("ecg_id")
held = [e for e in v1.index if e not in dev_ids]
assert len(held) == 14160
u = pd.read_csv(ROOT / "data/ptbxl_plus/unig_features.csv").set_index("ecg_id")
s = pd.read_csv(ROOT / "data/ptbxl_plus/12sl_features.csv").set_index("ecg_id")
nonaf = v1.loc[held, "target_name"] != "AFIB/AFL"

checks = [  # (label, our column, reference series, restrict to non-AF)
    ("PR L1 vs 12SL global", "L1_pr_int", s["PR_Int_Global"], True), ("PR L2 vs 12SL global", "L2_pr_int", s["PR_Int_Global"], True),
    ("PR L1 vs Uni-G global", "L1_pr_int", u["PR_Int_Global"], True), ("PR L2 vs Uni-G global", "L2_pr_int", u["PR_Int_Global"], True),
    ("QRS L1 vs Uni-G I", "L1_qrs_dur", u["QRS_Dur_I"], False), ("QRS L2 vs Uni-G II", "L2_qrs_dur", u["QRS_Dur_II"], False),
    ("QRS L1 vs 12SL global", "L1_qrs_dur", s["QRS_Dur_Global"], False), ("QRS L2 vs 12SL global", "L2_qrs_dur", s["QRS_Dur_Global"], False),
    ("QT L1 vs Uni-G I", "L1_qt_int", u["QT_Int_I"], False), ("QT L2 vs Uni-G II", "L2_qt_int", u["QT_Int_II"], False),
    ("R amp L1 vs 12SL", "L1_r_amp", s["R_Amp_I"], False), ("R amp L2 vs 12SL", "L2_r_amp", s["R_Amp_II"], False),
    ("S amp L1 vs 12SL", "L1_s_amp", s["S_Amp_I"], False), ("S amp L2 vs 12SL", "L2_s_amp", s["S_Amp_II"], False),
    ("T amp L1 vs 12SL", "L1_t_amp", s["T_Amp_I"], False), ("T amp L2 vs 12SL", "L2_t_amp", s["T_Amp_II"], False)]
rows = []
for label, col, ref, af_ex in checks:
    ids = [e for e, ok in zip(held, nonaf) if ok] if af_ex else held
    rec = {"comparison": label, "records": "non-AF" if af_ex else "all"}
    for name, f in (("v1", v1), ("v2", v2)):
        d = pd.DataFrame({"a": f.loc[ids, col], "b": ref.reindex(ids)}).dropna()
        rec[f"n_{name}"] = len(d); rec[f"r_{name}"] = stats.pearsonr(d.a, d.b)[0]
        rec[f"median_diff_{name}"] = float(np.median(d.a - d.b))
    rec["delta_r"] = rec["r_v2"] - rec["r_v1"]
    rows.append(rec)
res = pd.DataFrame(rows)
ceil = {f"pct_{L}_qrs_at_146_{n}": 100 * float(np.isclose(f.loc[held, f"{L}_qrs_dur"], 146.0).mean())
        for n, f in (("v1", v1), ("v2", v2)) for L in ("L1", "L2")}
pr_sd = {f"sd_L2_pr_{n}": float(f.loc[held, "L2_pr_int"].std()) for n, f in (("v1", v1), ("v2", v2))}
res.to_csv(OUT / "heldout_evaluation.csv", index=False)
json.dump({**ceil, **pr_sd}, open(OUT / "heldout_extra.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False)); print(json.dumps({**ceil, **pr_sd}, indent=1))
