#!/usr/bin/env python3
"""WP2 (measurement only): QRS-window ceiling/fallback audit and agreement of FEAT_50 with PTB-XL+ reference features.

No feature is re-extracted and no model is trained. Uses the stored 15,160-patient feature table and the PTB-XL+
feature files (12SL, Uni-G, ECGDeli) for the same ecg_ids. Exploratory measurement analysis (../ANALIZ_PLANI_v22.md).

Outputs (revizyon_v22/results/wp2/):
  wp2_qrs_ceiling_fallback.csv   share of records at the 146 ms ceiling / 80 ms fallback, by lead and class
  wp2_qrs_vs_reference_bins.csv  our QRS duration versus reference QRS duration bins (where truncation would show)
  wp2_agreement.csv              Pearson/Spearman correlation, median difference, Bland-Altman limits per feature/lead/source
  wp2_run_manifest.json          inputs (SHA-256), definitions, counts
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revizyon_v22" / "results" / "wp2"
OUT.mkdir(parents=True, exist_ok=True)
FEAT = ROOT / "experiments_phase3/phase3d_full_cohort/ptbxl_eligible_cohort_15160_features.csv"
PLUS = {s: ROOT / f"data/ptbxl_plus/{s}_features.csv" for s in ("12sl", "unig", "ecgdeli")}
DESC = ROOT / "data/ptbxl_plus/feature_description.csv"
CEILING_MS, FALLBACK_MS = 146.0, 80.0
CLASSES = ["NORM", "AFIB/AFL", "CD", "STTC/MI", "HYP"]

# our feature suffix -> PTB-XL+ feature id (X = lead) ; global ids are used when no lead-specific value exists
PAIRS = {"qrs_dur": "QRS_Dur", "pr_int": "PR_Int", "qt_int": "QT_Int", "r_amp": "R_Amp", "s_amp": "S_Amp", "t_amp": "T_Amp"}
INTERVALS = {"qrs_dur", "pr_int", "qt_int"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    f = pd.read_csv(FEAT)
    assert len(f) == 15160 and f["ecg_id"].is_unique
    f["class"] = f["target_name"]
    desc = pd.read_csv(DESC, encoding="utf-8-sig").set_index("id")

    # ---- 1. ceiling / fallback audit
    rows = []
    for lead in ("L1", "L2"):
        q = f[f"{lead}_qrs_dur"]
        for cls in ["ALL"] + CLASSES:
            sel = f if cls == "ALL" else f[f["class"] == cls]
            qq = q.loc[sel.index]
            rows.append({"lead": lead, "class": cls, "n": len(sel),
                         "pct_at_or_above_ceiling_146ms": 100 * np.mean(qq >= CEILING_MS - 1e-9),
                         "pct_equal_fallback_80ms": 100 * np.mean(np.isclose(qq, FALLBACK_MS)),
                         "max_ms": float(qq.max()), "median_ms": float(qq.median())})
    pd.DataFrame(rows).to_csv(OUT / "wp2_qrs_ceiling_fallback.csv", index=False)

    # ---- 2. our QRS vs reference QRS duration bins (12SL global QRS duration as reference)
    s12 = pd.read_csv(PLUS["12sl"], usecols=["ecg_id", "QRS_Dur_Global"])
    m = f.merge(s12, on="ecg_id", how="left")
    bins = [0, 100, 120, 146, 160, 400]
    m["ref_bin"] = pd.cut(m["QRS_Dur_Global"], bins, right=False)
    rb = []
    for b, g in m.dropna(subset=["QRS_Dur_Global"]).groupby("ref_bin", observed=True):
        for lead in ("L1", "L2"):
            q = g[f"{lead}_qrs_dur"]
            rb.append({"reference_12SL_QRS_bin_ms": str(b), "lead": lead, "n": len(g),
                       "pct_CD_class": 100 * np.mean(g["class"] == "CD"),
                       "our_median_ms": float(q.median()), "our_pct_at_ceiling": 100 * np.mean(q >= CEILING_MS - 1e-9),
                       "our_pct_fallback": 100 * np.mean(np.isclose(q, FALLBACK_MS))})
    pd.DataFrame(rb).to_csv(OUT / "wp2_qrs_vs_reference_bins.csv", index=False)

    # ---- 3. agreement with PTB-XL+ (lead-specific where available, otherwise global)
    agr = []
    for src, path in PLUS.items():
        cols = pd.read_csv(path, nrows=0).columns
        want = ["ecg_id"]
        for ours, ref in PAIRS.items():
            for lead_ours, lead_ref in (("L1", "I"), ("L2", "II")):
                for cand in (f"{ref}_{lead_ref}", f"{ref}_Global"):
                    if cand in cols: want.append(cand); break
        ref = pd.read_csv(path, usecols=sorted(set(want)))
        mm = f.merge(ref, on="ecg_id", how="inner")
        for ours, refid in PAIRS.items():
            for lead_ours, lead_ref in (("L1", "I"), ("L2", "II")):
                cand = [c for c in (f"{refid}_{lead_ref}", f"{refid}_Global") if c in mm.columns]
                if not cand: continue
                c = cand[0]
                # The released PTB-XL+ CSVs are already in consistent units (amplitudes in mV, intervals in ms);
                # the 'consistent_units_*' factors in feature_description.csv refer to the raw algorithm outputs.
                scale = 1.0
                d = mm[[f"{lead_ours}_{ours}", c]].replace([np.inf, -np.inf], np.nan).dropna()
                a, r = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float) * scale
                if len(d) < 30: continue
                diff = a - r
                agr.append({"source": src, "feature": ours, "lead": lead_ours, "reference_column": c, "n": len(d),
                            "reference_scale_to_ours": scale, "pearson_r": stats.pearsonr(a, r)[0],
                            "spearman_rho": stats.spearmanr(a, r)[0], "median_diff_ours_minus_ref": float(np.median(diff)),
                            "median_abs_diff": float(np.median(np.abs(diff))),
                            "bland_altman_mean": float(diff.mean()), "loa_low": float(diff.mean() - 1.96 * diff.std(ddof=1)),
                            "loa_high": float(diff.mean() + 1.96 * diff.std(ddof=1))})
    pd.DataFrame(agr).to_csv(OUT / "wp2_agreement.csv", index=False)

    json.dump({"inputs": {str(p.relative_to(ROOT)): sha(p) for p in [FEAT, DESC, *PLUS.values()]},
               "ceiling_ms": CEILING_MS, "fallback_ms": FALLBACK_MS,
               "ceiling_definition": "derivative search window indices 68-141 around R at index 100 (500 Hz): (141-68)*2 ms = 146 ms",
               "n_records": int(len(f)), "n_matched_12sl_qrs": int(m["QRS_Dur_Global"].notna().sum())},
              open(OUT / "wp2_run_manifest.json", "w"), indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
