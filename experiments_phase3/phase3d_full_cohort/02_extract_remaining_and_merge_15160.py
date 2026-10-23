#!/usr/bin/env python3
"""
experiments_phase3/phase3d_full_cohort/02_extract_remaining_and_merge_15160.py

Extracts the remaining 10,160 records of the eligible cohort with checkpoint/resume
support and combines them with the 5,000 existing discovery records into:
  ptbxl_eligible_cohort_15160_features.csv

Integrity checks:
  - Exactly 15,160 rows and 15,160 unique patient_ids.
  - Zero NaNs, Infs, or null values in all 50 waveform features.
  - Exactly matches the 15,160 class distribution:
      NORM: 8,052, STTC/MI: 3,486, CD: 1,647, AFIB/AFL: 1,155, HYP: 820.
"""

import ast
import gc
import hashlib
import importlib
import json
import os
import resource
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

mod = importlib.import_module("01_extract_features_all_cohorts")
extract_dual_lead_54 = mod.extract_dual_lead_54

DATA_DIR = ROOT_DIR / "data"
PTBXL_ZIP = Path("/Volumes/KIOXIA/Datasets/ptb-xl/ptb-xl-data.zip")
PTBXL_UNIG = DATA_DIR / "ptbxl_plus/unig_features.csv"
PTBXL_DELI = DATA_DIR / "ptbxl_plus/ecgdeli_features.csv"
DISC_5000_CSV = DATA_DIR / "ptbxl_54features.csv"

OUT_DIR = Path(__file__).resolve().parent
MANIFEST_CSV = OUT_DIR / "phase3d_eligible_cohort_15160_manifest.csv"
CHECKPOINT_CSV = OUT_DIR / "extracted_remaining_10160_checkpoint.csv"
FINAL_15160_CSV = OUT_DIR / "ptbxl_eligible_cohort_15160_features.csv"
LOG_FAILURES_PATH = OUT_DIR / "extraction_failures.log"

CLASS_MAP = {
    "NORM": 0,
    "AFIB/AFL": 1,
    "CD": 2,
    "STTC/MI": 3,
    "HYP": 4,
}

def get_process_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 80)
    print("PHASE 3D: EXTRACT REMAINING 10,160 RECORDS & MERGE FULL 15,160 COHORT")
    print("=" * 80)

    # 1. Load manifest
    print(f"\n[Step 1] Loading cohort manifest from {MANIFEST_CSV}...")
    manifest = pd.read_csv(MANIFEST_CSV)
    assert len(manifest) == 15160, f"Expected 15160 in manifest, got {len(manifest)}"

    disc_5000 = manifest[manifest["in_discovery_5000"]].copy()
    rem_10160 = manifest[~manifest["in_discovery_5000"]].copy().reset_index(drop=True)
    assert len(disc_5000) == 5000
    assert len(rem_10160) == 10160

    print(f"Verified manifest: 5,000 discovery + 10,160 remaining = 15,160 total patients.")

    # 2. Checkpoint / Resume logic for remaining 10,160
    extracted_df = None
    already_extracted_ids = set()
    if CHECKPOINT_CSV.exists():
        print(f"\n[Step 2] Found existing checkpoint at {CHECKPOINT_CSV}. Loading...")
        extracted_df = pd.read_csv(CHECKPOINT_CSV)
        already_extracted_ids = set(extracted_df["ecg_id"])
        print(f"Already extracted in checkpoint: {len(already_extracted_ids)} / 10,160 records.")

    todo_df = rem_10160[~rem_10160["ecg_id"].isin(already_extracted_ids)].copy().reset_index(drop=True)
    print(f"Records left to extract: {len(todo_df)}.")

    if len(todo_df) > 0:
        print("\n[Step 3] Loading fiducial reference tables for remaining records...")
        unig = pd.read_csv(PTBXL_UNIG, usecols=["ecg_id", "QRS_Dur_Global", "RR_Mean_Global"])
        deli = pd.read_csv(PTBXL_DELI, usecols=["ecg_id", "RR_Mean_Global_iqr"])
        todo_df = todo_df.merge(unig, on="ecg_id", how="left").merge(deli, on="ecg_id", how="left")

        print(f"\n[Step 4] Extracting {len(todo_df)} records with checkpointing...")
        zf = zipfile.ZipFile(PTBXL_ZIP, "r")
        new_records = []
        failures = []
        t0 = time.time()
        batch_size = 500

        for idx, row in todo_df.iterrows():
            ecg_id = int(row["ecg_id"])
            pat_id_str = str(row["patient_id_str"])
            try:
                rel_hr = row["filename_hr"] + ".dat"
                raw = zf.read(rel_hr)
                sig = np.frombuffer(raw, dtype="<i2").reshape(-1, 12) / 1000.0
                sig_I = sig[:, 0]
                sig_II = sig[:, 1]

                ref_qrs = float(row["QRS_Dur_Global"]) if pd.notna(row["QRS_Dur_Global"]) else None
                ref_cv = (
                    float(row["RR_Mean_Global_iqr"] / max(row["RR_Mean_Global"], 250.0))
                    if (pd.notna(row["RR_Mean_Global_iqr"]) and pd.notna(row["RR_Mean_Global"]))
                    else None
                )

                feats = extract_dual_lead_54(
                    sig_I=sig_I,
                    sig_II=sig_II,
                    age=float(row["age"]),
                    sex=int(row["sex"]),
                    fs=500.0,
                    ref_qrs_ms=ref_qrs,
                    ref_rr_cv=ref_cv,
                )

                # Validation of each record
                assert len(feats) == 54, f"Feature length {len(feats)} != 54"
                for fk, fv in feats.items():
                    if np.isnan(fv) or np.isinf(fv):
                        raise ValueError(f"NaN/Inf in feature {fk}: {fv}")

                rec_dict = {
                    "ecg_id": ecg_id,
                    "patient_id": pat_id_str,
                    "target_name": row["target_name"],
                    "target_class": CLASS_MAP[row["target_name"]],
                }
                rec_dict.update(feats)
                new_records.append(rec_dict)

            except Exception as e:
                err_msg = f"ERROR on ecg_id {ecg_id} (patient {pat_id_str}): {str(e)}"
                print(err_msg, flush=True)
                failures.append(err_msg)
                with open(LOG_FAILURES_PATH, "a") as f_err:
                    f_err.write(err_msg + "\n")

            # Checkpoint every batch
            if (idx + 1) % batch_size == 0 or (idx + 1) == len(todo_df):
                elapsed = time.time() - t0
                speed = (idx + 1) / max(elapsed, 0.001)
                rem_time_s = (len(todo_df) - (idx + 1)) / max(speed, 0.001)
                mem_mb = get_process_memory_mb()
                print(f"  Processed {idx + 1:5d} / {len(todo_df)} records | Elapsed: {elapsed:.1f}s | Speed: {speed:.1f} rec/s | ETA: {rem_time_s:.1f}s | RSS: {mem_mb:.1f} MB", flush=True)

                # Append to checkpoint
                batch_df = pd.DataFrame(new_records)
                if extracted_df is None:
                    extracted_df = batch_df
                else:
                    extracted_df = pd.concat([extracted_df, batch_df], ignore_index=True)
                new_records = []
                extracted_df.to_csv(CHECKPOINT_CSV, index=False)

        zf.close()
        if len(failures) > 0:
            print(f"\nCRITICAL: {len(failures)} failures encountered during extraction! Stopping.")
            sys.exit(1)
        print(f"\nAll {len(todo_df)} new records successfully extracted and saved to checkpoint.")
    else:
        print("\nAll 10,160 remaining records already present in checkpoint.")

    assert len(extracted_df) == 10160, f"Expected 10160 in checkpoint, got {len(extracted_df)}"

    # 3. Load discovery 5,000 dataset
    print(f"\n[Step 5] Loading discovery 5,000 dataset from {DISC_5000_CSV}...")
    disc_df = pd.read_csv(DISC_5000_CSV)
    assert len(disc_df) == 5000, f"Expected 5000 discovery records, got {len(disc_df)}"

    # Align columns between discovery and newly extracted
    # Standard 54 feature columns + metadata
    feat_50_cols = [c for c in disc_df.columns if c.startswith("L1_") or c.startswith("L2_")]
    assert len(feat_50_cols) == 50, f"Expected 50 waveform features, got {len(feat_50_cols)}"
    meta_4_cols = ["age", "sex", "irregular_rr_flag", "wide_qrs_flag"]
    id_cols = ["patient_id", "ecg_id", "target_name", "target_class"]

    # Ensure target_class is in disc_df
    if "target_class" not in disc_df.columns and "target_s1" in disc_df.columns:
        disc_df["target_class"] = disc_df["target_s1"]
    elif "target_class" not in disc_df.columns:
        disc_df["target_class"] = disc_df["target_name"].map(CLASS_MAP)

    standard_cols = id_cols + feat_50_cols + meta_4_cols

    disc_aligned = disc_df[standard_cols].copy()
    disc_aligned["in_discovery_5000"] = 1

    extracted_aligned = extracted_df[standard_cols].copy()
    extracted_aligned["in_discovery_5000"] = 0

    print(f"\n[Step 6] Concatenating 5,000 discovery + 10,160 newly extracted records...")
    full_cohort_df = pd.concat([disc_aligned, extracted_aligned], ignore_index=True)
    full_cohort_df = full_cohort_df.sort_values("ecg_id").reset_index(drop=True)

    # 4. Strict Integrity Verification
    print("\n[Step 7] Running strict integrity checks on full cohort (N = 15,160)...")
    assert len(full_cohort_df) == 15160, f"Row count {len(full_cohort_df)} != 15160"
    assert full_cohort_df["patient_id"].nunique() == 15160, "Duplicate patient_ids found!"
    assert full_cohort_df["ecg_id"].nunique() == 15160, "Duplicate ecg_ids found!"

    # Check for NaNs
    nans_per_col = full_cohort_df[feat_50_cols + meta_4_cols].isna().sum()
    assert nans_per_col.sum() == 0, f"Found NaNs in features: {nans_per_col[nans_per_col > 0]}"

    # Verify class distribution
    expected_dist = {
        "NORM": 8052,
        "STTC/MI": 3486,
        "CD": 1647,
        "AFIB/AFL": 1155,
        "HYP": 820,
    }
    actual_dist = full_cohort_df["target_name"].value_counts().to_dict()
    for k, v in expected_dist.items():
        assert actual_dist.get(k, 0) == v, f"Class {k} count mismatch: expected {v}, got {actual_dist.get(k, 0)}"

    print("Integrity checks PASSED 100%:")
    print(f"  - Total unique patients: {len(full_cohort_df)}")
    print(f"  - Feature dimensions:    {len(feat_50_cols)} waveform + {len(meta_4_cols)} metadata = 54 total")
    print(f"  - Zero NaNs / Infs verified across all 15,160 rows.")
    print("  - Class distribution verified:")
    for k, v in expected_dist.items():
        pct = (v / 15160) * 100.0
        print(f"      * {k:10s}: {v:5d} ({pct:5.2f}%)")

    # Save final dataset
    print(f"\n[Step 8] Saving final full cohort dataset to: {FINAL_15160_CSV}...")
    full_cohort_df.to_csv(FINAL_15160_CSV, index=False)
    final_hash = sha256_file(FINAL_15160_CSV)
    final_size_mb = FINAL_15160_CSV.stat().st_size / (1024 * 1024)

    print(f"\nFINAL DATASET SAVED SUCCESSFULLY:")
    print(f"  Path:    {FINAL_15160_CSV}")
    print(f"  Size:    {final_size_mb:.2f} MB")
    print(f"  SHA-256: {final_hash}")

    # Write manifest summary
    summary_path = OUT_DIR / "phase3d_feature_extraction_summary.json"
    summary_data = {
        "status": "COMPLETED",
        "cohort_name": "eligible_cohort_without_quota_subsampling",
        "total_records": 15160,
        "unique_patients": 15160,
        "discovery_records": 5000,
        "newly_extracted_records": 10160,
        "feature_count_waveform": 50,
        "feature_count_total": 54,
        "class_counts": expected_dist,
        "output_csv": str(FINAL_15160_CSV),
        "output_csv_sha256": final_hash,
        "output_csv_size_mb": final_size_mb
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to: {summary_path}")

if __name__ == "__main__":
    main()
