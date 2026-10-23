#!/usr/bin/env python3
"""
experiments_phase3/phase3d_full_cohort/01_benchmark_extraction_100.py

Runs the initial 100-record extraction benchmark across classes on the remaining
10,160 unextracted adult patients of the eligible cohort (N = 15,160).

Measures:
  - File read / decompression time
  - Feature computation time
  - Total wall-clock time
  - Number of failed records
  - Peak memory / RSS before and after

Fails safely if any record encounters an error.
"""

import ast
import gc
import os
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import resource

# Add project root for canonical extraction module
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import importlib
mod = importlib.import_module("01_extract_features_all_cohorts")
extract_dual_lead_54 = mod.extract_dual_lead_54

DATA_DIR = ROOT_DIR / "data"
PTBXL_CSV = Path("/Volumes/KIOXIA/Datasets/ptb-xl/ptbxl_database.csv")
PTBXL_SCP = Path("/Volumes/KIOXIA/Datasets/ptb-xl/scp_statements.csv")
PTBXL_ZIP = Path("/Volumes/KIOXIA/Datasets/ptb-xl/ptb-xl-data.zip")
PTBXL_UNIG = DATA_DIR / "ptbxl_plus/unig_features.csv"
PTBXL_DELI = DATA_DIR / "ptbxl_plus/ecgdeli_features.csv"
DISC_5000_CSV = DATA_DIR / "ptbxl_54features.csv"

OUT_DIR = Path(__file__).resolve().parent

def get_process_memory_mb():
    # On macOS, ru_maxrss is in bytes
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_bytes / (1024 * 1024)

def main():
    print("=" * 80)
    print("PHASE 3D: COHORT FREEZE & 100-RECORD EXTRACTION BENCHMARK")
    print("=" * 80)

    # 1. Load database and scp statements
    print("\n[Step 1] Loading PTB-XL database and building eligible cohort pool (N = 15,160)...")
    df = pd.read_csv(PTBXL_CSV)
    scp = pd.read_csv(PTBXL_SCP, index_col=0)
    df = df[df["age"] >= 18].copy()
    df["age"] = df["age"].apply(lambda a: 90.0 if a > 110 else float(a))

    diag_map = scp[scp["diagnostic"] == 1.0]["diagnostic_class"].to_dict()

    def assign_class(scp_str):
        codes = ast.literal_eval(scp_str)
        has_af = ("AFIB" in codes) or ("AFLT" in codes)
        superclasses = set(diag_map[k] for k, v in codes.items() if k in diag_map and v >= 50)
        if has_af:
            return "AFIB/AFL"
        if len(superclasses) == 1:
            sc = list(superclasses)[0]
            if sc == "NORM": return "NORM"
            if sc == "CD": return "CD"
            if sc in ["STTC", "MI"]: return "STTC/MI"
            if sc == "HYP": return "HYP"
        elif superclasses == {"STTC", "MI"}:
            return "STTC/MI"
        elif "HYP" in superclasses and superclasses.issubset({"HYP", "STTC"}):
            return "HYP"
        return "MULTI/OTHER"

    df["target_name"] = df["scp_codes"].apply(assign_class)
    df_clean = df[df["target_name"] != "MULTI/OTHER"].copy()

    # Enforce longitudinal single-class consistency
    pat_classes = df_clean.groupby("patient_id")["target_name"].nunique()
    single_pats = pat_classes[pat_classes == 1].index
    df_clean = df_clean[df_clean["patient_id"].isin(single_pats)].copy()

    # Deduplicate to first record per unique patient
    df_first = df_clean.sort_values("ecg_id").drop_duplicates(subset=["patient_id"], keep="first").copy()
    df_first["patient_id_str"] = df_first["patient_id"].apply(lambda p: f"PTB_{int(p)}")

    assert len(df_first) == 15160, f"Expected 15160, got {len(df_first)}"
    print(f"Eligible pool verified: EXACTLY {len(df_first)} unique adult patients.")
    print("Class distribution in full eligible pool:")
    for cls_name, count in df_first["target_name"].value_counts().items():
        print(f"  - {cls_name:10s}: {count:5d} ({count / len(df_first) * 100:.2f}%)")

    # Verify discovery 5,000
    disc = pd.read_csv(DISC_5000_CSV)
    disc_pats = set(disc["patient_id"])
    assert len(disc_pats) == 5000, f"Expected 5000 unique patients in discovery, got {len(disc_pats)}"
    assert disc_pats.issubset(set(df_first["patient_id_str"])), "Discovery patients not subset of eligible pool!"
    print("\n[Step 2] 5,000 discovery patients verified as 100% subset of eligible pool.")

    df_first["in_discovery_5000"] = df_first["patient_id_str"].isin(disc_pats)
    df_remaining = df_first[~df_first["in_discovery_5000"]].copy().reset_index(drop=True)
    assert len(df_remaining) == 10160, f"Expected 10160 remaining, got {len(df_remaining)}"
    print(f"Remaining unextracted patients: EXACTLY {len(df_remaining)}.")
    print("Class distribution of remaining 10,160 patients:")
    for cls_name, count in df_remaining["target_name"].value_counts().items():
        print(f"  - {cls_name:10s}: {count:5d} ({count / len(df_remaining) * 100:.2f}%)")

    # Save the frozen cohort manifest
    manifest_csv = OUT_DIR / "phase3d_eligible_cohort_15160_manifest.csv"
    save_cols = ["patient_id", "patient_id_str", "ecg_id", "age", "sex", "target_name", "in_discovery_5000", "filename_hr", "filename_lr"]
    df_first[save_cols].to_csv(manifest_csv, index=False)
    print(f"\nSaved frozen cohort manifest to: {manifest_csv}")

    # Load UNIG and DELI metadata
    print("\n[Step 3] Loading fiducial reference metadata (UNIG, DELI)...")
    unig = pd.read_csv(PTBXL_UNIG, usecols=["ecg_id", "QRS_Dur_Global", "RR_Mean_Global"])
    deli = pd.read_csv(PTBXL_DELI, usecols=["ecg_id", "RR_Mean_Global_iqr"])
    df_remaining = df_remaining.merge(unig, on="ecg_id", how="left").merge(deli, on="ecg_id", how="left")

    # Select balanced 100 records for the benchmark:
    # Classes available in remaining: NORM (25), STTC/MI (25), CD (25), AFIB/AFL (25)
    print("\n[Step 4] Selecting 100 benchmark records distributed across classes (25 per class)...")
    bench_records = []
    for cls_name in ["NORM", "STTC/MI", "CD", "AFIB/AFL"]:
        sub = df_remaining[df_remaining["target_name"] == cls_name]
        bench_records.append(sub.sample(n=25, random_state=42))
    df_bench = pd.concat(bench_records, ignore_index=True)
    assert len(df_bench) == 100

    # 5. Execute benchmark measurement
    print("\n[Step 5] Running 100-record extraction benchmark...")
    mem_before_mb = get_process_memory_mb()
    read_times = []
    calc_times = []
    failed_records = []

    zf = zipfile.ZipFile(PTBXL_ZIP, "r")
    t_start_total = time.perf_counter()

    for idx, row in df_bench.iterrows():
        try:
            # Measure read/decompression time
            t_r0 = time.perf_counter()
            rel_hr = row["filename_hr"] + ".dat"
            raw = zf.read(rel_hr)
            sig = np.frombuffer(raw, dtype="<i2").reshape(-1, 12) / 1000.0
            sig_I = sig[:, 0]
            sig_II = sig[:, 1]
            t_r1 = time.perf_counter()
            read_times.append(t_r1 - t_r0)

            # Measure calculation time
            t_c0 = time.perf_counter()
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
            # Verify feature output integrity
            assert len(feats) == 54, f"Expected 54 features, got {len(feats)}"
            for k, v in feats.items():
                assert not np.isnan(v) and not np.isinf(v), f"NaN/Inf in feature {k} for ecg_id {row['ecg_id']}"
            t_c1 = time.perf_counter()
            calc_times.append(t_c1 - t_c0)

        except Exception as e:
            failed_records.append({"ecg_id": row["ecg_id"], "patient_id": row["patient_id_str"], "error": str(e)})

    zf.close()
    t_end_total = time.perf_counter()
    total_wall_time = t_end_total - t_start_total
    mem_after_mb = get_process_memory_mb()

    # Compile benchmark metrics
    avg_read_ms = np.mean(read_times) * 1000.0
    avg_calc_ms = np.mean(calc_times) * 1000.0
    total_calc_per_rec_ms = (total_wall_time / 100.0) * 1000.0
    projected_10k_single_thread_min = (total_wall_time / 100.0 * 10160) / 60.0
    projected_10k_8workers_min = projected_10k_single_thread_min / 6.0  # conservative 6x speedup

    bench_results = {
        "n_records": 100,
        "n_failed": len(failed_records),
        "total_wall_time_sec": total_wall_time,
        "avg_read_decompression_ms": avg_read_ms,
        "avg_feature_calculation_ms": avg_calc_ms,
        "total_time_per_record_ms": total_calc_per_rec_ms,
        "process_memory_before_mb": mem_before_mb,
        "process_memory_after_mb": mem_after_mb,
        "memory_delta_mb": mem_after_mb - mem_before_mb,
        "projected_10160_single_core_minutes": projected_10k_single_thread_min,
        "projected_10160_multiprocessing_minutes": projected_10k_8workers_min,
        "failed_records": failed_records
    }

    import json
    bench_json_path = OUT_DIR / "phase3d_extraction_100_benchmark.json"
    with open(bench_json_path, "w") as f:
        json.dump(bench_results, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (100 RECORDS):")
    print("=" * 80)
    print(f"Total Wall-Clock Time:      {total_wall_time:.2f} seconds")
    print(f"Average Read/Decomp Time:   {avg_read_ms:.3f} ms / record")
    print(f"Average Calc Time:          {avg_calc_ms:.3f} ms / record")
    print(f"Total Time per Record:      {total_calc_per_rec_ms:.2f} ms")
    print(f"Failed Records:             {len(failed_records)} / 100")
    print(f"Process Memory (RSS):       {mem_before_mb:.1f} MB -> {mem_after_mb:.1f} MB (Delta: {mem_after_mb - mem_before_mb:+.2f} MB)")
    print(f"Projected 10,160 Runtime:   ~{projected_10k_8workers_min:.1f} minutes with 6-8 worker processes")
    print(f"Results saved to:           {bench_json_path}")

    if len(failed_records) > 0:
        print("\nCRITICAL WARNING: Failures detected in benchmark!")
        for fr in failed_records:
            print("  ", fr)
        sys.exit(1)
    else:
        print("\nALL 100 RECORDS EXTRACTED WITH 100% SUCCESS AND ZERO NANS/INFS.")

if __name__ == "__main__":
    main()
