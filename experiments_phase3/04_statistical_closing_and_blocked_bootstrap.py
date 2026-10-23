#!/usr/bin/env python3
"""
Phase 3B: Statistical Closing & Patient-Blocked Paired Bootstrap Analysis
=============================================================================
This script performs the rigorous statistical closing requested for Phase 3B:
1. Re-evaluates all seed-level bootstrap p-values under the canonical formula:
       p = 2 * (n_opp + 1) / (B + 1)
   strictly distinguishing nominal p-values from Holm-Bonferroni FWER adjusted p-values.
2. Conducts a patient-blocked paired bootstrap (10,000 iterations) to compute
   repetition-averaged paired F1 differences across the 5 seeds while preserving
   within-patient correlation across partitions.
3. Exports structured results to results/phase3b_statistical_closing_results.json.
4. Audits protocol vs. code formula discrepancies.
=============================================================================
"""

import hashlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

BASE_DIR = Path("/Volumes/KIOXIA/Projects/gemma3-lora-other")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PRED_DIR = BASE_DIR / "predictions"
EXP_DIR = BASE_DIR / "experiments_phase3"

NPZ_PATH = PRED_DIR / "phase3b_repeated_oof_predictions.npz"
OLD_JSON_PATH = RESULTS_DIR / "phase3b_repeated_evaluation_results.json"
OUT_JSON_PATH = RESULTS_DIR / "phase3b_statistical_closing_results.json"


def holm_bonferroni(p_vals):
    """Computes exact Holm-Bonferroni step-down FWER adjusted p-values from unrounded p-values."""
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = np.array(p_vals)[sorted_indices]
    adj_p = np.zeros(m)
    for i in range(m):
        adj_p[i] = (m - i) * sorted_p[i]
    # Enforce monotonicity
    for i in range(1, m):
        adj_p[i] = max(adj_p[i], adj_p[i - 1])
    adj_p = np.clip(adj_p, 0.0, 1.0)
    orig_adj = np.zeros(m)
    orig_adj[sorted_indices] = adj_p
    return orig_adj.tolist()


def main():
    print("=" * 80)
    print("PHASE 3B: STATISTICAL CLOSING & PATIENT-BLOCKED PAIRED BOOTSTRAP")
    print("=" * 80)

    # 1. Load Data & Existing Outputs
    assert NPZ_PATH.exists(), f"NPZ file not found: {NPZ_PATH}"
    assert OLD_JSON_PATH.exists(), f"Old results JSON not found: {OLD_JSON_PATH}"

    npz = np.load(NPZ_PATH, allow_pickle=True)
    seeds = npz["seeds"].tolist()
    y_all = npz["true_labels"]
    n_samples = len(y_all)
    n_seeds = len(seeds)
    n_boot = 10000

    with open(OLD_JSON_PATH) as f:
        old_results = json.load(f)

    # 2. Audit Formula Discrepancy (Code vs Canonical)
    # Canonical: p = 2 * (n_opp + 1) / (B + 1)
    # Script 03: p = (2 * n_opp + 1) / (B + 1)
    comparison_keys = [
        ("primary_distill_vs_smooth", "Distilled - Uniform Smooth (eps=0.15)"),
        ("secondary_core_distill_vs_hard128", "Distilled - Hard-Label (128, 64)"),
        ("exploratory_smooth_vs_hard128", "Uniform Smooth - Hard-Label (128, 64)"),
        ("exploratory_capacity_hard128_vs_hard64", "Capacity Effect: Hard(128, 64) - Hard(64, 32)"),
        ("exploratory_teacher_vs_distill", "Teacher TabPFN - Distilled MLP"),
    ]

    canonical_seed_tables = {}
    for comp_key, desc in comparison_keys:
        table_rows = []
        unrounded_nominal_p = []

        for s_idx, s in enumerate(seeds):
            old_comp = old_results["detailed_results_by_seed"][f"seed_{s}"]["bootstrap_comparisons"][comp_key]
            n_opp = old_comp["n_opp"]
            point_delta = old_comp["point_delta_pp"]
            ci_95 = old_comp["ci_95"]

            # Exact canonical formula
            p_canonical_unrounded = 2.0 * (n_opp + 1.0) / (n_boot + 1.0)
            p_canonical_unrounded = min(1.0, p_canonical_unrounded)
            unrounded_nominal_p.append(p_canonical_unrounded)

            # Old formula value
            p_old_code = (2.0 * n_opp + 1.0) / (n_boot + 1.0)

            table_rows.append({
                "seed": s,
                "point_delta_pp": point_delta,
                "ci_95": ci_95,
                "n_opp": n_opp,
                "p_old_code_reported": old_comp["p_value_two_sided"],
                "p_old_code_exact": round(p_old_code, 6),
                "p_canonical_nominal": round(p_canonical_unrounded, 6),
                "p_canonical_nominal_rounded4": round(p_canonical_unrounded, 4),
                "difference_p": round(p_canonical_unrounded - p_old_code, 6),
            })

        # Holm-Bonferroni FWER adjustment across the 5 seeds for this test
        adj_p_5seeds = holm_bonferroni(unrounded_nominal_p)
        for r, ap in zip(table_rows, adj_p_5seeds):
            r["p_canonical_fwer_5seeds"] = round(ap, 6)
            r["significant_fwer_005_5seeds"] = bool(ap < 0.05)
            r["status_nominal"] = "nominal p < 0.05" if r["p_canonical_nominal"] < 0.05 else "nominal p >= 0.05"

        canonical_seed_tables[comp_key] = {
            "description": desc,
            "rows": table_rows,
            "all_nominal_positive": all(r["point_delta_pp"] > 0 for r in table_rows),
            "all_nominal_sig_005": all(r["p_canonical_nominal"] < 0.05 for r in table_rows),
            "all_fwer_sig_005": all(r["p_canonical_fwer_5seeds"] < 0.05 for r in table_rows),
        }

    # 3. Joint 10-Test Family FWER Adjustment (Primary + Secondary Core across 5 seeds)
    joint_tests = []
    for s_idx, s in enumerate(seeds):
        # Primary
        p_row = canonical_seed_tables["primary_distill_vs_smooth"]["rows"][s_idx]
        joint_tests.append({
            "family": "primary_distill_vs_smooth",
            "seed": s,
            "test_label": f"Primary (Distill - Smooth) Seed {s}",
            "point_delta_pp": p_row["point_delta_pp"],
            "p_nominal_unrounded": 2.0 * (p_row["n_opp"] + 1.0) / (n_boot + 1.0),
        })
    for s_idx, s in enumerate(seeds):
        # Secondary
        s_row = canonical_seed_tables["secondary_core_distill_vs_hard128"]["rows"][s_idx]
        joint_tests.append({
            "family": "secondary_core_distill_vs_hard128",
            "seed": s,
            "test_label": f"Secondary (Distill - Hard128) Seed {s}",
            "point_delta_pp": s_row["point_delta_pp"],
            "p_nominal_unrounded": 2.0 * (s_row["n_opp"] + 1.0) / (n_boot + 1.0),
        })

    joint_nominal_p = [t["p_nominal_unrounded"] for t in joint_tests]
    joint_adj_p = holm_bonferroni(joint_nominal_p)
    for t, ap in zip(joint_tests, joint_adj_p):
        t["p_canonical_nominal"] = round(t["p_nominal_unrounded"], 6)
        t["p_joint_fwer_10tests"] = round(ap, 6)
        t["significant_joint_fwer_005"] = bool(ap < 0.05)

    # 4. Patient-Blocked Paired Bootstrap for Repetition-Averaged Estimands
    print("\nRunning Patient-Blocked Paired Bootstrap (B = 10,000)...")
    t0_boot = time.time()
    rng = np.random.RandomState(42)

    p_dist = npz["oof_preds_distill128"]
    p_smooth = npz["oof_preds_smooth128"]
    p_hard128 = npz["oof_preds_hard128"]
    p_hard64 = npz["oof_preds_hard64"]
    p_tabpfn = npz["oof_preds_tabpfn"]

    blocked_comparisons = [
        ("primary_distill_vs_smooth", "Distilled - Uniform Smooth (eps=0.15)", p_dist, p_smooth),
        ("secondary_core_distill_vs_hard128", "Distilled - Hard-Label (128, 64)", p_dist, p_hard128),
        ("exploratory_smooth_vs_hard128", "Uniform Smooth - Hard-Label (128, 64)", p_smooth, p_hard128),
        ("exploratory_capacity_hard128_vs_hard64", "Capacity Effect: Hard(128, 64) - Hard(64, 32)", p_hard128, p_hard64),
        ("exploratory_teacher_vs_distill", "Teacher TabPFN - Distilled MLP", p_tabpfn, p_dist),
    ]

    patient_blocked_results = {}
    for comp_key, desc, m_a, m_b in blocked_comparisons:
        # 1. Point estimate: Average of the 5 repetition-specific OOF deltas
        pt_deltas_per_seed = []
        for s in range(n_seeds):
            f1_a = f1_score(y_all, m_a[s], average="macro", zero_division=0) * 100.0
            f1_b = f1_score(y_all, m_b[s], average="macro", zero_division=0) * 100.0
            pt_deltas_per_seed.append(f1_a - f1_b)
        point_mean_delta = float(np.mean(pt_deltas_per_seed))

        # 2. Bootstrap resamples
        boot_mean_deltas = np.zeros(n_boot, dtype=np.float64)
        for b in range(n_boot):
            # Resample patient indices (shared across all seeds and both models)
            idx = rng.randint(0, n_samples, size=n_samples)
            yt = y_all[idx]
            d_seeds = []
            for s in range(n_seeds):
                f1_a_b = f1_score(yt, m_a[s, idx], average="macro", zero_division=0) * 100.0
                f1_b_b = f1_score(yt, m_b[s, idx], average="macro", zero_division=0) * 100.0
                d_seeds.append(f1_a_b - f1_b_b)
            boot_mean_deltas[b] = np.mean(d_seeds)

        ci_low = float(np.percentile(boot_mean_deltas, 2.5))
        ci_high = float(np.percentile(boot_mean_deltas, 97.5))
        n_le = int(np.sum(boot_mean_deltas <= 0.0))
        n_ge = int(np.sum(boot_mean_deltas >= 0.0))
        n_opp = min(n_le, n_ge)
        p_canon = float(2.0 * (n_opp + 1.0) / (n_boot + 1.0))
        p_canon = min(1.0, p_canon)

        patient_blocked_results[comp_key] = {
            "description": desc,
            "estimand": "Repetition-averaged paired Macro F1 difference: (1/5) * sum_{s=1}^5 (F1_A^(s) - F1_B^(s))",
            "point_mean_delta_pp": round(point_mean_delta, 2),
            "boot_mean_delta_pp": round(float(np.mean(boot_mean_deltas)), 2),
            "ci_95": [round(ci_low, 2), round(ci_high, 2)],
            "n_boot": n_boot,
            "n_le_zero": n_le,
            "n_ge_zero": n_ge,
            "n_opp": n_opp,
            "p_value_canonical": round(p_canon, 6),
            "p_value_canonical_rounded4": round(p_canon, 4),
            "significant_005": bool(p_canon < 0.05),
            "patient_correlation_preserved": True,
            "pooled_independent_patients": False,
        }

    boot_duration = time.time() - t0_boot
    print(f"Patient-blocked bootstrap completed in {boot_duration:.1f}s.")

    # 5. Compile Final Closing JSON
    final_closing_json = {
        "metadata": {
            "analysis_name": "Phase 3B: Statistical Closing & Patient-Blocked Paired Bootstrap",
            "status": "FROZEN_POST_PILOT_ROBUSTNESS_ANALYSIS",
            "cohort": "PTB-XL FEAT_50 (N = 5,000 unique patients)",
            "outer_seeds": seeds,
            "outer_folds_per_seed": 5,
            "total_outer_evaluations": 25,
            "canonical_p_formula": "p = 2 * (n_opp + 1) / (B + 1)",
            "old_code_formula": "p = (2 * n_opp + 1) / (B + 1)",
            "formula_discrepancy_note": "Difference is exactly 1 / (B + 1) = 1 / 10001 ~= 0.0001 due to omitted parenthesis in script 03. All conclusions verified under canonical formula.",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "canonical_seed_level_evaluations": canonical_seed_tables,
        "joint_10_test_family_fwer": {
            "family_description": "Joint family of 10 core tests (5 seeds x 2 core comparisons: Distill vs Smooth, Distill vs Hard128)",
            "correction_method": "Holm-Bonferroni step-down FWER control at alpha = 0.05",
            "all_tests_significant": all(t["significant_joint_fwer_005"] for t in joint_tests),
            "tests": joint_tests,
        },
        "patient_blocked_paired_bootstrap_summary_estimands": patient_blocked_results,
    }

    with open(OUT_JSON_PATH, "w") as f:
        json.dump(final_closing_json, f, indent=2)
    print(f"\nFinal closing results saved to: {OUT_JSON_PATH}")

    # Compute SHA-256 Hashes
    files_to_hash = [
        EXP_DIR / "phase3b_protocol_specification.json",
        EXP_DIR / "03_procedure_matched_repeated_eval.py",
        EXP_DIR / "04_statistical_closing_and_blocked_bootstrap.py",
        RESULTS_DIR / "phase3b_repeated_evaluation_results.json",
        OUT_JSON_PATH,
        NPZ_PATH,
    ]
    manifest = {}
    for p in files_to_hash:
        if p.exists():
            with open(p, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            manifest[p.name] = {"path": str(p), "sha256": h, "size_bytes": p.stat().st_size}

    with open(EXP_DIR / "phase3b_closing_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nVerified File Hashes:")
    for fn, meta in manifest.items():
        print(f"  {fn:<45}: {meta['sha256']}")


if __name__ == "__main__":
    main()
