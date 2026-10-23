#!/usr/bin/env python3
"""
06_generate_publication_figures.py

Generates Q1 publication-quality 300 DPI figures aligned with the FEAT_50 (Cross-Cohort Primary)
and Strict-54 (Verified Adult Primary) analysis hierarchy:
  - figure2_macro_f1_comparison.png: Task 1 Multi-Class Macro F1 across PTB-XL (N=5,000 patients),
    Chapman-Shaoxing (N=5,000 patients), and CPSC-2018 (N=1,600 records) under the Primary 50-Feature
    Waveform-Derived Schema (FEAT_50, zero demographic imputation).
  - figure3_ci_overlap_task1.png: 10,000-Iteration Bootstrap 95% CI Forest Plot & KDE Distribution
    Overlap for Task 1 (5-Class Differential Diagnosis, FEAT_50 & Strict-54).
  - figure4_ci_overlap_task2.png: 10,000-Iteration Bootstrap 95% CI Forest Plot & KDE Distribution
    Overlap for Task 2 (Binary Normal-vs.-Abnormal Screening, FEAT_50).
  - figure5_external_validation_transfer.png: Cross-Dataset Transfer (Strict Zero-Shot vs. Unsupervised
    Target Standardization on Chapman-Shaoxing & CPSC-2018 FEAT_50) and PTB-XL Feature Provenance
    Comparison (FEAT_50 vs. Strict-54 vs. Ref-54).
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.9

BASE_DIR = Path("/Volumes/KIOXIA/Projects/gemma3-lora-other")
PRED_DIR = BASE_DIR / "predictions"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/Users/mehmetkuntalp/.gemini/antigravity/brain/8c893aa4-e621-4aaf-83de-90c798f9240f")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def plot_ci_and_kde(npz_path: Path, res_dict: dict, title_prefix: str, out_name: str):
    data = np.load(npz_path)
    order = [
        ("teacher_tabpfn", "Teacher TabPFN (Cloud)", "#1b4f72"),
        ("distilled_xgboost", "Distilled XGBoost (Edge Student)", "#2e86c1"),
        ("distilled_mlp", "Distilled MLP (Edge Student)", "#17a589"),
        ("baseline_xgboost", "Baseline XGBoost (Edge)", "#e67e22"),
        ("baseline_mlp", "Baseline MLP (Edge)", "#c0392b"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)
    fig.subplots_adjust(wspace=0.25)

    tab_ci = res_dict["teacher_tabpfn"]["bootstrap_10k"]["macro_f1_ci_95"]
    ax1.axvspan(tab_ci[0], tab_ci[1], color="#d4e6f1", alpha=0.55, label="TabPFN 95% CI Zone")

    y_pos = np.arange(len(order))[::-1]
    for idx, (k, label, col) in enumerate(order):
        y = y_pos[idx]
        mean_val = res_dict[k]["patient_macro_f1"]
        ci_low, ci_high = res_dict[k]["bootstrap_10k"]["macro_f1_ci_95"]
        ax1.errorbar(
            mean_val, y,
            xerr=[[mean_val - ci_low], [ci_high - mean_val]],
            fmt="o", color=col, ecolor=col, elinewidth=2.4, capsize=5, markersize=8,
        )
        ax1.text(
            ci_high + 0.25, y,
            f"{mean_val:.2f}% [{ci_low:.2f}, {ci_high:.2f}]",
            va="center", ha="left", fontsize=9.2, fontweight="bold" if "teacher" in k or "distilled" in k else "normal",
            color="#1c2833",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([x[1] for x in order], fontsize=10, fontweight="bold")
    ax1.set_xlabel("Out-of-Fold Patient Macro F1 (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title(f"(A) {title_prefix}: 95% Bootstrap CIs", fontsize=11.0, fontweight="bold", pad=10)
    all_lows = [res_dict[k]["bootstrap_10k"]["macro_f1_ci_95"][0] for k, _, _ in order]
    all_highs = [res_dict[k]["bootstrap_10k"]["macro_f1_ci_95"][1] for k, _, _ in order]
    ax1.set_xlim(min(all_lows) - 1.2, max(all_highs) + 5.2)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    x_grid = np.linspace(min(all_lows) - 1.5, max(all_highs) + 1.5, 400)
    for k, label, col in order:
        arr = data[k]
        kde = gaussian_kde(arr)
        dens = kde(x_grid)
        ax2.plot(x_grid, dens, label=label, color=col, linewidth=2.2)
        ax2.fill_between(x_grid, 0, dens, color=col, alpha=0.14)

    ax2.set_xlabel("Bootstrap Patient Macro F1 (%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Empirical Probability Density", fontsize=10.5, fontweight="bold")
    ax2.set_title("(B) 10,000-Iteration Empirical KDE Distribution", fontsize=11.0, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.8)
    ax2.grid(linestyle="--", alpha=0.4)

    out_path = FIG_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    shutil.copy2(out_path, ARTIFACT_DIR / out_name)
    print(f"Saved {out_path} and copied to artifact directory.")


def plot_multi_cohort_summary(ptb_res: dict, rev1_res: dict, rev2_res: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=300)
    fig.subplots_adjust(wspace=0.26)

    labels = ["Base\nMLP", "Distilled\nMLP", "Base\nXGB", "Distilled\nXGB", "Teacher\nTabPFN"]
    colors = ["#c0392b", "#17a589", "#e67e22", "#2e86c1", "#1b4f72"]

    ptb_vals = [
        ptb_res["task1_50feat_ablation"]["baseline_mlp"]["patient_macro_f1"],
        ptb_res["task1_50feat_ablation"]["distilled_mlp"]["patient_macro_f1"],
        ptb_res["task1_50feat_ablation"]["baseline_xgboost"]["patient_macro_f1"],
        ptb_res["task1_50feat_ablation"]["distilled_xgboost"]["patient_macro_f1"],
        ptb_res["task1_50feat_ablation"]["teacher_tabpfn"]["patient_macro_f1"],
    ]
    chap_m = rev2_res["chapman_feat50_task1"]["metrics"]
    chap_vals = [
        chap_m["base_mlp"]["macro_f1"],
        chap_m["dist_mlp"]["macro_f1"],
        chap_m["base_xgb"]["macro_f1"],
        chap_m["dist_xgb"]["macro_f1"],
        chap_m["tabpfn"]["macro_f1"],
    ]
    cpsc_m = rev1_res["exp2_strict_2lead_and_controls"]["cpsc2018_50feat_waveform_only_task1"]
    cpsc_vals = [
        cpsc_m["base_mlp"]["macro_f1"],
        cpsc_m["dist_mlp"]["macro_f1"],
        cpsc_m["base_xgb"]["macro_f1"],
        cpsc_m["dist_xgb"]["macro_f1"],
        cpsc_m["tabpfn"]["macro_f1"],
    ]

    panels = [
        ("PTB-XL Discovery (5-Class, N=5,000 Patients, FEAT_50)", ptb_vals, (66, 78.5)),
        ("Chapman-Shaoxing (5-Class, N=5,000 Patients, FEAT_50)", chap_vals, (60, 72.0)),
        ("CPSC-2018 Multi-Center (4-Class, N=1,600 Records, FEAT_50)", cpsc_vals, (71, 84.5)),
    ]

    for ax, (title, vals, ylim) in zip(axes, panels):
        bars = ax.bar(labels, vals, color=colors, width=0.58, edgecolor="#1c2833", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.30, f"{v:.2f}%", ha="center", va="bottom", fontsize=9.2, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_ylabel("Out-of-Fold Macro F1 (%)", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=10.0, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_name = "figure2_macro_f1_comparison.png"
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    shutil.copy2(out_path, ARTIFACT_DIR / out_name)
    print(f"Saved {out_path} and copied to artifact directory.")


def plot_zeroshot_and_ablation(ptb_res: dict, ext_res: dict, rev1_res: dict, rev2_res: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=300)
    fig.subplots_adjust(wspace=0.25)

    models_ext = ["baseline_mlp", "distilled_mlp", "baseline_xgboost", "distilled_xgboost", "teacher_tabpfn"]
    models_r2 = ["base_mlp", "dist_mlp", "base_xgb", "dist_xgb", "tabpfn"]
    short_labels = ["Base MLP", "Distilled MLP", "Base XGB", "Distilled XGB", "TabPFN"]
    x = np.arange(len(short_labels))
    w = 0.25

    # Panel A: Cross-Dataset Transfer (Strict Zero-Shot vs Unsupervised Target Standardization on Chapman & CPSC FEAT_50)
    chap_naive = [ext_res["zero_shot_transfer"]["chapman_zeroshot"]["naive_task2"][m]["patient_macro_f1"] for m in models_ext]
    chap_harm = [ext_res["zero_shot_transfer"]["chapman_zeroshot"]["harmonized_task2"][m]["patient_macro_f1"] for m in models_ext]
    cpsc_harm_50 = [rev2_res["cross_dataset_feat50"]["ptb_to_cpsc_feat50_task2_standardized_A4"][m]["macro_f1"] for m in models_r2]

    b1 = ax1.bar(x - w, chap_naive, width=w, label="Strict Zero-Shot (Frozen PTB-XL Scaler)", color="#95a5a6", edgecolor="#1c2833")
    b2 = ax1.bar(x, chap_harm, width=w, label="Unsupervised Target-Std (PTB-XL → Chapman, N=5,000)", color="#2e86c1", edgecolor="#1c2833")
    b3 = ax1.bar(x + w, cpsc_harm_50, width=w, label="Unsupervised Target-Std (PTB-XL → CPSC FEAT_50, N=1,600)", color="#17a589", edgecolor="#1c2833")

    for bars in [b2, b3]:
        for b in bars:
            v = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(35, 84)
    ax1.set_ylabel("Macro F1 (%)", fontsize=10, fontweight="bold")
    ax1.set_title("(A) Cross-Dataset Transfer: Strict Zero-Shot vs. Target Standardization", fontsize=10.3, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8.1, frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: PTB-XL Task 1 Across Feature Provenance Schemas (FEAT_50 vs Strict-54 vs Ref-54)
    f50 = [ptb_res["task1_50feat_ablation"][m]["patient_macro_f1"] for m in models_ext]
    s54_dict = rev1_res["exp2_strict_2lead_and_controls"]["strict_2lead_54feat_task1"]
    s54 = [
        s54_dict["strict2lead_base_mlp_64_32"]["macro_f1"],
        s54_dict["strict2lead_dist_mlp_128_64"]["macro_f1"],
        s54_dict["strict2lead_base_xgb"]["macro_f1"],
        s54_dict["strict2lead_dist_xgb"]["macro_f1"],
        s54_dict["strict2lead_tabpfn"]["macro_f1"],
    ]
    f54 = [ptb_res["task1_54feat"][m]["patient_macro_f1"] for m in models_ext]

    b50 = ax2.bar(x - w, f50, width=w, label="Schema 1: FEAT_50 (Pure Waveform-Derived, Primary)", color="#5dade2", edgecolor="#1c2833")
    bs54 = ax2.bar(x, s54, width=w, label="Schema 2: Strict-54 (Strict Lead I/II Flags + Demo, Primary)", color="#17a589", edgecolor="#1c2833")
    br54 = ax2.bar(x + w, f54, width=w, label="Schema 3: Ref-54 (Reference-Assisted Sensitivity)", color="#1b4f72", edgecolor="#1c2833")

    for bars in [b50, bs54, br54]:
        for b in bars:
            v = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.20, f"{v:.1f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(66, 79.0)
    ax2.set_ylabel("Out-of-Fold Patient Macro F1 (%)", fontsize=10, fontweight="bold")
    ax2.set_title("(B) Feature Provenance Comparison on PTB-XL (FEAT_50 vs. Strict-54 vs. Ref-54)", fontsize=10.3, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=8.1, frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    out_name = "figure5_external_validation_transfer.png"
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    shutil.copy2(out_path, ARTIFACT_DIR / out_name)
    print(f"Saved {out_path} and copied to artifact directory.")


def main():
    with open(RESULTS_DIR / "ptbxl_distillation_results.json", "r", encoding="utf-8") as f:
        ptb_res = json.load(f)
    with open(RESULTS_DIR / "external_validation_results.json", "r", encoding="utf-8") as f:
        ext_res = json.load(f)
    with open(RESULTS_DIR / "reviewer_revisions_audit.json", "r", encoding="utf-8") as f:
        rev1_res = json.load(f)
    with open(RESULTS_DIR / "second_round_revisions.json", "r", encoding="utf-8") as f:
        rev2_res = json.load(f)

    plot_multi_cohort_summary(ptb_res, rev1_res, rev2_res)
    plot_ci_and_kde(
        PRED_DIR / "boot10k_task1_50feat.npz",
        ptb_res["task1_50feat_ablation"],
        "Task 1 (5-Class Diagnosis, Primary FEAT_50 Schema)",
        "figure3_ci_overlap_task1.png",
    )
    plot_ci_and_kde(
        PRED_DIR / "boot10k_task2_50feat.npz",
        ptb_res["task2_50feat_ablation"],
        "Task 2 (Binary Screening, Primary FEAT_50 Schema)",
        "figure4_ci_overlap_task2.png",
    )
    plot_zeroshot_and_ablation(ptb_res, ext_res, rev1_res, rev2_res)


if __name__ == "__main__":
    main()
