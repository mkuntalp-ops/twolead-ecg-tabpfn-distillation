#!/usr/bin/env python3
"""Generate main-text Figures 1–3 of the PMEA manuscript from stored study outputs.

All plotted numbers are read from saved files (no manual transcription):
  Figure 1  cohort sizes and class counts from the Phase 3D cohort manifest, the Phase 3B feature table,
            the Phase 3C OOF archive and excluded-record list; data-role sizes from the Phase 3B protocol.
  Figure 2  paired macro-F1 differences and 95% bootstrap intervals from
            results/phase3b_statistical_closing_results.json,
            experiments_phase3/phase3c_sensitivity/phase3c_sensitivity_results.json (evaluation B) and
            experiments_phase3/phase3d_full_cohort/phase3d_statistical_tests_table.csv.
  Figure 3  per-class F1 and recall recomputed from the Phase 3D OOF predictions.

Usage (from the project root):  python figures_main/make_main_figures.py
Outputs: figures_main/Figure_{1,2,3}.pdf (vector) and .png (300 dpi).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

# Validated categorical slots 1–3 (all-pairs CVD ΔE ≥ 9.2) plus a neutral reference grey
C_DISTILL, C_SMOOTH, C_HARD, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASSES = ["NORM", "AFIB/AFL", "CD", "STTC/MI", "HYP"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
CM = 1 / 2.54


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 1
def figure1():
    manifest = pd.read_csv(ROOT / "experiments_phase3/phase3d_full_cohort/phase3d_eligible_cohort_15160_manifest.csv")
    sel = pd.read_csv(ROOT / "data/ptbxl_54features.csv")
    c3 = np.load(ROOT / "experiments_phase3/phase3c_sensitivity/phase3c_oof_predictions.npz", allow_pickle=True)
    excl = pd.read_csv(ROOT / "experiments_phase3/phase3c_sensitivity/excluded_records_hyp_sttc.csv")
    n_d, n_b, n_c, n_x = len(manifest), len(sel), len(c3["y_common"]), len(excl)
    assert manifest["patient_id"].is_unique and n_b - n_x == n_c
    cnt_d = manifest["target_name"].value_counts()
    cnt_b = sel["target_s1"].map(dict(enumerate(CLASSES))).value_counts()
    cnt_c = pd.Series(c3["y_common"]).map(dict(enumerate(CLASSES))).value_counts()
    fmt = lambda c: " / ".join(f"{c.get(k, 0):,}" for k in CLASSES)

    fig = plt.figure(figsize=(16 * CM, 9.2 * CM))
    ax = fig.add_axes([0, 0, 0.47, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 9.75, "(a) Cohorts (PTB-XL, leads I and II)", fontsize=9, weight="bold", color=INK, va="top")

    def box(a, x, y, w, h, title, lines, edge):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                   fc="white", ec=edge, lw=1.2))
        a.text(x + 0.2, y + h - 0.25, title, fontsize=7.4, weight="bold", color=INK, va="top")
        for i, ln in enumerate(lines):
            a.text(x + 0.2, y + h - 0.75 - 0.42 * i, ln, fontsize=7, color=MUTED, va="top")

    ax.text(0.2, 9.15, "Class counts: NORM / AFIB/AFL / CD / STTC/MI / HYP", fontsize=6.8, color=MUTED, va="top")
    box(ax, 0.2, 6.3, 9.4, 2.5, f"Eligibility-filtered pool: Phase 3D (n = {n_d:,})",
        ["adults; consistent single-class assignment;", "first eligible recording per patient",
         fmt(cnt_d)], MUTED)
    box(ax, 0.2, 3.4, 9.4, 2.2, f"Quota-selected cohort: Phase 3B (n = {n_b:,})",
        ["subset of the pool; quota seed 42;", "five outer partition seeds (42–46)",
         fmt(cnt_b)], C_DISTILL)
    box(ax, 0.2, 0.5, 9.4, 2.2, f"HYP+STTC exclusion: Phase 3C (n = {n_c:,})",
        [f"{n_x} co-occurring HYP+STTC patients removed;", "seed 42; evaluation A (rescore), B (retrain)",
         fmt(cnt_c)], C_SMOOTH)
    for y0, y1, lab in [(6.3, 5.6, "quota subsample"), (3.4, 2.7, f"exclude {n_x}")]:
        ax.annotate("", xy=(4.9, y1), xytext=(4.9, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
        ax.text(5.15, (y0 + y1) / 2, lab, fontsize=7, color=MUTED, va="center")

    bx = fig.add_axes([0.52, 0, 0.48, 1]); bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis("off")
    bx.text(0, 9.75, "(b) Data roles within each outer fold", fontsize=9, weight="bold", color=INK, va="top")
    # proportional strip: outer test 20%, student training 64%, validation 16% (selected-cohort sizes)
    segs = [("Test", 1000, "#d9d8d4"), ("Student training", 3200, "#cfe0f6"), ("Val.", 800, "#fbe0d4")]
    x = 0.2; y, h = 7.2, 1.3; total = sum(s[1] for s in segs)
    for lab, n, col in segs:
        w = 9.4 * n / total
        bx.add_patch(FancyBboxPatch((x, y), w - 0.08, h, boxstyle="round,pad=0,rounding_size=0.12", fc=col, ec="none"))
        bx.text(x + (w - 0.08) / 2, y + h / 2 + 0.18, lab, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
        bx.text(x + (w - 0.08) / 2, y + h / 2 - 0.3, f"{n:,}", ha="center", va="center", fontsize=7, color=MUTED)
        x += w
    bx.text(0.2, 6.75, "Outer test / student training / validation per fold in Phase 3B;\nPhase 3D uses the same proportions.",
            fontsize=6.8, color=MUTED, va="top")
    steps = [("Scaler", ["fitted on student training only"]),
             ("Teacher targets", ["3-fold inner OOF TabPFN (4 estimators)", "within student training"]),
             ("Student targets", ["one-hot  |  0.85·one-hot + 0.15/C  |", "0.5·one-hot + 0.5·teacher"]),
             ("Early stopping", ["validation macro F1, true labels", "(patience 15)"]),
             ("Teacher reference", ["TabPFN (8 estimators) fitted on", "student training"]),
             ("Evaluation", ["outer test patients only;", "one prediction per patient"])]
    for i, (k, lines) in enumerate(steps):
        yy = 5.55 - 0.88 * i
        bx.text(0.2, yy, k, fontsize=7.3, weight="bold", color=INK, va="top")
        bx.text(3.7, yy, "\n".join(lines), fontsize=6.9, color=MUTED, va="top", linespacing=1.25)
    save(fig, "Figure_1")


# --------------------------------------------------------------------------- Figure 2
def figure2():
    b = json.load(open(ROOT / "results/phase3b_statistical_closing_results.json"))["patient_blocked_paired_bootstrap_summary_estimands"]
    c = json.load(open(ROOT / "experiments_phase3/phase3c_sensitivity/phase3c_sensitivity_results.json"))["evaluation_b_retrained_sensitivity"]["bootstrap_comparisons"]
    d = pd.read_csv(ROOT / "experiments_phase3/phase3d_full_cohort/phase3d_statistical_tests_table.csv")
    d = d[d["family_size_m"] == 5].set_index("comparison")
    rows = {
        "Phase 3B\nquota-selected\n(n = 5,000; 5 partitions)": [
            (b["secondary_core_distill_vs_hard128"]["point_mean_delta_pp"], *b["secondary_core_distill_vs_hard128"]["ci_95"]),
            (b["primary_distill_vs_smooth"]["point_mean_delta_pp"], *b["primary_distill_vs_smooth"]["ci_95"]),
            (b["exploratory_smooth_vs_hard128"]["point_mean_delta_pp"], *b["exploratory_smooth_vs_hard128"]["ci_95"])],
        "Phase 3C\nHYP+STTC excluded\n(n = 4,504; retrained)": [
            (c["distill_vs_hard128"]["point_delta_pp"], *c["distill_vs_hard128"]["ci_95"]),
            (c["distill_vs_smooth"]["point_delta_pp"], *c["distill_vs_smooth"]["ci_95"]),
            (c["smooth_vs_hard128"]["point_delta_pp"], *c["smooth_vs_hard128"]["ci_95"])],
        "Phase 3D\neligibility-filtered\n(n = 15,160)": [
            tuple(d.loc["Distilled vs. Hard-Label (128, 64)", ["point_delta_pp", "ci_95_low", "ci_95_high"]]),
            tuple(d.loc["Distilled vs. Uniform-Smoothed (128, 64)", ["point_delta_pp", "ci_95_low", "ci_95_high"]]),
            tuple(d.loc["Uniform-Smoothed vs. Hard-Label (128, 64)", ["point_delta_pp", "ci_95_low", "ci_95_high"]])],
    }
    series = [("Distilled − hard label", C_DISTILL, "o"), ("Distilled − smoothing", C_SMOOTH, "s"),
              ("Smoothing − hard label", C_HARD, "^")]
    fig, ax = plt.subplots(figsize=(15 * CM, 8.5 * CM))
    ylabels, yticks = [], []
    for g, (lab, vals) in enumerate(rows.items()):
        base = (len(rows) - 1 - g) * 4
        yticks.append(base + 1); ylabels.append(lab)
        for j, ((pt, lo, hi), (name, col, mk)) in enumerate(zip(vals, series)):
            y = base + 2 - j
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(pt, y, marker=mk, ms=7, mfc=col, mec="white", mew=1.2, ls="none", zorder=3,
                    label=name if g == 0 else None)
            m = lambda v, s="": f"{v:{s}.2f}".replace("-", "−")
            ax.text(max(hi, 0) + 0.08, y, f"{m(pt, '+')} [{m(lo)}, {m(hi)}]", va="center", fontsize=6.8, color=MUTED)
        if g:
            ax.axhline(base + 3.5, color=GRID, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Paired difference in macro F1 (percentage points), 95% patient-blocked bootstrap interval")
    ax.set_xlim(-1.2, 3.6); ax.set_ylim(-0.8, 11.2)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, frameon=False, handletextpad=0.4)
    save(fig, "Figure_2")


# --------------------------------------------------------------------------- Figure 3
def figure3():
    z = np.load(ROOT / "experiments_phase3/phase3d_full_cohort/phase3d_full_cohort_oof_predictions.npz", allow_pickle=True)
    y = z["y_true"]
    models = [("Distilled MLP (128–64)", "distilled_mlp_128_64", C_DISTILL, "o"),
              ("Smoothed MLP (128–64)", "smooth_mlp_128_64", C_SMOOTH, "s"),
              ("Hard-label MLP (128–64)", "hard_mlp_128_64", C_HARD, "^"),
              ("TabPFN teacher (reference)", "standalone_tabpfn", C_REF, "D")]
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.8 * CM), sharey=False)
    for ax, metric, title in [(axes[0], f1_score, "(a) Class F1"), (axes[1], recall_score, "(b) Class recall (sensitivity)")]:
        for k, (name, key, col, mk) in enumerate(models):
            vals = 100 * metric(y, z[f"{key}_preds"], labels=range(5), average=None, zero_division=0)
            xs = np.arange(5) + (k - 1.5) * 0.16
            hollow = key == "standalone_tabpfn"
            ax.plot(xs, vals, marker=mk, ms=6.5, ls="none", mec=col if hollow else "white", mew=1.3 if hollow else 1.0,
                    mfc="white" if hollow else col, label=name, zorder=3)
        ax.set_xticks(range(5)); ax.set_xticklabels(CLASSES)
        ax.set_ylim(0, 100); ax.set_ylabel("%"); ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for x in np.arange(4) + 0.5:
            ax.axvline(x, color=GRID, lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Figure_3")


if __name__ == "__main__":
    figure1(); figure2(); figure3()
    print("wrote", sorted(p.name for p in OUT.glob("Figure_*")))
