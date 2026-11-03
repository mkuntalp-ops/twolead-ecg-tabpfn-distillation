#!/usr/bin/env python3
"""Figures of the v22 manuscript, generated only from stored WP3 outputs (no values are typed in).

Usage (project root):  python revizyon_v22/manuscript/make_figures_v22.py
Writes revizyon_v22/manuscript/figures/Fig{1..4}.{pdf,png}.
  Fig1  study design (schematic; cohort counts read from the WP3 cohort record files)
  Fig2  paired differences in Macro F1 (D_tuned - S_tuned, D_tuned - EM_tuned, D_tuned - H; MLP and
        XGBoost): partition-level differences and single-partition bootstrap intervals (wp3_contrasts_per_seed.csv)
  Fig3  descriptive grid: outer-test Macro F1 of every MLP target versus mean target entropy (wp3_grid_test_f1.csv);
        test data were not used for any selection, the grid is shown for description only
  Fig4  top-label reliability, PTB-XL 15,160 and Chapman 9,315 (wp3_secondary.py outputs, re-plotted side by side)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revizyon_v22/wp3"))
from wp3_common import CLASSES, COHORTS, OUT_ROOT  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
SUM = OUT_ROOT / "summary"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"H": "#1baf7a", "S": "#eb6834", "D": "#2a78d6", "EM": "#6250d6", "teacher": "#6b6a66"}
LABEL = {"ptbxl_quota5000": "PTB-XL quota-5000", "ptbxl_natural5000": "PTB-XL natural-5000",
         "ptbxl_natural15160": "PTB-XL eligible-15,160", "chapman_quota5000": "Chapman quota-5000",
         "chapman_natural": "Chapman eligible-9,315"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1():
    counts = {}
    for c in COHORTS:
        r = pd.read_csv(OUT_ROOT / c / "cohort_records.csv")
        counts[c] = (len(r), np.bincount(r["y"], minlength=5))
    fig, ax = plt.subplots(figsize=(7.6, 4.8)); ax.set_axis_off(); ax.set_xlim(0, 101); ax.set_ylim(-1.2, 62)

    def box(x, y, w, h, text, fc="#f4f3f0", ec=INK2, fs=7.2, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))

    ax.text(0, 60, "(a) Cohorts (one recording per patient)", fontsize=8, weight="bold", color=INK)
    ax.text(0, 57.4, "Class counts: " + " / ".join(CLASSES), fontsize=6.8, color=INK2)
    rows = [("ptbxl_natural15160", "eligible pool; 1 partition"), ("ptbxl_quota5000", "class-quota subset; 5 partitions"),
            ("ptbxl_natural5000", "stratified random subset; 5 partitions"),
            ("chapman_natural", "eligible pool; 1 partition"), ("chapman_quota5000", "class-quota subset; 5 partitions")]
    for i, (c, role) in enumerate(rows):
        n, k = counts[c]
        box(0, 48.5 - i * 7.2, 46, 5.6, f"{LABEL[c]}  (N = {n:,})\n{role}\n{' / '.join(f'{v:,}' for v in k)}", fs=6.2)
    ax.text(0, 5, "Features: 50 per recording (25 from lead I, 25 from lead II):\n"
            "intervals, amplitudes/ST, RR variability,\nspectral and statistical descriptors", fontsize=6.8, color=INK2)

    ax.text(53, 60, "(b) Data roles in each outer fold", fontsize=8, weight="bold", color=INK)
    box(53, 50, 18, 4.6, "Outer test (20%)\nscored once", fc="#fdf0ea", fs=6.6)
    box(75, 50, 25, 4.6, "Outer training pool (80%)", fs=6.6)
    box(75, 41, 11, 4.6, "Student\ntraining (64%)", fs=6.0)
    box(87.6, 41, 12.4, 4.6, "Student\nvalidation (16%)", fs=6.0)
    arrow(80.5, 50, 80.5, 45.8); arrow(93.8, 50, 93.8, 45.8)
    box(53, 31, 34, 5.4, "TabPFN teacher: 3-fold grouped out-of-fold\nprobabilities within student training only", fs=6.6)
    arrow(80.5, 41, 75, 36.6)
    box(53, 21, 47, 6.2, "Targets for the same student (shared initial weights):\nH hard · S(ε) uniform smoothing · D(α, T) distillation ·\n"
        "EM(α, T) uniform smoothing with the entropy of D(α, T)", fs=6.4)
    arrow(70, 31, 70, 27.4)
    box(53, 10, 22, 6.2, "MLP 50–128–64–5\n(15,109 parameters)", fs=6.6)
    box(78, 10, 22, 6.2, "XGBoost\n(soft targets as weights)", fs=6.6)
    arrow(64, 21, 64, 16.4); arrow(89, 21, 89, 16.4)
    box(53, 0.0, 47, 6.4, "Student validation: early stopping, and choice of ε and (α, T)\nby Macro F1; outer-test predictions pooled per partition\n"
        "and compared pairwise", fs=6.2)
    save(fig, "Fig1")


def fig2():
    """Pre-specified summary: every partition-level difference (five-partition cohorts, small markers; large marker = mean)
    and the single-partition difference with its B = 2,000 interval."""
    d = pd.read_csv(SUM / "wp3_contrasts_per_seed.csv")
    con = [("D_tuned - S_tuned", "(a) Distillation −\nuniform smoothing"),
           ("D_tuned - EM_tuned", "(b) Distillation −\nentropy-matched smoothing"), ("D_tuned - H", "(c) Distillation −\nhard labels")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    ys = np.arange(len(COHORTS))[::-1]
    for ax, (key, title) in zip(axes, con):
        for lrn, mk, off in [("mlp", "o", 0.14), ("xgb", "s", -0.14)]:
            col = COL["D"] if lrn == "mlp" else COL["S"]
            for y, c in zip(ys, COHORTS):
                g = d[(d.contrast == key) & (d.learner == lrn) & (d.cohort == c)]
                if len(g) == 1:
                    q = g.iloc[0]
                    ax.errorbar(q.d_macro_F1, y + off, xerr=[[q.d_macro_F1 - q.d_macro_F1_lo], [q.d_macro_F1_hi - q.d_macro_F1]],
                                fmt=mk, ms=4.5, color=col, ecolor=col, elinewidth=1.4, capsize=0)
                else:
                    ax.scatter(g.d_macro_F1, np.full(len(g), y + off), marker=mk, s=9, facecolor="white", edgecolor=col,
                               linewidth=0.8, zorder=2)
                    ax.scatter(g.d_macro_F1.mean(), y + off, marker=mk, s=26, color=col, zorder=3)
            ax.scatter([], [], marker=mk, s=26, color=col, label="MLP student" if lrn == "mlp" else "XGBoost student")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=7.6, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("Difference in Macro F1 (pp)")
    axes[0].set_yticks(ys, [f"{LABEL[c]}\n({d.query('cohort == @c').seed.nunique()} partition"
                            f"{'s' if c.endswith('5000') else ''})" for c in COHORTS])
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    save(fig, "Fig2")


def fig3():
    g = pd.read_csv(SUM / "wp3_grid_test_f1.csv").query("learner == 'mlp'")
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.9))
    axes = axs.ravel()[:5]
    for ax, c in zip(axes, COHORTS):
        h = g[g.cohort == c]
        dd = h[h.arm.str.startswith("D_")].set_index(h[h.arm.str.startswith("D_")].arm.str[2:])
        em = h[h.arm.str.startswith("EM_")].set_index(h[h.arm.str.startswith("EM_")].arm.str[3:])
        for k in dd.index:  # matched pairs share the same entropy by construction
            ax.plot([dd.loc[k, "target_mean_entropy"]] * 2, [em.loc[k, "test_macro_F1"], dd.loc[k, "test_macro_F1"]],
                    color="#c9c8c3", lw=1.0, zorder=1)
        s = h[h.arm.str.startswith("S_")]; hh = h[h.arm == "H"]
        ax.plot(s.target_mean_entropy, s.test_macro_F1, "-s", color=COL["S"], ms=3.5, lw=1.2, label="Uniform smoothing S(ε)")
        ax.scatter(em.target_mean_entropy, em.test_macro_F1, marker="^", s=30, facecolor="none", edgecolor=COL["EM"],
                   linewidth=1.0, zorder=5, label="Entropy-matched EM(α, T)")  # open, drawn over D so coinciding points stay visible
        ax.scatter(dd.target_mean_entropy, dd.test_macro_F1, marker="o", s=18, color=COL["D"], zorder=4,
                   label="Distillation D(α, T)", edgecolor="white", linewidth=0.4)
        ax.scatter(hh.target_mean_entropy, hh.test_macro_F1, marker="D", s=20, color=COL["H"], zorder=4, label="Hard labels H")
        ax.set_title(LABEL[c], fontsize=7.4, loc="left", color=INK)
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
    for ax in (axs[0, 0], axs[1, 0]):
        ax.set_ylabel("Outer-test Macro F1 (%)")
    for ax in axs[1, :2]:
        ax.set_xlabel("Mean target entropy (nats)")
    axs[0, 2].set_xlabel("Mean target entropy (nats)")
    leg = axs[1, 2]; leg.set_axis_off()
    leg.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7.2, loc="center left",
               title="MLP student; mean over partitions.\nGrey lines join each D(α, T) to\nits entropy-matched EM(α, T).",
               title_fontsize=7, alignment="left")
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, "Fig3")


def fig4():
    import wp3_secondary as S  # noqa: E402
    wp4 = S.wp4
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    style = {"TabPFN teacher": (COL["teacher"], "D"), "MLP hard": (COL["H"], "^"), "MLP smooth (fixed)": (COL["S"], "s"),
             "MLP distil (fixed)": (COL["D"], "o"), "MLP distil (tuned)": (COL["EM"], "v")}
    for ax, c in zip(axes, ["ptbxl_natural15160", "chapman_natural"]):
        y, _, probs = S.cohort_models(c, 42)
        for m, (col, mk) in style.items():
            p = probs[m]; conf = p.max(1); corr = p.argmax(1) == y
            b = np.minimum((conf * wp4.N_BINS).astype(int), wp4.N_BINS - 1)
            xs = [conf[b == i].mean() for i in range(wp4.N_BINS) if (b == i).sum() >= 30]
            ys_ = [corr[b == i].mean() for i in range(wp4.N_BINS) if (b == i).sum() >= 30]
            ax.plot(xs, ys_, marker=mk, color=col, lw=1.4, ms=3.8, label=f"{m} (ECE {100 * wp4.top_label_ece(y, p):.1f}%)")
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.7, ls="--")
        ax.set_xlim(0.2, 1); ax.set_ylim(0.2, 1); ax.set_xlabel("Mean top-label confidence (15 bins, n ≥ 30)")
        ax.set_title(f"{LABEL[c]} (N = {len(y):,})", fontsize=7.6, loc="left", color=INK)
        ax.legend(frameon=False, fontsize=6.4, loc="lower right")
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Observed accuracy")
    save(fig, "Fig4")


if __name__ == "__main__":
    fig1(); fig3()
    if (SUM / "wp3_contrasts_per_seed.csv").exists():
        fig2()
    fig4()
