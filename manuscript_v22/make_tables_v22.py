#!/usr/bin/env python3
"""Tables of the v22 manuscript and supplement, generated only from stored WP3 outputs (no values are typed in).

Usage (project root):  python revizyon_v22/manuscript/make_tables_v22.py
Writes revizyon_v22/manuscript/tables/*.md (pipe tables; included by the manuscript build) and tables_numbers.json
(every number quoted in the text, for a consistency check of the prose against the tables).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revizyon_v22/wp3"))
from wp3_common import CLASSES, COHORTS, OUT_ROOT  # noqa: E402

TAB = Path(__file__).resolve().parent / "tables"
SUM, SEC, COST = OUT_ROOT / "summary", OUT_ROOT / "secondary", OUT_ROOT / "cost"
LABEL = {"ptbxl_quota5000": "PTB-XL quota-5000", "ptbxl_natural5000": "PTB-XL natural-5000",
         "ptbxl_natural15160": "PTB-XL eligible-15,160", "chapman_quota5000": "Chapman quota-5000",
         "chapman_natural": "Chapman eligible-9,315"}
MODELS = [("teacher_tabpfn", "teacher", "TabPFN teacher"), ("mlp", "H", "MLP, hard labels"),
          ("mlp", "S_fixed", "MLP, smoothing ε = 0.15"), ("mlp", "D_fixed", "MLP, distillation α = 0.5, T = 1"),
          ("mlp", "S_tuned", "MLP, smoothing (tuned)"), ("mlp", "EM_tuned", "MLP, entropy-matched (tuned)"),
          ("mlp", "D_tuned", "MLP, distillation (tuned)"), ("xgb", "H", "XGBoost, hard labels"),
          ("xgb", "S_tuned", "XGBoost, smoothing (tuned)"), ("xgb", "EM_tuned", "XGBoost, entropy-matched (tuned)"),
          ("xgb", "D_fixed", "XGBoost, distillation α = 0.5, T = 1"), ("xgb", "D_tuned", "XGBoost, distillation (tuned)")]
CON = [("D_tuned - S_tuned", "Distillation − smoothing (tuned)"),
       ("D_tuned - EM_tuned", "Distillation − entropy-matched (tuned)"),
       ("D_tuned - H", "Distillation (tuned) − hard labels"), ("S_tuned - H", "Smoothing (tuned) − hard labels"),
       ("EM_tuned - H", "Entropy-matched (tuned) − hard labels"),
       ("D_fixed - S_fixed", "Distillation − smoothing (fixed α = 0.5, ε = 0.15)"),
       ("D_fixed - H", "Distillation (fixed) − hard labels")]
NUM: dict = {}
METRIC_NAME = {"macro_F1": "Macro F1 (%)", "macro_AUROC": "Macro AUROC (%)", "macro_AUPRC": "Macro AUPRC (%)",
               "NLL": "NLL (nats)", "Brier": "Brier", "ECE_top_label_15bin": "ECE, top-label (%)",
               "ECE_classwise_15bin": "ECE, class-wise (%)", "binary_sensitivity": "Sensitivity, abnormal (%)",
               "binary_specificity": "Specificity, abnormal (%)", "HYP_recall": "HYP recall (%)"}
SUBGROUP_NAME = {"male": "Men", "female": "Women", "age<65": "Age < 65 years", "age>=65": "Age ≥ 65 years"}
SEC_MODEL = {"TabPFN teacher": "TabPFN teacher", "MLP hard": "MLP, hard labels", "MLP smooth (fixed)": "MLP, smoothing ε = 0.15",
             "MLP distil (fixed)": "MLP, distillation α = 0.5, T = 1", "MLP smooth (tuned)": "MLP, smoothing (tuned)",
             "MLP entropy-matched (tuned)": "MLP, entropy-matched (tuned)", "MLP distil (tuned)": "MLP, distillation (tuned)",
             "XGB hard": "XGBoost, hard labels", "XGB smooth (fixed)": "XGBoost, smoothing ε = 0.15",
             "XGB smooth (tuned)": "XGBoost, smoothing (tuned)", "XGB entropy-matched (tuned)": "XGBoost, entropy-matched (tuned)",
             "XGB distil (fixed)": "XGBoost, distillation α = 0.5, T = 1", "XGB distil (tuned)": "XGBoost, distillation (tuned)"}


def arm_name(a: str) -> str:
    """Readable target name, e.g. D_a0.50_T1 -> D(α = 0.50, T = 1)."""
    if a == "H":
        return "Hard labels"
    if a.startswith("S_e"):
        return f"S(ε = {a[3:]})"
    fam, al, t = a.split("_")
    return f"{fam}(α = {al[1:]}, T = {t[1:]})"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out) + "\n"


def write(name, df, caption):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.md").write_text(f"{caption}\n\n{md(df)}")


def ms(x: pd.Series, d=2):
    return f"{x.mean():.{d}f}" if len(x) == 1 else f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}"


def table1():
    rows = []
    roles = {"ptbxl_quota5000": ("PTB-XL eligible pool", "Class-quota subsample (seed 42): 1,045 per class, all 820 HYP", "42–46"),
             "ptbxl_natural5000": ("PTB-XL eligible pool", "Class-stratified random subsample (seed 2026), natural proportions", "42–46"),
             "ptbxl_natural15160": ("PTB-XL, 21,799 records / 18,869 patients", "All eligible adults, first eligible record per patient", "42"),
             "chapman_quota5000": ("Chapman-Shaoxing eligible pool", "Class-quota subsample (seed 42): all 590 CD and 820 HYP; 1,196 NORM, 1,197 AFIB/AFL and 1,197 STTC/MI drawn at random", "42–46"),
             "chapman_natural": ("Chapman-Shaoxing, 10,646 records", "All eligible adults (≥ 18 years) with one assignable class", "42")}
    for c in COHORTS:
        r = pd.read_csv(OUT_ROOT / c / "cohort_records.csv")
        k = np.bincount(r["y"], minlength=5)
        assert r["group"].is_unique, "more than one record per patient"
        rows.append({"Cohort": LABEL[c], "Source": roles[c][0], "Selection": roles[c][1], "N (patients)": f"{len(r):,}",
                     "NORM / AFIB/AFL / CD / STTC/MI / HYP": " / ".join(f"{v:,}" for v in k),
                     "Partition seeds": roles[c][2]})
        NUM[f"n_{c}"] = int(len(r)); NUM[f"classes_{c}"] = k.tolist()
    write("Table1_cohorts", pd.DataFrame(rows),
          "**Table 1.** Cohorts. Each patient contributes one recording, so patient-grouped and record-level splits coincide. "
          "The PTB-XL quota-5000 and natural-5000 cohorts are subsets of the PTB-XL eligible pool, and Chapman quota-5000 is a "
          "subset of the Chapman eligible pool; overlapping cohorts are not independent samples.")


def table2():
    m = pd.read_csv(SUM / "wp3_metrics_per_seed.csv")
    rows = []
    for lrn, mod, name in MODELS:
        row = {"Model": name}
        for c in COHORTS:
            g = m[(m.cohort == c) & (m.learner == lrn) & (m.model == mod)]
            row[LABEL[c]] = ms(g["macro_F1"])
            NUM[f"F1_{c}_{lrn}_{mod}"] = float(g["macro_F1"].mean())
            NUM[f"AUROC_{c}_{lrn}_{mod}"] = float(g["macro_AUROC"].mean())
            NUM[f"ECE_{c}_{lrn}_{mod}"] = float(g["ECE_top_label_15bin"].mean())
            NUM[f"NLL_{c}_{lrn}_{mod}"] = float(g["NLL"].mean())
        rows.append(row)
    write("Table2_macroF1", pd.DataFrame(rows),
          "**Table 2.** Outer-test Macro F1 (%) by cohort. Five-partition cohorts: mean ± SD over the five partition-level "
          "scores (each computed on all N pooled out-of-fold predictions); single-partition cohorts: point estimate. "
          "'Tuned' rows use the configuration chosen in each outer fold on the student-validation set.")
    rows = []
    for lrn, mod, name in MODELS:
        row = {"Model": name}
        for c in ["ptbxl_quota5000", "ptbxl_natural15160", "chapman_natural"]:
            g = m[(m.cohort == c) & (m.learner == lrn) & (m.model == mod)]
            row[f"{LABEL[c]} AUROC"] = ms(g["macro_AUROC"])
            row[f"{LABEL[c]} ECE"] = ms(g["ECE_top_label_15bin"])
        rows.append(row)
    write("Table3_auroc_ece", pd.DataFrame(rows),
          "**Table 3.** Macro one-versus-rest AUROC (%) and top-label expected calibration error (ECE, %, 15 equal-width bins) "
          "in three cohorts; all cohorts and further metrics are in Supplementary Table {{ST:full}}. Format as in Table 2.")


def table4():
    """Pre-specified summary of the analysis plan: per-partition differences (five-partition cohorts) or one
    bootstrap interval (single-partition cohorts), B = 2,000 per partition. The partition-averaged B = 10,000 intervals
    were added after the results had been seen and are reported only in the Supplementary Material (post hoc)."""
    s = pd.read_csv(SUM / "wp3_contrasts_per_seed.csv")
    pooled = pd.read_csv(SUM / "wp3_contrasts_pooled.csv")
    rows, prow = [], []
    for lrn, lname in [("mlp", "MLP"), ("xgb", "XGBoost")]:
        for key, name in CON:
            row = {"Student": lname, "Contrast": name}
            pr = {"Student": lname, "Contrast": name}
            for c in COHORTS:
                g = s[(s.cohort == c) & (s.learner == lrn) & (s.contrast == key)]
                r = pooled[(pooled.cohort == c) & (pooled.learner == lrn) & (pooled.contrast == key)].iloc[0]
                assert abs(g.d_macro_F1.mean() - r.d_macro_F1) < 1e-9, "pooled point estimate differs from the partition mean"
                if len(g) == 5:
                    pos, up, down = int((g.d_macro_F1 > 0).sum()), int((g.d_macro_F1_lo > 0).sum()), int((g.d_macro_F1_hi < 0).sum())
                    row[LABEL[c]] = f"{g.d_macro_F1.mean():+.2f} ± {g.d_macro_F1.std():.2f} ({pos}, {up}, {down})"
                    NUM[f"ps_{c}_{lrn}_{key}"] = [float(g.d_macro_F1.mean()), float(g.d_macro_F1.std()), pos, up, down]
                else:
                    assert len(g) == 1
                    q = g.iloc[0]
                    row[LABEL[c]] = f"{q.d_macro_F1:+.2f} [{q.d_macro_F1_lo:+.2f}, {q.d_macro_F1_hi:+.2f}]"
                    NUM[f"ps_{c}_{lrn}_{key}"] = [float(q.d_macro_F1), float(q.d_macro_F1_lo), float(q.d_macro_F1_hi)]
                pr[LABEL[c]] = f"{r.d_macro_F1:+.2f} [{r.d_macro_F1_lo:+.2f}, {r.d_macro_F1_hi:+.2f}]"
                for mtr in ("macro_F1", "NLL", "ECE_top_label_15bin"):
                    NUM[f"pooled_{c}_{lrn}_{key}_{mtr}"] = [float(r[f'd_{mtr}']), float(r[f'd_{mtr}_lo']),
                                                            float(r[f'd_{mtr}_hi']), float(r[f'p_{mtr}'])]
            rows.append(row); prow.append(pr)
    write("Table4_contrasts", pd.DataFrame(rows),
          "**Table 4.** Paired differences in Macro F1 (percentage points), summarized as specified in the analysis plan; all "
          "contrasts are exploratory. Five-partition cohorts: mean ± SD of the five partition-level differences, followed by "
          "(number of partitions with a positive difference, number whose 95% patient-blocked bootstrap interval lay above zero, "
          "number whose interval lay below zero). Single-partition cohorts: difference with its 95% patient-blocked bootstrap "
          "interval. Bootstrap B = 2,000 per partition. Intervals are conditional on the fitted models and fixed partitions and are "
          "not adjusted for multiplicity.")
    write("TableS_pooled", pd.DataFrame(prow),
          "**Supplementary Table.** Post hoc summary, added after the results had been seen: partition-averaged paired "
          "differences in Macro F1 (pp) with 95% patient-blocked bootstrap intervals (B = 10,000; in five-partition cohorts the "
          "same resampled patients are used in every partition). The point estimates equal the means in Table 4; the intervals "
          "reflect patient sampling only, not partitioning or training variability, and are not adjusted for multiplicity.")


def table5():
    x = pd.read_csv(COST / "xgb_reference_models.csv")
    n = pd.read_csv(COST / "nn_costs.csv")
    t = pd.read_csv(COST / "xgb_tree_counts.csv")
    mlp = n[n.model.str.startswith("mlp")].iloc[0]
    rows = [{"Model": "MLP student 50–128–64–5", "Stored values": f"{int(mlp.n_parameters):,}",
             "Serialized size": f"{mlp.size_bytes / 1024:.0f} kB",
             "Latency, 1 record (ms)": f"{mlp.latency_1_record_ms:.3f}",
             "Latency, 1,000 records (ms)": f"{mlp.latency_1000_records_ms:.2f}"}]
    NUM["mlp_params"] = int(mlp.n_parameters); NUM["mlp_lat1"] = round(float(mlp.latency_1_record_ms), 4)
    for mod, name in [("xgb_H", "XGBoost, hard labels"), ("xgb_D_fixed", "XGBoost, distillation α = 0.5, T = 1")]:
        g = x[x.model == mod]
        rows.append({"Model": name,
                     "Stored values": f"{int(g.stored_parameters.min()):,}–{int(g.stored_parameters.max()):,}",
                     "Serialized size": f"{g.size_ubj_bytes.min() / 2**20:.1f}–{g.size_ubj_bytes.max() / 2**20:.1f} MB",
                     "Latency, 1 record (ms)": f"{g.latency_1_record_ms.min():.3f}–{g.latency_1_record_ms.max():.3f}",
                     "Latency, 1,000 records (ms)": f"{g.latency_1000_records_ms.min():.2f}–{g.latency_1000_records_ms.max():.2f}"})
        NUM[f"{mod}_params_range"] = [int(g.stored_parameters.min()), int(g.stored_parameters.max())]
        NUM[f"{mod}_size_MB"] = [float(g.size_ubj_bytes.min() / 2**20), float(g.size_ubj_bytes.max() / 2**20)]
        NUM[f"{mod}_trees_range"] = [int(g.n_trees.min()), int(g.n_trees.max())]
        NUM[f"{mod}_param_ratio"] = [float(g.stored_parameters.min() / mlp.n_parameters),
                                     float(g.stored_parameters.max() / mlp.n_parameters)]
        assert (np.abs(g.test_F1_refit - g.test_F1_stored) < 1e-9).all(), "refitted XGBoost differs from stored predictions"
    tp = n[n.model.str.startswith("tabpfn") & n["per_record_ms"].notna()]
    for dev in ("mps", "cpu"):
        g = tp[tp.model.str.endswith(dev)]
        if len(g):
            rows.append({"Model": f"TabPFN teacher, 8 estimators ({'Apple M4 GPU, MPS' if dev == 'mps' else 'CPU'})",
                         "Stored values": "pre-trained transformer", "Serialized size": f"{g.size_bytes.iloc[0] / 2**20:.0f} MB checkpoint",
                         "Latency, 1 record (ms)": "n/a (in-context)",
                         "Latency, 1,000 records (ms)": f"{1e3 * g.per_record_ms.min():,.0f}–{1e3 * g.per_record_ms.max():,.0f} "
                                                        "(fit + predict, scaled to 1,000 test records)"})
            NUM[f"tabpfn_{dev}_s_range"] = [float(g.fit_predict_test_set_s.min()), float(g.fit_predict_test_set_s.max())]
            NUM[f"tabpfn_{dev}_s_per1000"] = [float(g.per_record_ms.min()), float(g.per_record_ms.max())]
    NUM["xgb_param_ratio"] = [float(x.stored_parameters.min() / mlp.n_parameters), float(x.stored_parameters.max() / mlp.n_parameters)]
    NUM["xgb_latency_ratio"] = [float(x.latency_1_record_ms.min() / mlp.latency_1_record_ms),
                                float(x.latency_1_record_ms.max() / mlp.latency_1_record_ms)]
    NUM["mlp_size_kB"] = float(mlp.size_bytes / 1024); NUM["tabpfn_ckpt_MB"] = float(n.size_bytes.dropna().iloc[-1] / 2**20)
    NUM["xgb_trees_med_H"] = float(t[t.arm == "H"].n_trees.median())
    NUM["xgb_trees_med_D"] = float(t[t.arm == "D_a0.50_T1"].n_trees.median())
    write("Table5_cost", pd.DataFrame(rows),
          "**Table 5.** Size and inference cost on one Apple M4 workstation (CPU, one thread, unless stated). XGBoost values are "
          "ranges over the reference fold (first partition, fold 1) of the five cohorts, after truncation at the early-stopping "
          "iteration. Stored values are MLP weights and biases and, for XGBoost, one feature index and one threshold per split "
          "node and one value per leaf; these are different kinds of quantity, so serialized size is the more direct memory "
          "comparison. Latencies are for the classifier alone; filtering, beat detection and feature extraction are excluded. "
          "TabPFN timings include fitting (storing the context) and predicting one outer-test set of 1,000–3,032 records, "
          "and therefore measure a different operation from the prediction-only latencies of the students; "
          "CPU runs with more than 5,000 context samples are refused by the tabpfn package by default and were not timed. "
          "All timings were taken with no other analysis running on the machine.")


def supplementary():
    g = pd.read_csv(SUM / "wp3_grid_test_f1.csv")
    for lrn in ("mlp", "xgb"):
        h = g[g.learner == lrn].pivot(index="arm", columns="cohort", values="test_macro_F1")[COHORTS].map(lambda v: f"{v:.2f}")
        h.columns = [LABEL[c] for c in COHORTS]
        e = g[g.learner == lrn].groupby("arm").target_mean_entropy.mean().round(3)
        h.insert(0, "Mean target entropy (nats)", e.map(lambda v: f"{v:.3f}"))
        h.index = [arm_name(a) for a in h.index]
        write(f"TableS_grid_{lrn}", h.reset_index().rename(columns={"index": "Target"}),
              f"**Supplementary Table.** Outer-test Macro F1 (%) of every {'MLP' if lrn == 'mlp' else 'XGBoost'} target configuration "
              "(mean over partitions; target entropy averaged over all cohorts). S, uniform smoothing; D, distillation; EM, "
              "entropy-matched uniform smoothing. Descriptive only: no configuration was selected on these values.")
    s = pd.read_csv(SUM / "wp3_selected_configs.csv")
    s["label"] = s["arm"].map(arm_name) + " (" + s["times_selected"].astype(str) + ")"
    sel = s.sort_values("times_selected", ascending=False).groupby(["cohort", "learner", "family"])["label"] \
        .apply(lambda v: ", ".join(v)).unstack("family").reset_index()
    sel["cohort"] = sel["cohort"].map(LABEL); sel["learner"] = sel["learner"].map({"mlp": "MLP", "xgb": "XGBoost"})
    sel = sel.rename(columns={"cohort": "Cohort", "learner": "Student", "S": "Smoothing", "D": "Distillation", "EM": "Entropy-matched"})
    write("TableS_selected", sel, "**Supplementary Table.** Configurations chosen on the student-validation sets "
          "(number of outer folds × partitions in parentheses; 25 per family in five-partition cohorts, 5 otherwise).")
    if (SEC / "wp3_subgroups.csv").exists():
        sg = pd.read_csv(SEC / "wp3_subgroups.csv")
        sg["value"] = sg.apply(lambda r: f"{r.estimate:.1f} [{r.ci95_low:.1f}, {r.ci95_high:.1f}]", axis=1)
        for c in ("ptbxl_natural15160", "chapman_natural"):
            w = sg[sg.cohort == c].pivot_table(index=["subgroup", "n", "model"], columns="metric", values="value", aggfunc="first")
            w = w[["macro_F1", "macro_AUROC", "binary_sensitivity", "binary_specificity", "HYP_recall", "ECE_top_label_15bin"]]
            w = w.rename(columns=METRIC_NAME).reset_index()
            w["subgroup"] = w["subgroup"].map(SUBGROUP_NAME); w["model"] = w["model"].map(SEC_MODEL)
            w["order"] = w["subgroup"].map({v: i for i, v in enumerate(SUBGROUP_NAME.values())})
            w = w.sort_values(["order"], kind="stable").drop(columns="order")
            write(f"TableS_subgroups_{c}", w.rename(columns={"subgroup": "Subgroup", "n": "N", "model": "Model"}),
                  f"**Supplementary Table.** Sex and age subgroups, {LABEL[c]} (%, 95% bootstrap interval within subgroup, "
                  "B = 2,000). Binary sensitivity/specificity collapse the five-class prediction into NORM versus any other class.")
            for r in sg[sg.cohort == c].itertuples():
                NUM[f"sub_{c}_{r.subgroup}_{r.model}_{r.metric}"] = [round(r.estimate, 2), round(r.ci95_low, 2), round(r.ci95_high, 2), int(r.n)]
    if (SEC / "wp3_classwise.csv").exists():
        cw = pd.read_csv(SEC / "wp3_classwise.csv")
        keep = ["TabPFN teacher", "MLP hard", "MLP smooth (fixed)", "MLP distil (fixed)", "MLP distil (tuned)", "XGB hard"]
        a = cw[cw.model.isin(keep)].groupby(["cohort", "model", "class"])[["F1", "recall", "precision"]].mean().round(1)
        for c in COHORTS:
            w = a.loc[c].reset_index().pivot(index="model", columns="class", values="F1")[CLASSES].loc[keep]
            w.index = [SEC_MODEL[k] for k in w.index]
            write(f"TableS_classF1_{c}", w.reset_index().rename(columns={"index": "Model"}), f"**Supplementary Table.** Class-specific F1 (%), {LABEL[c]} "
                  "(mean over partitions).")
            for mdl in keep:
                for cl in CLASSES:
                    NUM[f"classF1_{c}_{mdl}_{cl}"] = float(a.loc[(c, mdl, cl), "F1"])
                    NUM[f"classRecall_{c}_{mdl}_{cl}"] = float(a.loc[(c, mdl, cl), "recall"])
    if (SEC / "wp3_full_metrics_per_seed.csv").exists():
        fm = pd.read_csv(SEC / "wp3_full_metrics_per_seed.csv")
        cols = ["macro_F1", "macro_AUROC", "macro_AUPRC", "NLL", "Brier", "ECE_top_label_15bin", "ECE_classwise_15bin",
                "binary_sensitivity", "binary_specificity", "HYP_recall"]
        out = fm.groupby(["cohort", "model"])[cols].mean().reset_index()
        missing = set(SEC_MODEL) - set(out.model)
        assert not missing, f"models missing from the secondary outputs: {missing}"
        out["order"] = out["model"].map({k: i for i, k in enumerate(SEC_MODEL)})
        out["corder"] = out["cohort"].map({k: i for i, k in enumerate(COHORTS)})
        out = out.sort_values(["corder", "order"]).drop(columns=["order", "corder"])
        for k in cols:
            out[k] = out[k].map(lambda v, d=(3 if k in ("NLL", "Brier") else 2): f"{v:.{d}f}")
        out["cohort"] = out["cohort"].map(LABEL); out["model"] = out["model"].map(SEC_MODEL)
        out = out.rename(columns={"cohort": "Cohort", "model": "Model", **METRIC_NAME})
        write("TableS_full_metrics", out, "**Supplementary Table.** All discrimination and calibration metrics (mean over "
              "partitions). AUROC and AUPRC are macro one-versus-rest; NLL, negative log-likelihood; Brier, multiclass Brier "
              "score; ECE, expected calibration error with 15 equal-width bins; sensitivity and specificity for NORM versus any "
              "other class, with abnormal as the positive class.")


def archived():
    """Earlier fixed-configuration analyses (archived scripts), with the original (v1) and revised (v2) features."""
    src = {"v1": {"3B": ROOT / "results/phase3b_repeated_evaluation_results.json",
                  "3C": ROOT / "experiments_phase3/phase3c_sensitivity/phase3c_sensitivity_results.json",
                  "3D": ROOT / "experiments_phase3/phase3d_full_cohort/phase3d_full_cohort_results.json"},
           "v2": {"3B": ROOT / "revizyon_v22/runs/3b_v2/ws/results/phase3b_repeated_evaluation_results.json",
                  "3C": ROOT / "revizyon_v22/runs/3c_v2/ws/experiments_phase3/phase3c_sensitivity/phase3c_sensitivity_results.json",
                  "3D": ROOT / "revizyon_v22/runs/3d_v2/ws/experiments_phase3/phase3d_full_cohort/phase3d_full_cohort_results.json"}}
    rows = []
    for v, f in src.items():
        b = json.load(open(f["3B"]))
        s, p = b["repetition_summary_across_5_seeds"], b["pairwise_aggregation_across_5_seeds"]
        rows.append({"Features": v, "Analysis": "PTB-XL quota-5000, 5 partitions (inner teacher 4 estimators)",
                     "Distilled / smoothed / hard (Macro F1, %)": f"{s['distill_mlp_128_64']['macro_f1']['mean']:.2f} / "
                     f"{s['smooth_mlp_128_64']['macro_f1']['mean']:.2f} / {s['hard_mlp_128_64']['macro_f1']['mean']:.2f}",
                     "Distilled − smoothed (pp)": f"{p['primary_distill_vs_smooth']['mean_delta_pp']:+.2f} ± {p['primary_distill_vs_smooth']['sd_delta_pp']:.2f} (mean ± SD of partitions)",
                     "Distilled − hard (pp)": f"{p['secondary_core_distill_vs_hard128']['mean_delta_pp']:+.2f} ± {p['secondary_core_distill_vs_hard128']['sd_delta_pp']:.2f}"})
        c = json.load(open(f["3C"]))["evaluation_b_retrained_sensitivity"]
        m, bc = c["metrics"], c["bootstrap_comparisons"]
        fmt = lambda k: f"{bc[k]['point_delta_pp']:+.2f} [{bc[k]['ci_95'][0]:+.2f}, {bc[k]['ci_95'][1]:+.2f}]"  # noqa: E731
        rows.append({"Features": v, "Analysis": "PTB-XL quota-5000 without HYP+STTC co-occurrence (N = 4,504), retrained, partition 42",
                     "Distilled / smoothed / hard (Macro F1, %)": f"{m['distill_mlp_128_64']['macro_f1']:.2f} / {m['smooth_mlp_128_64']['macro_f1']:.2f} / {m['hard_mlp_128_64']['macro_f1']:.2f}",
                     "Distilled − smoothed (pp)": fmt("distill_vs_smooth"), "Distilled − hard (pp)": fmt("distill_vs_hard128")})
        dd = json.load(open(f["3D"]))
        mm, bb = dd["model_metrics"], dd["bootstrap_paired_10000"]
        NUM[f"arch_{v}_3D_DS"] = bb["distilled_mlp_128_64_vs_smooth_mlp_128_64"]["point_diff"]
        NUM[f"arch_{v}_3D_DH"] = bb["distilled_mlp_128_64_vs_hard_mlp_128_64"]["point_diff"]
        NUM[f"arch_{v}_3B_DS"] = p["primary_distill_vs_smooth"]["mean_delta_pp"]
        rows.append({"Features": v, "Analysis": "PTB-XL eligible-15,160, partition 42 (different MLP procedure, see note)",
                     "Distilled / smoothed / hard (Macro F1, %)": f"{mm['distilled_mlp_128_64']['macro_f1']:.2f} / {mm['smooth_mlp_128_64']['macro_f1']:.2f} / {mm['hard_mlp_128_64']['macro_f1']:.2f}",
                     "Distilled − smoothed (pp)": f"{bb['distilled_mlp_128_64_vs_smooth_mlp_128_64']['point_diff']:+.2f} (p = {bb['distilled_mlp_128_64_vs_smooth_mlp_128_64']['p_value']:.3f})",
                     "Distilled − hard (pp)": f"{bb['distilled_mlp_128_64_vs_hard_mlp_128_64']['point_diff']:+.2f} (p = {bb['distilled_mlp_128_64_vs_hard_mlp_128_64']['p_value']:.3f})"})
    write("TableS_archived", pd.DataFrame(rows),
          "**Supplementary Table.** Earlier fixed-configuration analyses (ε = 0.15, α = 0.5, T = 1) re-run with the original (v1) "
          "and revised (v2) features; scripts, partitions and seeds unchanged. The eligible-15,160 analysis used a different MLP "
          "procedure from the other two (batch normalization and dropout 0.1, AdamW with weight decay 1e-4, at most 100 epochs, "
          "early stopping on validation cross-entropy, inner teacher folds not grouped by patient, which is equivalent here because "
          "each patient has one record); this was not stated correctly in the earlier report. The main analyses of the present "
          "paper use one procedure throughout.")


if __name__ == "__main__":
    table1(); table2()
    if (SUM / "wp3_contrasts_pooled.csv").exists():
        table4()
    if (COST / "nn_costs.csv").exists():
        table5()
    supplementary()
    archived()
    json.dump(NUM, open(TAB / "tables_numbers.json", "w"), indent=1)
    print(sorted(p.name for p in TAB.glob("*.md")))
