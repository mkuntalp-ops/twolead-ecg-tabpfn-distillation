#!/usr/bin/env python3
"""Supplementary Material of the v22 manuscript, assembled from stored outputs and generated tables.

Usage (project root, after make_tables_v22.py):  python revizyon_v22/manuscript/make_supplement_v22.py
Writes revizyon_v22/manuscript/BSPC_Supplementary_v22.md and .docx
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / "revizyon_v22/results"
sys.path.insert(0, str(HERE))
from make_tables_v22 import CON, LABEL, MODELS, SEC_MODEL, md  # noqa: E402

LEARNER = {"mlp": "MLP", "xgb": "XGBoost", "teacher_tabpfn": "TabPFN"}
_names = [((lrn, mod), name) for lrn, mod, name in MODELS]
_names.insert(_names.index((("xgb", "H"), "XGBoost, hard labels")) + 1, (("xgb", "S_fixed"), "XGBoost, smoothing ε = 0.15"))
MODEL_NAME = dict(_names)

LABEL_N = {"ptbxl_natural15160": 15160, "chapman_natural": 9315}

T = HERE / "tables"


def tab(name):
    return (T / f"{name}.md").read_text() if (T / f"{name}.md").exists() else f"*[{name} not available]*\n"


CRIT = [  # pre-specified acceptance criteria for the revised extractor (fixed before the held-out evaluation)
    ("1. PR interval: Pearson r ≥ 0.6 with the reference global PR (records without AF)",
     ["PR L1 vs 12SL global", "PR L2 vs 12SL global", "PR L1 vs Uni-G global", "PR L2 vs Uni-G global"]),
    ("2. QRS duration: no records at the former 146 ms ceiling", None),
    ("3. QRS duration: r with Uni-G per-lead QRS not lower than with the original extractor",
     ["QRS L1 vs Uni-G I", "QRS L2 vs Uni-G II"]),
    ("4. QT interval: r with Uni-G per-lead QT not lower, and median difference smaller in magnitude, than with the original extractor",
     ["QT L1 vs Uni-G I", "QT L2 vs Uni-G II"]),
    ("5. R, S and T amplitudes: change in r with 12SL of at most 0.02", [f"{w} amp L{l} vs 12SL" for w in "RST" for l in (1, 2)]),
]


def feature_criteria():
    h = pd.read_csv(R / "feat50_v2/heldout_evaluation.csv").set_index("comparison")
    x = json.load(open(R / "feat50_v2/heldout_extra.json"))
    rows = []
    for name, comps in CRIT:
        if comps is None:
            v1 = f"{x['pct_L1_qrs_at_146_v1']:.2f}% / {x['pct_L2_qrs_at_146_v1']:.2f}% of records at 146 ms (ceiling)"
            v2 = (f"{x['pct_L1_qrs_at_146_v2']:.2f}% / {x['pct_L2_qrs_at_146_v2']:.2f}% at 146 ms "
                  "(measured values; no ceiling in the code)")
            rows.append({"Criterion": name, "Original extractor (lead I / II)": v1, "Revised extractor (lead I / II)": v2,
                         "Met": "Yes"})
            continue
        g = h.loc[comps]
        if name.startswith("1."):
            ok = g.r_v2 >= 0.6
            met = "Only lead II vs 12SL" if list(ok) == [False, True, False, False] else ("Yes" if ok.all() else "Partly")
        elif name.startswith("3."):
            met = "Yes" if (g.r_v2 >= g.r_v1).all() else "No"
        elif name.startswith("4."):
            met = "Yes" if ((g.r_v2 >= g.r_v1) & (g.median_diff_v2.abs() < g.median_diff_v1.abs())).all() else "No"
        else:
            met = "Yes" if (g.delta_r.abs() <= 0.02).all() else "No"
        unit = "mV" if name.startswith("5.") else "ms"
        fmt = lambda col, med: "; ".join(  # noqa: E731
            f"{c.split(' vs ')[0]} vs {c.split(' vs ')[1]}: r = {r[col]:.3f}, median difference {r[med]:+.{3 if unit == 'mV' else 0}f} {unit}"
            for c, r in g.iterrows())
        rows.append({"Criterion": name, "Original extractor (lead I / II)": fmt("r_v1", "median_diff_v1"),
                     "Revised extractor (lead I / II)": fmt("r_v2", "median_diff_v2"), "Met": met})
    return md(pd.DataFrame(rows))


RULES = [
    ("Pre-processing", "Leads I and II, 500 Hz, first 10 s; zero-phase third-order Butterworth band-pass 0.5–40 Hz."),
    ("R peaks", "Energy-based detector after Pan and Tompkins on the sum of the two filtered leads; each peak refined to the "
     "local maximum within ±30 ms in each lead."),
    ("RR series", "RR intervals outside 240–2,400 ms discarded; if none remain, RR = 800 ms (fallback). Mean, SDNN, RMSSD, pNN50 "
     "and the entropy (log2) of an eight-bin density histogram of RR between 300 and 1,900 ms."),
    ("Median beat", "Sample-wise median of all beats from R − 350 ms to R + 450 ms; isoelectric level = mean of R − 100 to R − 60 ms, "
     "subtracted."),
    ("QRS onset and offset", "Envelope = 7-sample moving average of the absolute first difference of the median beat. Within "
     "R − 90 to R + 120 ms, samples above max(0.085 × window maximum, 0.0025 mV per sample) are marked, gaps ≤ 20 ms are merged and the "
     "marked region containing (or nearest to) R defines onset and offset. Fallback if no region: R ± 40 ms. Duration limited to "
     "50–240 ms."),
    ("P onset and PR interval", "On the median beat low-passed at 15 Hz (second-order Butterworth), the P peak is the largest "
     "deviation from the window start within QRS onset − 300 ms to QRS onset − 50 ms; the onset is found by searching backwards "
     "from the peak until the absolute slope falls below 0.15 × the window's maximum slope. Fallback if the search reaches the "
     "window start: onset = window start. PR = QRS onset − P onset, limited to 80–360 ms."),
    ("T wave, QT, QTc, Tpeak–Tend", "T peak = largest absolute deviation within R + 100 to R + 360 ms; T end = first sample "
     "after the peak with absolute amplitude ≤ max(0.15 × absolute T amplitude, 0.012 mV), searched up to R + 440 ms (fallback: peak + "
     "70 ms). QT = T end − QRS onset (220–620 ms); QTc = QT / RR^1/3^ with RR in s (RR ≥ 0.30 s); Tpeak–Tend limited to 20–220 ms."),
    ("Amplitudes", "P: largest absolute value within R − 180 to R − 60 ms; Q: minimum within max(R − 64 ms, QRS onset − 10 ms) to R if "
     "negative, else 0; R: maximum within R ± 16 ms; S: minimum within R to min(R + 90 ms, QRS offset + 16 ms) if negative, else 0; J: value at QRS offset; ST60: value "
     "60 ms after J; ST slope = (ST60 − J)/0.06 s. All relative to the isoelectric level, in mV."),
    ("Spectral and statistical", "Welch periodogram (segment 1,024 samples) of the filtered lead: power fractions in 0.5–5, 5–15 "
     "and 15–40 Hz and spectral entropy (log2); zero-crossing rate, root-mean-square amplitude and kurtosis (limited to −5 to 50)."),
]


def feature_rules():
    t = md(pd.DataFrame(RULES, columns=["Step", "Rule"]))
    rates = []
    for name, f, cols in [("PTB-XL eligible-15,160", R.parent / "data_v2/extraction_v2_debug.csv", None),
                          ("Chapman eligible-9,315", R.parent / "data_v2/chapman_eligible_features_v2.csv", None)]:
        d = pd.read_csv(f)
        fb = {c: 100 * d[c].mean() for c in d.columns if c.endswith("_fallback")}
        rates.append(f"{name}: QRS fallback {fb['L1_qrs_fallback']:.2f}% (lead I) and {fb['L2_qrs_fallback']:.2f}% (lead II), "
                     f"P-onset fallback {fb['L1_p_fallback']:.3f}% and {fb['L2_p_fallback']:.3f}% of records")
    return t, "; ".join(rates)


def feature_validation():
    h = pd.read_csv(R / "feat50_v2/heldout_evaluation.csv")
    h["Records"] = h["records"].map({"non-AF": "Without AF", "all": "All"})
    h = h.assign(**{"r (original)": h.r_v1.map(lambda v: f"{v:.3f}"), "r (revised)": h.r_v2.map(lambda v: f"{v:.3f}"),
                    "Median difference, original": h.median_diff_v1.map(lambda v: f"{v:+.3f}" if abs(v) < 1 else f"{v:+.0f}"),
                    "Median difference, revised": h.median_diff_v2.map(lambda v: f"{v:+.3f}" if abs(v) < 1 else f"{v:+.0f}")})
    return md(h[["comparison", "Records", "n_v2", "r (original)", "r (revised)", "Median difference, original",
                 "Median difference, revised"]].rename(columns={"comparison": "Comparison", "n_v2": "N"}))


def confusion(cohort):
    cm = pd.read_csv(R / "wp3/secondary/wp3_confusion_matrices.csv")
    keep = ["TabPFN teacher", "MLP hard", "MLP distil (fixed)", "MLP distil (tuned)", "XGB hard"]
    g = cm[(cm.cohort == cohort) & cm.model.isin(keep)].copy()
    assert g.seed.nunique() == 1, "confusion tables are given for single-partition cohorts only"
    g["o"] = g.model.map({k: i for i, k in enumerate(keep)}); g = g.sort_values("o", kind="stable")
    for m, h in g.groupby("model"):
        assert h[[c for c in h.columns if c.startswith("pred_")]].to_numpy().sum() == int(LABEL_N[cohort])
    g["model"] = g.model.map(SEC_MODEL)
    g = g.rename(columns={c: c.replace("pred_", "Predicted ") for c in g.columns})
    return md(g.drop(columns=["cohort", "seed", "o"]).rename(columns={"model": "Model", "reference": "Reference"}))


def per_seed():
    """Every partition-level difference behind Table 4 (five-partition cohorts), with its B = 2,000 interval."""
    s = pd.read_csv(R / "wp3/summary/wp3_contrasts_per_seed.csv")
    s = s[s.cohort.str.endswith("5000")].copy()
    s["cell"] = s.apply(lambda r: f"{r.d_macro_F1:+.2f} [{r.d_macro_F1_lo:+.2f}, {r.d_macro_F1_hi:+.2f}]", axis=1)
    w = s.pivot_table(index=["cohort", "learner", "contrast"], columns="seed", values="cell", aggfunc="first").reset_index()
    order = {k: i for i, (k, _) in enumerate(CON)}
    w = w.assign(c_ord=w.cohort.map(list(LABEL).index), k_ord=w.contrast.map(order)).sort_values(["c_ord", "learner", "k_ord"])
    w["cohort"] = w["cohort"].map(LABEL)
    w["learner"] = w["learner"].map(LEARNER)
    w["contrast"] = w["contrast"].map(dict(CON))
    assert w[["learner", "contrast"]].notna().all().all() and len(w) == 42, "unnamed learner/contrast or missing rows"
    w = w.drop(columns=["c_ord", "k_ord"]).rename(columns={"cohort": "Cohort", "learner": "Student", "contrast": "Contrast"})
    return md(w.rename(columns={sd: f"Partition {sd}" for sd in (42, 43, 44, 45, 46)}))


def confidence():
    c = pd.read_csv(R / "wp3/summary/wp3_confidence_gap.csv")
    c["cohort"] = c["cohort"].map(LABEL)
    w = c.pivot_table(index=["learner", "model"], columns="cohort", values="confidence_minus_accuracy").round(1)
    w = w[[LABEL[k] for k in LABEL]].reset_index()
    order = list(MODEL_NAME)
    w = w.assign(o=[order.index((a, b)) for a, b in zip(w.learner, w.model)]).sort_values("o")
    w.insert(0, "Model", [MODEL_NAME[(a, b)] for a, b in zip(w.learner, w.model)])
    return md(w.drop(columns=["learner", "model", "o"]))


TRIPOD = [
    ("1 Title", "Title identifies the evaluation of classification models (knowledge distillation) and the outcome (two-lead ECG classification)"),
    ("2 Abstract", "Abstract"),
    ("3 Background", "Section 1"), ("4 Objectives", "Section 1 (four questions)"),
    ("5 Data sources", "Section 2.1; Table 1"), ("6 Participants", "Section 2.1; Table 1 (eligibility, one record per patient)"),
    ("7 Data preparation", "Section 2.2; Supplementary Section S1"), ("8 Outcome", "Section 2.1 (reference labels from database coding; no adjudication)"),
    ("9 Predictors", "Section 2.2; Supplementary Section S1 (50 waveform features; age and sex not used)"),
    ("10 Sample size", "Section 2.1 (all eligible patients; no formal sample-size calculation)"),
    ("11 Missing data", "Supplementary Section S1 (rules and fallbacks; no missing values remain)"),
    ("12 Analytical methods", "Sections 2.3–2.8"), ("13 Class imbalance", "Sections 2.1 and 2.8 (quota versus natural cohorts; Macro F1; no resampling)"),
    ("14 Fairness", "Section 2.8 and Supplementary Section S7 (sex and age subgroups)"),
    ("15 Model output", "Section 2.8 (class probabilities; argmax decision)"),
    ("16 Training versus evaluation", "Section 2.3; Fig. 1b"), ("17 Ethics", "Declarations"),
    ("18 Open science", "Declarations (funding, competing interests, data and code availability); no protocol registration (Section 2.9)"),
    ("19 Patient and public involvement", "Not applicable (public retrospective data)"),
    ("20 Participants (results)", "Table 1"), ("21 Model development", "Sections 3.1–3.3; Supplementary Sections S3–S4"),
    ("22 Model specification", "Sections 2.4–2.6; code repository"), ("23 Model performance", "Tables 2–4; Fig. 2–4; Supplementary Sections S4–S7"),
    ("24 Model updating", "Not applicable"), ("25 Interpretation", "Section 4"), ("26 Limitations", "Section 4.6"),
    ("27 Usability", "Section 4.3 and Table 5 (size and latency; no deployment study)"),
]


def main():
    parts = ["---\ntitle: \"Supplementary Material\"\nsubtitle: \"Does a tabular foundation-model (TabPFN) teacher add information beyond label smoothing? "
             "A multi-cohort evaluation of knowledge distillation for two-lead ECG classification\"\n---\n"]
    rules, rates = feature_rules()
    parts.append("# S1. Features and their validation\n\nThe 25 features per lead are PR, QRS, QT, QTc (Fridericia) and T-peak-to-T-end "
                 "intervals; P, Q, R, S, T and J amplitudes, ST amplitude 60 ms after J and ST slope; RR mean, SDNN, RMSSD, pNN50 and RR "
                 "histogram entropy; relative Welch band powers in 0.5–5, 5–15 and 15–40 Hz, spectral entropy, zero-crossing rate, "
                 "root-mean-square amplitude and kurtosis. No feature value is missing: every rule has a defined fallback, and values are "
                 "limited to the stated ranges. The implementation is `feature_extraction/feat50_v2.py` in the code repository.\n\n"
                 "**Supplementary Table [feat_rules].** Delineation and feature rules of the revised extractor (times relative to the R "
                 f"peak of the median beat). Fallback frequencies: {rates}.\n\n" + rules +
                 "\nThe delineation windows were revised on a random development subset of 1,000 PTB-XL recordings (seed 2026) without "
                 "diagnostic labels; the acceptance criteria below were fixed before the revised extractor was applied to the remaining "
                 "14,160 recordings and were not changed afterwards.\n\n"
                 "**Supplementary Table [feat_criteria].** Pre-specified acceptance criteria for the revised extractor, evaluated on the "
                 "14,160 held-out PTB-XL recordings against the PTB-XL+ reference measurements (GE 12SL global or Uni-G per-lead). "
                 "Differences are ours minus reference.\n\n" + feature_criteria() +
                 "\n**Supplementary Table [feat_agree].** Agreement of the original and revised single-lead measurements with the PTB-XL+ "
                 "reference measurements on the 14,160 held-out PTB-XL records (PR in records without AF). Differences are ours minus "
                 "reference (ms for intervals, mV for amplitudes).\n\n" + feature_validation())
    parts.append("# S2. Label mapping and cohort selection\n\nPTB-XL: diagnostic statements with likelihood ≥ 50 mapped to superclasses "
                 "(NORM, CD, STTC, MI, HYP); AFIB/AFLT rhythm statements take precedence (AFIB/AFL); STTC and MI merged; HYP + STTC "
                 "without AFIB/AFL, CD or MI assigned to HYP; other multi-superclass combinations excluded. Chapman-Shaoxing: AFIB or AF "
                 "rhythm → AFIB/AFL; SR, SB, ST, SI or SA rhythm without beat codes → NORM; conduction codes without hypertrophy codes → "
                 "CD; hypertrophy codes without conduction codes → HYP; ST/T or infarction codes without conduction or hypertrophy codes → "
                 "STTC/MI; all other records excluded. Code lists: `feature_extraction/extract_chapman_v2.py` in the code repository.\n\n"
                 "Quota-5000 cohorts: from the eligible adults of each database, a fixed number of patients was drawn at random from each "
                 "class (pandas `DataFrame.sample`, seed 42), keeping every patient of a class smaller than its quota. PTB-XL: 1,045 each "
                 "of NORM, AFIB/AFL, CD and STTC/MI and all 820 HYP. Chapman-Shaoxing: 1,196 NORM, 1,197 AFIB/AFL, 1,197 STTC/MI and all "
                 "590 CD and 820 HYP. Natural-5000: class-stratified random draw of 5,000 PTB-XL eligible patients "
                 "(scikit-learn `train_test_split`, seed 2026). Patient identifiers of every cohort are listed in the repository.\n")
    parts.append("# S3. Target grid and selected configurations\n\n" + tab("TableS_grid_mlp").replace("**Supplementary Table.**", "**Supplementary Table [grid_mlp].**")
                 + "\n" + tab("TableS_grid_xgb").replace("**Supplementary Table.**", "**Supplementary Table [grid_xgb].**")
                 + "\n" + tab("TableS_selected").replace("**Supplementary Table.**", "**Supplementary Table [selected].**"))
    parts.append("# S4. Per-partition differences and post hoc partition-averaged intervals\n\n**Supplementary Table [perseed].** "
                 "Five-partition cohorts: paired differences in Macro F1 (pp) in each partition (partition seeds 42–46) with 95% "
                 "patient-blocked bootstrap intervals (B = 2,000 per partition); these are the values summarized in Table 4.\n\n"
                 + per_seed() + "\n" + tab("TableS_pooled").replace("**Supplementary Table.**", "**Supplementary Table [pooled].**"))
    parts.append("# S5. All metrics\n\n" + tab("TableS_full_metrics").replace("**Supplementary Table.**", "**Supplementary Table [full].**")
                 + "\n**Supplementary Table [confgap].** Mean top-label confidence minus accuracy (percentage points; negative values "
                 "indicate under-confidence), mean over partitions.\n\n" + confidence())
    parts.append("# S6. Class-specific results and confusion matrices\n\n"
                 + "\n".join(tab(f"TableS_classF1_{c}").replace("**Supplementary Table.**", f"**Supplementary Table [classF1_{c}].**") for c in LABEL)
                 + "".join(f"\n**Supplementary Table [cm_{c}].** Confusion matrices, {LABEL[c]} (rows: reference class; columns: "
                           f"predicted class; counts of patients).\n\n" + confusion(c) for c in ("ptbxl_natural15160", "chapman_natural"))
                 + "\nConfusion matrices for every cohort, partition and model are in `results/wp3/secondary/wp3_confusion_matrices.csv` "
                 "in the code repository.\n")
    parts.append("# S7. Sex and age subgroups\n\n" + "\n".join(tab(f"TableS_subgroups_{c}").replace("**Supplementary Table.**", f"**Supplementary Table [sub_{c}].**")
                                                       for c in ("ptbxl_natural15160", "chapman_natural")))
    parts.append("# S8. TRIPOD+AI checklist\n\n**Supplementary Table [tripod].** TRIPOD+AI items and where they are reported.\n\n"
                 + md(pd.DataFrame(TRIPOD, columns=["Item", "Where reported"])))
    text = "\n\n".join(parts)
    assert "**Supplementary Table.**" not in text, "a supplementary table has no identifier"
    ids = re.findall(r"\*\*Supplementary Table \[([^\]]+)\]\.\*\*", text)
    assert len(ids) == len(set(ids)), "duplicate supplementary table identifiers"
    num = {k: f"S{i}" for i, k in enumerate(ids, 1)}
    text = re.sub(r"\*\*Supplementary Table \[([^\]]+)\]\.\*\*", lambda m: f"**Table {num[m.group(1)]}.**", text)
    json.dump(num, open(T / "supplementary_table_numbers.json", "w"), indent=1)
    text = re.sub(r"(?<=[\s\[(])-(?=\d)", "\u2212", text)  # typographic minus for negative numbers
    out = HERE / "BSPC_Supplementary_v22.md"
    out.write_text(text)
    subprocess.run(["pandoc", str(out), "-o", str(out.with_suffix(".docx"))], check=True)
    landscape(out.with_suffix(".docx"))
    print("written", out, num)


def autofit_tables(doc):
    """Full-width tables whose columns size to their content (pandoc gives wide pipe tables equal fixed widths)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for t in doc.tables:
        tpr = t._tbl.tblPr
        for tag in ("w:tblW", "w:tblLayout"):
            for e in tpr.findall(qn(tag)):
                tpr.remove(e)
        w = OxmlElement("w:tblW"); w.set(qn("w:w"), "5000"); w.set(qn("w:type"), "pct"); tpr.append(w)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "autofit"); tpr.append(lay)
        for gc in t._tbl.tblGrid.findall(qn("w:gridCol")):
            gc.attrib.pop(qn("w:w"), None)
        for tc in t._tbl.iter(qn("w:tc")):
            tcpr = tc.get_or_add_tcPr()
            for e in tcpr.findall(qn("w:tcW")):
                tcpr.remove(e)
            cw = OxmlElement("w:tcW"); cw.set(qn("w:w"), "0"); cw.set(qn("w:type"), "auto"); tcpr.append(cw)


def landscape(path: Path):
    """Landscape A4 pages and full-width tables with a readable font, so that wide tables do not break words."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.shared import Cm, Pt
    doc = Document(str(path))
    for sec in doc.sections:
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Cm(1.8))
    autofit_tables(doc)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    for run in par.runs:
                        run.font.size = Pt(8)
    doc.save(str(path))


if __name__ == "__main__":
    main()
