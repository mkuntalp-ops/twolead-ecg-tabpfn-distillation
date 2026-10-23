#!/usr/bin/env python3
"""
04_generate_ecg_lora_dataset.py

Generates the structured Electrophysiology JSON-to-Report Dataset (`train.jsonl` and `valid.jsonl`)
for local Gemma 3 270M LoRA Fine-Tuning.

Each sample pairs a structured JSON prompt (containing TabPFN predicted class, class probability vector,
confidence score, ventricular heart rate, Fridericia QTc, QRS duration, Lead I/II ST-segment deviation,
and binary rhythm regularity flags `irregular_rr_flag` and `wide_qrs_flag`) with a deterministic
4-stage Cardiology Triage Summary:
  1. Electrophysiological Findings
  2. Arrhythmia & Ischemia Risk Tier
  3. Recommended Clinical Actions
  4. Lead I/II Biometric Profile
"""

import json
import random
from pathlib import Path

BASE_DIR = Path("/Volumes/KIOXIA/Projects/gemma3-lora-other")
LORA_DATA_DIR = BASE_DIR / "data_ecg_clinical"
LORA_DATA_DIR.mkdir(parents=True, exist_ok=True)


def make_example(payload: dict, s1_findings: str, s2_risk: str, s3_actions: str, s4_profile: str) -> dict:
    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    user_prompt = (
        "You are an expert clinical cardiac electrophysiologist. Evaluate the following ambulatory two-lead "
        "(Lead I + Lead II) ECG tabular foundation model (TabPFN) diagnostic output and fiducial waveform telemetry:\n\n"
        f"```json\n{json_str}\n```\n\n"
        "Generate a structured 4-stage Cardiology Triage Summary with the exact headings:\n"
        "1. Electrophysiological Findings\n"
        "2. Arrhythmia & Ischemia Risk Tier\n"
        "3. Recommended Clinical Actions\n"
        "4. Lead I/II Biometric Profile"
    )
    assistant_text = (
        "CARDIOLOGY TRIAGE SUMMARY\n"
        "-------------------------\n"
        f"1. Electrophysiological Findings:\n{s1_findings}\n\n"
        f"2. Arrhythmia & Ischemia Risk Tier:\n{s2_risk}\n\n"
        f"3. Recommended Clinical Actions:\n{s3_actions}\n\n"
        f"4. Lead I/II Biometric Profile:\n{s4_profile}"
    )
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def generate_all_samples():
    rng = random.Random(42)
    samples = []

    # 1. AFIB/AFL (Atrial Fibrillation / Flutter) - 45 samples
    for _ in range(45):
        conf = round(rng.uniform(91.0, 99.6), 1)
        hr = round(rng.uniform(88.0, 148.0), 1)
        qrs = round(rng.uniform(76.0, 112.0), 1)
        qtc = round(rng.uniform(395.0, 465.0), 1)
        st1 = round(rng.uniform(-0.06, 0.04), 2)
        st2 = round(rng.uniform(-0.08, 0.05), 2)
        rem = round(100.0 - conf, 1)
        p_norm = round(rem * 0.15, 1)
        p_cd = round(rem * 0.35, 1)
        p_sttc = round(rem * 0.30, 1)
        p_hyp = round(max(0.0, rem - p_norm - p_cd - p_sttc), 1)
        payload = {
            "tabpfn_predicted_class": "AFIB/AFL (Atrial Fibrillation / Flutter)",
            "triage_decision": "Urgent Cardiology Escalation (Class 1)",
            "confidence_pct": conf,
            "class_probabilities_pct": {
                "NORM": p_norm, "AFIB/AFL": conf, "CD": p_cd, "STTC/MI": p_sttc, "HYP": p_hyp,
            },
            "heart_rate_bpm": hr,
            "qrs_duration_ms": qrs,
            "qtc_fridericia_ms": qtc,
            "st60_deviation_mV": {"Lead_I": st1, "Lead_II": st2},
            "rhythm_flags": {"irregular_rr_flag": True, "wide_qrs_flag": False},
        }
        samples.append(make_example(
            payload=payload,
            s1_findings=(
                f"Ambulatory Lead I/II telemetry demonstrates an irregularly irregular supraventricular rhythm "
                f"consistent with Atrial Fibrillation/Flutter ({conf:.1f}% TabPFN confidence) at a ventricular rate of "
                f"{hr:.1f} bpm. Beat-to-beat RR dispersion is markedly elevated (irregular_rr_flag = True) with absence "
                f"of organized sinus P waves and preserved narrow intraventricular conduction (QRS = {qrs:.1f} ms)."
            ),
            s2_risk=(
                f"High Risk — Urgent Cardiology Escalation. Uncontrolled atrial tachyarrhythmia ({hr:.1f} bpm) carries "
                f"immediate thromboembolic stroke risk and potential hemodynamic decompensation."
            ),
            s3_actions=(
                "- Obtain a confirmatory 12-lead resting ECG to verify atrial fibrillatory/flutter waves.\n"
                "- Assess hemodynamic stability, symptom onset duration, and CHA2DS2-VASc thromboembolic score.\n"
                "- Initiate ventricular rate control (e.g., beta-blocker or non-dihydropyridine calcium channel blocker) "
                "and evaluate oral anticoagulation per clinical guidelines."
            ),
            s4_profile=(
                f"TabPFN Diagnosis: AFIB/AFL ({conf:.1f}%) | Triage: Urgent Escalation | HR: {hr:.1f} bpm | "
                f"QRS: {qrs:.1f} ms | QTcF: {qtc:.1f} ms | ST60 (I/II): {st1:+.2f}/{st2:+.2f} mV | "
                f"Irregular RR: Present (+) | Wide QRS: Absent (-)."
            ),
        ))

    # 2. CD (Atrioventricular & Intraventricular Conduction Disturbance: LBBB/RBBB/AV Block) - 45 samples
    for _ in range(45):
        conf = round(rng.uniform(89.0, 99.4), 1)
        hr = round(rng.uniform(52.0, 86.0), 1)
        qrs = round(rng.uniform(124.0, 168.0), 1)
        qtc = round(rng.uniform(420.0, 485.0), 1)
        st1 = round(rng.uniform(-0.09, 0.06), 2)
        st2 = round(rng.uniform(-0.08, 0.07), 2)
        rem = round(100.0 - conf, 1)
        p_norm = round(rem * 0.15, 1)
        p_af = round(rem * 0.15, 1)
        p_sttc = round(rem * 0.45, 1)
        p_hyp = round(max(0.0, rem - p_norm - p_af - p_sttc), 1)
        payload = {
            "tabpfn_predicted_class": "CD (Conduction Disturbance: Bundle Branch / AV Block)",
            "triage_decision": "Urgent Cardiology Escalation (Class 1)",
            "confidence_pct": conf,
            "class_probabilities_pct": {
                "NORM": p_norm, "AFIB/AFL": p_af, "CD": conf, "STTC/MI": p_sttc, "HYP": p_hyp,
            },
            "heart_rate_bpm": hr,
            "qrs_duration_ms": qrs,
            "qtc_fridericia_ms": qtc,
            "st60_deviation_mV": {"Lead_I": st1, "Lead_II": st2},
            "rhythm_flags": {"irregular_rr_flag": False, "wide_qrs_flag": True},
        }
        samples.append(make_example(
            payload=payload,
            s1_findings=(
                f"Two-lead waveform analysis identifies significant intraventricular or atrioventricular conduction delay "
                f"({conf:.1f}% TabPFN confidence), characterized by prolonged QRS duration ({qrs:.1f} ms; wide_qrs_flag = True) "
                f"and regular RR intervals (HR = {hr:.1f} bpm). Secondary repolarization prolongation (QTcF = {qtc:.1f} ms) "
                f"is consistent with bundle branch block morphology."
            ),
            s2_risk=(
                f"High Risk — Urgent Cardiology Escalation. Wide-complex conduction block (QRS = {qrs:.1f} ms) requires "
                f"prompt exclusion of new-onset left bundle branch block, high-grade AV block, or underlying structural cardiomyopathy."
            ),
            s3_actions=(
                "- Perform an immediate 12-lead ECG (including precordial leads V1–V6) to differentiate LBBB vs. RBBB and quantify PR/AV conduction.\n"
                "- Compare with prior baseline ECGs to determine if the conduction block is acute or chronic.\n"
                "- Evaluate for syncope, presyncope, or ischemic chest symptoms and consider transthoracic echocardiography."
            ),
            s4_profile=(
                f"TabPFN Diagnosis: CD ({conf:.1f}%) | Triage: Urgent Escalation | HR: {hr:.1f} bpm | "
                f"QRS: {qrs:.1f} ms | QTcF: {qtc:.1f} ms | ST60 (I/II): {st1:+.2f}/{st2:+.2f} mV | "
                f"Irregular RR: Absent (-) | Wide QRS: Present (+)."
            ),
        ))

    # 3. STTC/MI (Ischemic ST-T Changes & Myocardial Infarction) - 45 samples
    for _ in range(45):
        conf = round(rng.uniform(88.0, 99.2), 1)
        hr = round(rng.uniform(64.0, 104.0), 1)
        qrs = round(rng.uniform(82.0, 110.0), 1)
        qtc = round(rng.uniform(415.0, 478.0), 1)
        st1 = round(rng.choice([rng.uniform(-0.22, -0.10), rng.uniform(0.11, 0.25)]), 2)
        st2 = round(rng.choice([rng.uniform(-0.24, -0.11), rng.uniform(0.12, 0.28)]), 2)
        rem = round(100.0 - conf, 1)
        p_norm = round(rem * 0.15, 1)
        p_af = round(rem * 0.15, 1)
        p_cd = round(rem * 0.30, 1)
        p_hyp = round(max(0.0, rem - p_norm - p_af - p_cd), 1)
        payload = {
            "tabpfn_predicted_class": "STTC/MI (Ischemic ST-T Abnormality / Myocardial Infarction)",
            "triage_decision": "Urgent Cardiology Escalation (Class 1)",
            "confidence_pct": conf,
            "class_probabilities_pct": {
                "NORM": p_norm, "AFIB/AFL": p_af, "CD": p_cd, "STTC/MI": conf, "HYP": p_hyp,
            },
            "heart_rate_bpm": hr,
            "qrs_duration_ms": qrs,
            "qtc_fridericia_ms": qtc,
            "st60_deviation_mV": {"Lead_I": st1, "Lead_II": st2},
            "rhythm_flags": {"irregular_rr_flag": False, "wide_qrs_flag": False},
        }
        samples.append(make_example(
            payload=payload,
            s1_findings=(
                f"Ambulatory Lead I/II morphology indicates ischemic ST-T segment deviation or myocardial infarction signature "
                f"({conf:.1f}% TabPFN confidence), with J+60 ms ST-segment shifts of {st1:+.2f} mV in Lead I and {st2:+.2f} mV in Lead II. "
                f"Sinus rhythm is regular (HR = {hr:.1f} bpm, QRS = {qrs:.1f} ms) with QTcF = {qtc:.1f} ms."
            ),
            s2_risk=(
                f"High Risk — Urgent Cardiology Escalation. Significant frontal-plane ST-T deviation in Leads I/II raises concern "
                f"for acute inferior or lateral myocardial ischemia/injury."
            ),
            s3_actions=(
                "- Acquire a stat 12-lead ECG immediately to evaluate contiguous precordial (V1–V6) and limb lead ST elevation/depression.\n"
                "- Draw serial high-sensitivity cardiac troponins (hs-cTn) and assess for angina or anginal equivalents.\n"
                "- Note that two-lead ambulatory monitoring has spatial blind spots for isolated anterior wall ischemia; full 12-lead correlation is mandatory."
            ),
            s4_profile=(
                f"TabPFN Diagnosis: STTC/MI ({conf:.1f}%) | Triage: Urgent Escalation | HR: {hr:.1f} bpm | "
                f"QRS: {qrs:.1f} ms | QTcF: {qtc:.1f} ms | ST60 (I/II): {st1:+.2f}/{st2:+.2f} mV | "
                f"Irregular RR: Absent (-) | Wide QRS: Absent (-)."
            ),
        ))

    # 4. HYP (Ventricular / Atrial Hypertrophy) - 35 samples
    for _ in range(35):
        conf = round(rng.uniform(86.0, 98.5), 1)
        hr = round(rng.uniform(58.0, 92.0), 1)
        qrs = round(rng.uniform(92.0, 114.0), 1)
        qtc = round(rng.uniform(410.0, 460.0), 1)
        st1 = round(rng.uniform(-0.12, -0.04), 2)
        st2 = round(rng.uniform(-0.10, -0.03), 2)
        rem = round(100.0 - conf, 1)
        p_norm = round(rem * 0.20, 1)
        p_af = round(rem * 0.10, 1)
        p_cd = round(rem * 0.25, 1)
        p_sttc = round(max(0.0, rem - p_norm - p_af - p_cd), 1)
        payload = {
            "tabpfn_predicted_class": "HYP (Ventricular / Atrial Hypertrophy)",
            "triage_decision": "Urgent Cardiology Escalation (Class 1)",
            "confidence_pct": conf,
            "class_probabilities_pct": {
                "NORM": p_norm, "AFIB/AFL": p_af, "CD": p_cd, "STTC/MI": p_sttc, "HYP": conf,
            },
            "heart_rate_bpm": hr,
            "qrs_duration_ms": qrs,
            "qtc_fridericia_ms": qtc,
            "st60_deviation_mV": {"Lead_I": st1, "Lead_II": st2},
            "rhythm_flags": {"irregular_rr_flag": False, "wide_qrs_flag": False},
        }
        samples.append(make_example(
            payload=payload,
            s1_findings=(
                f"Quantitative Lead I/II feature profile indicates cardiac chamber hypertrophy ({conf:.1f}% TabPFN confidence) "
                f"with augmented R-wave voltage and secondary repolarization strain (ST60 = {st1:+.2f} mV in Lead I, {st2:+.2f} mV in Lead II). "
                f"Underlying rhythm is regular at {hr:.1f} bpm with QRS duration of {qrs:.1f} ms."
            ),
            s2_risk=(
                f"Moderate-to-High Risk — Cardiology Escalation. High-voltage QRS complexes with secondary ST-T strain pattern "
                f"suggest chronic pressure overload (e.g., systemic hypertension or valvular heart disease)."
            ),
            s3_actions=(
                "- Obtain a 12-lead ECG to quantify Sokolow-Lyon and Cornell voltage criteria across precordial leads.\n"
                "- Schedule transthoracic echocardiography (TTE) to measure interventricular septal and posterior wall thickness.\n"
                "- Optimize blood pressure management and longitudinal outpatient cardiology follow-up."
            ),
            s4_profile=(
                f"TabPFN Diagnosis: HYP ({conf:.1f}%) | Triage: Urgent Escalation | HR: {hr:.1f} bpm | "
                f"QRS: {qrs:.1f} ms | QTcF: {qtc:.1f} ms | ST60 (I/II): {st1:+.2f}/{st2:+.2f} mV | "
                f"Irregular RR: Absent (-) | Wide QRS: Absent (-)."
            ),
        ))

    # 5. NORM (Normal Sinus Rhythm / Outpatient Reassurance) - 45 samples
    for _ in range(45):
        conf = round(rng.uniform(92.0, 99.8), 1)
        hr = round(rng.uniform(60.0, 88.0), 1)
        qrs = round(rng.uniform(74.0, 98.0), 1)
        qtc = round(rng.uniform(380.0, 430.0), 1)
        st1 = round(rng.uniform(-0.03, 0.03), 2)
        st2 = round(rng.uniform(-0.03, 0.04), 2)
        rem = round(100.0 - conf, 1)
        p_af = round(rem * 0.15, 1)
        p_cd = round(rem * 0.25, 1)
        p_sttc = round(rem * 0.35, 1)
        p_hyp = round(max(0.0, rem - p_af - p_cd - p_sttc), 1)
        payload = {
            "tabpfn_predicted_class": "NORM (Normal Sinus Rhythm)",
            "triage_decision": "Outpatient Reassurance (Class 0)",
            "confidence_pct": conf,
            "class_probabilities_pct": {
                "NORM": conf, "AFIB/AFL": p_af, "CD": p_cd, "STTC/MI": p_sttc, "HYP": p_hyp,
            },
            "heart_rate_bpm": hr,
            "qrs_duration_ms": qrs,
            "qtc_fridericia_ms": qtc,
            "st60_deviation_mV": {"Lead_I": st1, "Lead_II": st2},
            "rhythm_flags": {"irregular_rr_flag": False, "wide_qrs_flag": False},
        }
        samples.append(make_example(
            payload=payload,
            s1_findings=(
                f"Ambulatory two-lead (Lead I + II) recording demonstrates physiological Normal Sinus Rhythm "
                f"({conf:.1f}% TabPFN confidence) at {hr:.1f} bpm. Atrioventricular and intraventricular conduction "
                f"intervals are normal (QRS = {qrs:.1f} ms, QTcF = {qtc:.1f} ms) with isoelectric ST segments "
                f"({st1:+.2f}/{st2:+.2f} mV) and regular beat-to-beat RR periodicity."
            ),
            s2_risk=(
                f"Low Risk — Outpatient Reassurance. No electrophysiological evidence of atrial tachyarrhythmia, "
                f"bundle branch block, or acute frontal-plane ST-T deviation."
            ),
            s3_actions=(
                "- Routine outpatient preventive care; no urgent cardiology escalation is indicated.\n"
                "- If the patient experiences paroxysmal palpitations, chest pain, or dizziness, repeat ECG recording during symptoms."
            ),
            s4_profile=(
                f"TabPFN Diagnosis: NORM ({conf:.1f}%) | Triage: Outpatient Reassurance | HR: {hr:.1f} bpm | "
                f"QRS: {qrs:.1f} ms | QTcF: {qtc:.1f} ms | ST60 (I/II): {st1:+.2f}/{st2:+.2f} mV | "
                f"Irregular RR: Absent (-) | Wide QRS: Absent (-)."
            ),
        ))

    rng.shuffle(samples)
    n_train = int(len(samples) * 0.85)
    train_samples = samples[:n_train]
    valid_samples = samples[n_train:]

    with open(LORA_DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(LORA_DATA_DIR / "valid.jsonl", "w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated ECG LoRA clinical reporting dataset: {len(train_samples)} train, {len(valid_samples)} valid samples in {LORA_DATA_DIR}")


if __name__ == "__main__":
    generate_all_samples()
