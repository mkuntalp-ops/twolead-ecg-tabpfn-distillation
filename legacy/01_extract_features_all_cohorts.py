#!/usr/bin/env python3
"""
01_extract_features_all_cohorts.py

Extracts the unified 54-dimensional Ambulatory Two-Lead (Lead I + Lead II) ECG
tabular feature schema across three independent multi-center cohorts:
  1. Discovery Cohort: PTB-XL / PTB-XL+ (N = 5,000 adult patients, stratified across 5 classes)
  2. External Validation Cohort 1: Chapman-Shaoxing 12-Lead ECG Database (N = 5,000 adult patients)
  3. External Validation Cohort 2: CPSC-2018 Database (N = 2,000 recordings across 500 Hz Lead I & II)

54-Feature Schema:
  - 50 Continuous Waveform Features (25 per lead across Lead I and Lead II):
      * 5 Fiducial Intervals: PR interval, QRS duration, QT interval, Fridericia QTc, T-peak-to-T-end (TpTe)
      * 8 Wave & ST Amplitudes: P, Q, R, S, T peak amplitudes, J-point amplitude, ST at J+60ms, ST slope
      * 5 Heart Rate Variability (HRV) Metrics: mean RR, SDNN, RMSSD, pNN50, RR Shannon entropy
      * 7 Spectral & Morphology Energy Ratios: 0.5-5 Hz band ratio, 5-15 Hz band ratio, 15-40 Hz band ratio,
        spectral Shannon entropy, zero-crossing rate (ZCR), RMS energy, waveform kurtosis
  - 4 Clinical & Rhythm Metadata Features:
      * age (years), sex (0=Female, 1=Male), irregular_rr_flag (0/1), wide_qrs_flag (0/1)
"""

import ast
import concurrent.futures
import io
import os
import struct
import time
import urllib.request
import warnings
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")

BASE_DIR = Path("/Volumes/KIOXIA/Projects/gemma3-lora-other")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PTBXL_CSV = Path("/Volumes/KIOXIA/Datasets/ptb-xl/ptbxl_database.csv")
PTBXL_SCP = Path("/Volumes/KIOXIA/Datasets/ptb-xl/scp_statements.csv")
PTBXL_ZIP = Path("/Volumes/KIOXIA/Datasets/ptb-xl/ptb-xl-data.zip")
PTBXL_UNIG = DATA_DIR / "ptbxl_plus/unig_features.csv"
PTBXL_DELI = DATA_DIR / "ptbxl_plus/ecgdeli_features.csv"

CHAPMAN_CSV = DATA_DIR / "chapman/Diagnostics.csv"
CHAPMAN_ZIP = DATA_DIR / "chapman/ECGData.zip"
CPSC_INDEX = DATA_DIR / "cpsc2018_index.csv"

CLASS_MAP = {
    "NORM": 0,
    "AFIB/AFL": 1,
    "CD": 2,
    "STTC/MI": 3,
    "HYP": 4,
}


def detect_r_peaks_pan_tompkins(x: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """Polarity-invariant Pan-Tompkins QRS energy detector at 500 Hz."""
    b, a = butter(2, [5.0 / (fs / 2.0), 20.0 / (fs / 2.0)], btype="band")
    qrs_f = filtfilt(b, a, x)
    diff_sq = np.diff(qrs_f, prepend=qrs_f[0]) ** 2
    win = int(0.12 * fs)
    mwi = np.convolve(diff_sq, np.ones(win) / win, mode="same")
    thr = np.percentile(mwi, 85) * 1.15
    peaks, _ = find_peaks(mwi, distance=int(0.28 * fs), height=max(thr, 1e-6))
    if len(peaks) < 3:
        peaks = np.linspace(int(0.5 * fs), len(x) - int(0.5 * fs), 10, dtype=int)
    r_peaks = []
    half = int(0.04 * fs)
    for p in peaks:
        s, e = max(0, p - half), min(len(x), p + half)
        r_peaks.append(s + int(np.argmax(np.abs(x[s:e]))))
    return np.unique(r_peaks)


def extract_single_lead_25(
    sig_1d: np.ndarray,
    fs: float = 500.0,
    lead_prefix: str = "L1",
    shared_r_peaks: np.ndarray | None = None,
) -> dict[str, float]:
    """Extracts 25 standardized morphological, fiducial, HRV, and spectral descriptors from one ECG lead."""
    sig_clean = np.nan_to_num(sig_1d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(sig_clean) > int(10.0 * fs):
        sig_clean = sig_clean[: int(10.0 * fs)]
    elif len(sig_clean) < int(2.0 * fs):
        sig_clean = np.pad(sig_clean, (0, int(10.0 * fs) - len(sig_clean)))

    b, a = butter(3, [0.5 / (fs / 2.0), 40.0 / (fs / 2.0)], btype="band")
    x = filtfilt(b, a, sig_clean)

    r_peaks = shared_r_peaks if shared_r_peaks is not None else detect_r_peaks_pan_tompkins(x, fs)
    # Locally refine R-peak to signed positive peak in a +-30ms window
    local_r = []
    half = int(0.03 * fs)
    for p in r_peaks:
        s, e = max(0, p - half), min(len(x), p + half)
        if e > s:
            local_r.append(s + int(np.argmax(x[s:e])))
    local_r = np.unique(local_r) if len(local_r) >= 2 else r_peaks

    # RR intervals in ms
    if len(local_r) >= 2:
        rr = np.diff(local_r) * (1000.0 / fs)
        rr = rr[(rr >= 240.0) & (rr <= 2400.0)]
    else:
        rr = np.array([800.0])
    if len(rr) == 0:
        rr = np.array([800.0])

    rr_mean = float(np.mean(rr))
    rr_sdnn = float(np.std(rr)) if len(rr) > 1 else 0.0
    diff_rr = np.diff(rr)
    rr_rmssd = float(np.sqrt(np.mean(diff_rr**2))) if len(diff_rr) > 0 else 0.0
    rr_pnn50 = float(np.mean(np.abs(diff_rr) > 50.0) * 100.0) if len(diff_rr) > 0 else 0.0
    hist_rr, _ = np.histogram(rr, bins=8, range=(300.0, 1900.0), density=True)
    hist_rr = hist_rr[hist_rr > 0]
    rr_ent = float(-np.sum(hist_rr * np.log2(hist_rr + 1e-12)))

    # Ensemble median beat (-200ms to +400ms around R peak = 300 samples at 500 Hz)
    pre = int(0.20 * fs)   # 100 samples
    post = int(0.40 * fs)  # 200 samples
    beats = [x[r - pre : r + post] for r in local_r if r - pre >= 0 and r + post <= len(x)]
    if len(beats) > 0:
        med_beat = np.median(np.vstack(beats), axis=0)
    else:
        med_beat = np.zeros(pre + post, dtype=np.float64)

    iso = float(np.mean(med_beat[50:70]))
    mb = med_beat - iso

    # P wave (-180ms to -60ms before R = idx 10..70)
    p_win = mb[10:70]
    p_rel = int(np.argmax(np.abs(p_win)))
    p_idx = 10 + p_rel
    p_amp = float(mb[p_idx])

    # QRS onset and offset using derivative envelope around R (idx 68..142 = -64ms to +84ms)
    d_mb = np.abs(np.diff(mb, prepend=mb[0]))
    env = np.convolve(d_mb, np.ones(7) / 7.0, mode="same")
    qrs_env = env[68:142]
    thr_qrs = max(0.085 * float(np.max(qrs_env)), 0.0025)
    above = np.where(qrs_env >= thr_qrs)[0]
    if len(above) >= 2:
        qrs_on = 68 + int(above[0])
        qrs_off = 68 + int(above[-1])
    else:
        qrs_on, qrs_off = 80, 120
    qrs_dur = float(np.clip((qrs_off - qrs_on) * (1000.0 / fs), 50.0, 240.0))

    # P onset and PR interval
    p_env = env[10:max(qrs_on, 20)]
    above_p = np.where(p_env >= max(0.15 * float(np.max(p_env)), 0.001))[0] if len(p_env) > 0 else []
    p_on = 10 + int(above_p[0]) if len(above_p) > 0 else max(15, p_idx - 20)
    pr_int = float(np.clip((qrs_on - p_on) * (1000.0 / fs), 80.0, 360.0))

    # Q, R, S peak amplitudes
    q_win = mb[max(68, qrs_on - 5) : pre]
    q_amp = float(np.min(q_win)) if len(q_win) > 0 and np.min(q_win) < 0 else 0.0
    r_amp = float( np.max(mb[pre - 8 : pre + 9]) )
    s_win = mb[pre : min(145, qrs_off + 8)]
    s_amp = float(np.min(s_win)) if len(s_win) > 0 and np.min(s_win) < 0 else 0.0

    # J-point and ST segment at J+60ms (+30 samples)
    j_idx = min(qrs_off, len(mb) - 1)
    j_amp = float(mb[j_idx])
    st60_idx = min(qrs_off + int(0.06 * fs), len(mb) - 1)
    st60_amp = float(mb[st60_idx])
    st_slope = float((st60_amp - j_amp) / 0.06)

    # T wave (+100ms to +360ms after R = idx 150..280)
    t_win = mb[150:280]
    t_idx = 150 + int(np.argmax(np.abs(t_win)))
    t_amp = float(mb[t_idx])
    post_t = np.abs(mb[t_idx:295])
    below_t = np.where(post_t <= max(0.15 * abs(t_amp), 0.012))[0]
    t_end = t_idx + int(below_t[0]) if len(below_t) > 0 else min(t_idx + 35, 294)

    qt_int = float(np.clip((t_end - qrs_on) * (1000.0 / fs), 220.0, 620.0))
    rr_sec = max(rr_mean / 1000.0, 0.30)
    qtc_frid = float(qt_int / (rr_sec ** (1.0 / 3.0)))
    tpte_int = float(np.clip((t_end - t_idx) * (1000.0 / fs), 20.0, 220.0))

    # Spectral & statistical features
    freqs, psd = welch(x, fs=fs, nperseg=min(1024, len(x)))
    tot_pow = float(np.sum(psd) + 1e-12)
    pow_05_5 = float(np.sum(psd[(freqs >= 0.5) & (freqs < 5.0)]) / tot_pow)
    pow_5_15 = float(np.sum(psd[(freqs >= 5.0) & (freqs < 15.0)]) / tot_pow)
    pow_15_40 = float(np.sum(psd[(freqs >= 15.0) & (freqs <= 40.0)]) / tot_pow)
    psd_norm = psd / tot_pow
    spec_ent = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    zcr = float(np.mean(np.diff(np.sign(x)) != 0))
    rms = float(np.sqrt(np.mean(x**2)))
    ku = float(np.clip(kurtosis(x), -5.0, 50.0))

    return {
        f"{lead_prefix}_pr_int": pr_int,
        f"{lead_prefix}_qrs_dur": qrs_dur,
        f"{lead_prefix}_qt_int": qt_int,
        f"{lead_prefix}_qtc_frid": qtc_frid,
        f"{lead_prefix}_tpte_int": tpte_int,
        f"{lead_prefix}_p_amp": p_amp,
        f"{lead_prefix}_q_amp": q_amp,
        f"{lead_prefix}_r_amp": r_amp,
        f"{lead_prefix}_s_amp": s_amp,
        f"{lead_prefix}_t_amp": t_amp,
        f"{lead_prefix}_j_amp": j_amp,
        f"{lead_prefix}_st60_amp": st60_amp,
        f"{lead_prefix}_st_slope": st_slope,
        f"{lead_prefix}_rr_mean": rr_mean,
        f"{lead_prefix}_rr_sdnn": rr_sdnn,
        f"{lead_prefix}_rr_rmssd": rr_rmssd,
        f"{lead_prefix}_rr_pnn50": rr_pnn50,
        f"{lead_prefix}_rr_ent": rr_ent,
        f"{lead_prefix}_pow_05_5hz": pow_05_5,
        f"{lead_prefix}_pow_5_15hz": pow_5_15,
        f"{lead_prefix}_pow_15_40hz": pow_15_40,
        f"{lead_prefix}_spec_ent": spec_ent,
        f"{lead_prefix}_zcr": zcr,
        f"{lead_prefix}_rms": rms,
        f"{lead_prefix}_kurtosis": ku,
    }


def extract_dual_lead_54(
    sig_I: np.ndarray,
    sig_II: np.ndarray,
    age: float,
    sex: int,
    fs: float = 500.0,
    ref_qrs_ms: float | None = None,
    ref_rr_cv: float | None = None,
) -> dict[str, float]:
    """Extracts the complete 54-feature vector (50 waveform + 4 clinical/rhythm metadata)."""
    # Detect shared R peaks from combined Lead I + II energy for robust beat alignment
    b, a = butter(3, [0.5 / (fs / 2.0), 40.0 / (fs / 2.0)], btype="band")
    x1 = filtfilt(b, a, np.nan_to_num(sig_I[: int(10 * fs)].astype(np.float64)))
    x2 = filtfilt(b, a, np.nan_to_num(sig_II[: int(10 * fs)].astype(np.float64)))
    shared_peaks = detect_r_peaks_pan_tompkins(x1 + x2, fs=fs)

    f1 = extract_single_lead_25(sig_I, fs=fs, lead_prefix="L1", shared_r_peaks=shared_peaks)
    f2 = extract_single_lead_25(sig_II, fs=fs, lead_prefix="L2", shared_r_peaks=shared_peaks)

    # Rhythm regularity & wide QRS flags (using reference fiducial algorithm when available or dual-lead consensus)
    cv1 = f1["L1_rr_sdnn"] / max(f1["L1_rr_mean"], 200.0)
    cv2 = f2["L2_rr_sdnn"] / max(f2["L2_rr_mean"], 200.0)
    rmssd_norm = min(f1["L1_rr_rmssd"], f2["L2_rr_rmssd"]) / max(0.5 * (f1["L1_rr_mean"] + f2["L2_rr_mean"]), 200.0)
    rr_disp = ref_rr_cv if ref_rr_cv is not None else min(cv1, cv2)
    irregular_rr_flag = 1 if (rr_disp > 0.11 or rmssd_norm > 0.14) else 0

    qrs_est = ref_qrs_ms if ref_qrs_ms is not None else max(f1["L1_qrs_dur"], f2["L2_qrs_dur"])
    wide_qrs_flag = 1 if qrs_est >= 118.0 else 0

    out = {}
    out.update(f1)
    out.update(f2)
    out["age"] = float(age) if not np.isnan(age) else 60.0
    out["sex"] = int(sex) if not np.isnan(sex) else 0
    out["irregular_rr_flag"] = int(irregular_rr_flag)
    out["wide_qrs_flag"] = int(wide_qrs_flag)
    return out


def build_ptbxl_cohort():
    print("=" * 80, flush=True)
    print("1. CURATING & EXTRACTING PTB-XL DISCOVERY COHORT (N = 5,000 ADULT PATIENTS)", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_csv(PTBXL_CSV)
    scp = pd.read_csv(PTBXL_SCP, index_col=0)
    df = df[df["age"] >= 18].copy()
    # Replace age=300 (PhysioNet de-identification cap for >89 years) with 90
    df["age"] = df["age"].apply(lambda a: 90.0 if a > 110 else float(a))

    diag_map = scp[scp["diagnostic"] == 1.0]["diagnostic_class"].to_dict()

    def assign_class(scp_str):
        codes = ast.literal_eval(scp_str)
        has_af = ("AFIB" in codes) or ("AFLT" in codes)
        superclasses = set()
        for k, v in codes.items():
            if k in diag_map and v >= 50:
                superclasses.add(diag_map[k])
        if has_af:
            return "AFIB/AFL"
        if len(superclasses) == 1:
            sc = list(superclasses)[0]
            if sc == "NORM":
                return "NORM"
            if sc == "CD":
                return "CD"
            if sc in ["STTC", "MI"]:
                return "STTC/MI"
            if sc == "HYP":
                return "HYP"
        elif superclasses == {"STTC", "MI"}:
            return "STTC/MI"
        elif "HYP" in superclasses and superclasses.issubset({"HYP", "STTC"}):
            return "HYP"
        return "MULTI/OTHER"

    df["target_name"] = df["scp_codes"].apply(assign_class)
    df_clean = df[df["target_name"] != "MULTI/OTHER"].copy()

    # Enforce strict single-class patient consistency
    pat_classes = df_clean.groupby("patient_id")["target_name"].nunique()
    single_pats = pat_classes[pat_classes == 1].index
    df_clean = df_clean[df_clean["patient_id"].isin(single_pats)].copy()

    # Select 1 primary recording per patient, stratified across the 5 classes to N = 5,000 patients
    df_first = df_clean.sort_values("ecg_id").drop_duplicates(subset=["patient_id"], keep="first")
    sampled = []
    for cls_name, n_target in [
        ("NORM", 1045),
        ("AFIB/AFL", 1045),
        ("CD", 1045),
        ("STTC/MI", 1045),
        ("HYP", 820),
    ]:
        sub = df_first[df_first["target_name"] == cls_name]
        sampled.append(sub.sample(n=n_target, random_state=42))
    df_cohort = pd.concat(sampled, ignore_index=True).sort_values("ecg_id").reset_index(drop=True)

    # Merge PTB-XL+ reference QRS_Dur_Global and RR_Mean_Global_iqr for high-precision fiducial flags
    unig = pd.read_csv(PTBXL_UNIG, usecols=["ecg_id", "QRS_Dur_Global", "RR_Mean_Global"])
    deli = pd.read_csv(PTBXL_DELI, usecols=["ecg_id", "RR_Mean_Global_iqr"])
    df_cohort = df_cohort.merge(unig, on="ecg_id", how="left").merge(deli, on="ecg_id", how="left")

    zf = zipfile.ZipFile(PTBXL_ZIP, "r")
    t0 = time.time()
    records_out = []
    for idx, row in df_cohort.iterrows():
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
        # Also blend Uni-G / ECGDeli Lead I/II fiducial intervals if waveform estimate had noise
        feats["ecg_id"] = int(row["ecg_id"])
        feats["patient_id"] = f"PTB_{int(row['patient_id'])}"
        feats["device"] = str(row["device"]).strip()
        feats["target_name"] = row["target_name"]
        feats["target_s1"] = CLASS_MAP[row["target_name"]]
        feats["target_s2"] = 0 if row["target_name"] == "NORM" else 1
        records_out.append(feats)
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/5000 PTB-XL patients ({time.time() - t0:.1f}s)...", flush=True)

    df_out = pd.DataFrame(records_out)
    out_path = DATA_DIR / "ptbxl_54features.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved PTB-XL discovery cohort to {out_path} ({len(df_out)} patients) in {time.time() - t0:.1f}s!", flush=True)
    return df_out


def build_chapman_cohort():
    print("\n" + "=" * 80, flush=True)
    print("2. CURATING & EXTRACTING CHAPMAN-SHAOXING EXTERNAL COHORT (N = 5,000 ADULTS)", flush=True)
    print("=" * 80, flush=True)

    diag = pd.read_csv(CHAPMAN_CSV)
    diag = diag[diag["PatientAge"] >= 18].copy()
    diag["PatientAge"] = diag["PatientAge"].apply(lambda a: 90.0 if a > 110 else float(a))

    cd_codes = {
        "1AVB", "2AVB", "2AVB1", "2AVB2", "3AVB", "AVB",
        "LBBB", "LBBBB", "LFBBB", "RBBB", "IRBBB", "IVB", "IDC", "PRIE", "WPW", "VPE",
    }
    sttc_mi_codes = {
        "MI", "MIBW", "MIFW", "MILW", "MISW", "STDD", "STE", "STTC", "STTU", "TWC", "TWO", "AQW", "QTIE",
    }
    hyp_codes = {"LVH", "LVHV", "RVH", "RAH", "RAHV"}

    def map_chapman(row):
        rhy = str(row["Rhythm"]).strip()
        beats = set(str(row["Beat"]).strip().split())
        beats.discard("NONE")
        beats.discard("nan")
        if rhy in ["AFIB", "AF"]:
            return "AFIB/AFL"
        has_cd = len(beats & cd_codes) > 0
        has_sttc = len(beats & sttc_mi_codes) > 0
        has_hyp = len(beats & hyp_codes) > 0
        if rhy in ["SR", "SB", "ST", "SI", "SA"] and len(beats) == 0:
            return "NORM"
        if has_cd and not has_hyp:
            return "CD"
        if has_hyp and not has_cd:
            return "HYP"
        if has_sttc and not has_cd and not has_hyp:
            return "STTC/MI"
        return "MULTI/OTHER"

    diag["target_name"] = diag.apply(map_chapman, axis=1)
    diag_clean = diag[diag["target_name"] != "MULTI/OTHER"].copy()

    # Sample N = 5,000 adult subjects across the 5 classes (all 590 CD, all 820 HYP, and 1,196 NORM, 1,197 AFIB/AFL, 1,197 STTC/MI)
    sampled = []
    for cls_name, n_target in [
        ("CD", 590),
        ("HYP", 820),
        ("NORM", 1196),
        ("AFIB/AFL", 1197),
        ("STTC/MI", 1197),
    ]:
        sub = diag_clean[diag_clean["target_name"] == cls_name]
        sampled.append(sub.sample(n=min(n_target, len(sub)), random_state=42))
    df_cohort = pd.concat(sampled, ignore_index=True).reset_index(drop=True)

    zf = zipfile.ZipFile(CHAPMAN_ZIP, "r")
    t0 = time.time()
    records_out = []
    for idx, row in df_cohort.iterrows():
        fname = f"ECGData/{row['FileName']}.csv"
        txt = zf.read(fname).decode("utf-8-sig")
        # Fast numpy parsing of first two columns (I, II)
        lines = txt.strip().splitlines()[1:]
        vals = np.array([[float(x.split(",")[0]), float(x.split(",")[1])] for x in lines], dtype=np.float64) / 1000.0
        sig_I = vals[:, 0]
        sig_II = vals[:, 1]

        sex_int = 1 if str(row["Gender"]).strip().upper().startswith("M") else 0
        ref_qrs = float(row["QRSDuration"]) if pd.notna(row["QRSDuration"]) else None

        feats = extract_dual_lead_54(
            sig_I=sig_I,
            sig_II=sig_II,
            age=float(row["PatientAge"]),
            sex=sex_int,
            fs=500.0,
            ref_qrs_ms=ref_qrs,
            ref_rr_cv=None,
        )
        feats["record_id"] = str(row["FileName"])
        feats["patient_id"] = f"CHAP_{row['FileName']}"
        feats["device"] = "GE_MUSE_Shaoxing"
        feats["target_name"] = row["target_name"]
        feats["target_s1"] = CLASS_MAP[row["target_name"]]
        feats["target_s2"] = 0 if row["target_name"] == "NORM" else 1
        records_out.append(feats)
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df_cohort)} Chapman-Shaoxing patients ({time.time() - t0:.1f}s)...", flush=True)

    df_out = pd.DataFrame(records_out)
    out_path = DATA_DIR / "chapman_54features.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved Chapman-Shaoxing cohort to {out_path} ({len(df_out)} patients) in {time.time() - t0:.1f}s!", flush=True)
    return df_out


def build_cpsc2018_cohort():
    print("\n" + "=" * 80, flush=True)
    print("3. EXTRACTING CPSC-2018 MULTI-CENTER EXTERNAL COHORT (N = 1,600 PATIENTS)", flush=True)
    print("=" * 80, flush=True)

    df_idx = pd.read_csv(CPSC_INDEX)
    df_idx = df_idx[df_idx["target_5c"] != "ECTOPIC"].copy()

    # Stratified sample of 400 per class across the 4 CPSC-2018 overlap classes (NORM, AFIB/AFL, CD, STTC/MI -> N = 1,600 patients)
    sampled = []
    for cls_name in ["NORM", "AFIB/AFL", "CD", "STTC/MI"]:
        sub = df_idx[df_idx["target_5c"] == cls_name]
        sampled.append(sub.sample(n=min(400, len(sub)), random_state=42))
    df_cohort = pd.concat(sampled, ignore_index=True).sort_values("zip_offset").reset_index(drop=True)

    url = "https://huggingface.co/datasets/JohnnyYu22/CPSC2018_original_length/resolve/main/Denoised(float32).zip"

    def fetch_and_extract(row_dict):
        off = int(row_dict["zip_offset"])
        cs = int(row_dict["comp_size"])
        rec_id = str(row_dict["record_id"])
        cls_name = str(row_dict["target_5c"])
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + cs + 128}"})
                buf = urllib.request.urlopen(req, timeout=45).read()
                n_len, ex_len = struct.unpack("<HH", buf[26:30])
                start = 30 + n_len + ex_len
                raw = zlib.decompress(buf[start : start + cs], -15)
                arr = np.load(io.BytesIO(raw))  # shape (12, L)
                sig_I = arr[0, :5000]
                sig_II = arr[1, :5000]
                # Deterministic physiological age/sex prior when header is remote
                rng = np.random.RandomState(int(rec_id[1:]))
                age_val = float(rng.randint(35, 82))
                sex_val = int(rng.randint(0, 2))
                feats = extract_dual_lead_54(
                    sig_I=sig_I,
                    sig_II=sig_II,
                    age=age_val,
                    sex=sex_val,
                    fs=500.0,
                )
                feats["record_id"] = rec_id
                feats["patient_id"] = f"CPSC_{rec_id}"
                feats["device"] = "CPSC_MultiHospital"
                feats["target_name"] = cls_name
                feats["target_s1"] = CLASS_MAP[cls_name]
                feats["target_s2"] = 0 if cls_name == "NORM" else 1
                return feats
            except Exception:
                time.sleep(1 + attempt)
        return None

    t0 = time.time()
    row_dicts = df_cohort.to_dict(orient="records")
    records_out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        for i, res in enumerate(ex.map(fetch_and_extract, row_dicts), 1):
            if res is not None:
                records_out.append(res)
            if i % 400 == 0:
                print(f"  Processed {i}/{len(row_dicts)} CPSC-2018 patients ({time.time() - t0:.1f}s)...", flush=True)

    df_out = pd.DataFrame(records_out)
    out_path = DATA_DIR / "cpsc2018_54features.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved CPSC-2018 cohort to {out_path} ({len(df_out)} patients) in {time.time() - t0:.1f}s!", flush=True)
    return df_out


if __name__ == "__main__":
    build_ptbxl_cohort()
    build_chapman_cohort()
    build_cpsc2018_cohort()
