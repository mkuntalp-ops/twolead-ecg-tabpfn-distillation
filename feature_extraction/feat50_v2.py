"""FEAT_50 v2: corrected fiducial measurement for the two-lead (I, II) feature set.

Differences from v1 (`01_extract_features_all_cohorts.py`, kept unchanged as the archived version):
  * median beat spans R - 350 ms to R + 450 ms (v1: R - 200 / + 400 ms), so long PR and QT intervals fit;
  * QRS onset/offset = edges of the above-threshold derivative-envelope region connected to the R peak, searched
    outwards from R within R - 90 / + 120 ms (v1: first/last above-threshold sample inside a fixed window, which
    capped QRS duration at 146 ms);
  * P onset is searched backwards from the P peak on a 15 Hz low-passed beat (slope threshold), the P peak being
    located in the 300 ms before QRS onset (v1 searched forwards from the start of a truncated window and mostly
    returned that window start, giving an almost constant PR interval);
  * T end is searched up to R + 440 ms (v1: R + 390 ms, which capped QT at roughly 430 ms).
All other feature definitions (baseline, amplitudes, RR, spectral and statistical descriptors) are those of v1,
expressed in milliseconds relative to the R peak.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis

_V1 = Path(__file__).resolve().parents[2] / "01_extract_features_all_cohorts.py"
_spec = importlib.util.spec_from_file_location("feat50_v1", _V1)
_v1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_v1)
detect_r_peaks_pan_tompkins = _v1.detect_r_peaks_pan_tompkins

# Chosen on the 1,000-record development subset only (dev_tune_feat50_v2.py and the documented P/QRS exploration).
DEFAULT_PARAMS = {"qrs_thr_frac": 0.085, "qrs_thr_min": 0.0025, "qrs_gap_ms": 20.0,
                  "p_search_ms": 300.0, "p_min_gap_ms": 50.0, "p_thr_frac": 0.15}

SUFFIXES = ["pr_int", "qrs_dur", "qt_int", "qtc_frid", "tpte_int", "p_amp", "q_amp", "r_amp", "s_amp", "t_amp",
            "j_amp", "st60_amp", "st_slope", "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_ent",
            "pow_05_5hz", "pow_5_15hz", "pow_15_40hz", "spec_ent", "zcr", "rms", "kurtosis"]


def _ms(ms: float, fs: float) -> int:
    return int(round(ms * fs / 1000.0))


def _region_around(mask: np.ndarray, centre: int, gap: int) -> tuple[int, int] | None:
    """Edges of the above-threshold region containing (or nearest to) `centre`, merging gaps of <= `gap` samples."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    breaks = np.where(np.diff(idx) > gap + 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    inside = np.where((starts <= centre) & (ends >= centre))[0]
    k = inside[0] if len(inside) else int(np.argmin(np.minimum(np.abs(starts - centre), np.abs(ends - centre))))
    return int(starts[k]), int(ends[k])


def extract_single_lead_v2(sig_1d: np.ndarray, fs: float = 500.0, lead_prefix: str = "L1",
                           shared_r_peaks: np.ndarray | None = None, params: dict | None = None,
                           return_debug: bool = False):
    prm = {**DEFAULT_PARAMS, **(params or {})}
    sig = np.nan_to_num(sig_1d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n10 = int(10.0 * fs)
    sig = sig[:n10] if len(sig) > n10 else (np.pad(sig, (0, n10 - len(sig))) if len(sig) < int(2.0 * fs) else sig)
    b, a = butter(3, [0.5 / (fs / 2.0), 40.0 / (fs / 2.0)], btype="band")
    x = filtfilt(b, a, sig)

    r_peaks = shared_r_peaks if shared_r_peaks is not None else detect_r_peaks_pan_tompkins(x, fs)
    half = _ms(30, fs)
    local_r = [s + int(np.argmax(x[s:e])) for p in r_peaks for s, e in [(max(0, p - half), min(len(x), p + half))] if e > s]
    local_r = np.unique(local_r) if len(local_r) >= 2 else np.asarray(r_peaks)

    # ---- RR features (identical to v1)
    rr = np.diff(local_r) * (1000.0 / fs) if len(local_r) >= 2 else np.array([800.0])
    rr = rr[(rr >= 240.0) & (rr <= 2400.0)]
    if len(rr) == 0: rr = np.array([800.0])
    rr_mean = float(np.mean(rr)); rr_sdnn = float(np.std(rr)) if len(rr) > 1 else 0.0
    d_rr = np.diff(rr)
    rr_rmssd = float(np.sqrt(np.mean(d_rr ** 2))) if len(d_rr) else 0.0
    rr_pnn50 = float(np.mean(np.abs(d_rr) > 50.0) * 100.0) if len(d_rr) else 0.0
    h, _ = np.histogram(rr, bins=8, range=(300.0, 1900.0), density=True); h = h[h > 0]
    rr_ent = float(-np.sum(h * np.log2(h + 1e-12)))

    # ---- median beat, R at index `pre`
    pre, post = _ms(350, fs), _ms(450, fs)
    beats = [x[r - pre:r + post] for r in local_r if r - pre >= 0 and r + post <= len(x)]
    med = np.median(np.vstack(beats), axis=0) if beats else np.zeros(pre + post)
    iso = float(np.mean(med[pre - _ms(100, fs):pre - _ms(60, fs)]))          # v1 baseline: R-100..R-60 ms
    mb = med - iso
    env = np.convolve(np.abs(np.diff(mb, prepend=mb[0])), np.ones(7) / 7.0, mode="same")

    # ---- QRS: region connected to R inside R-90..R+120 ms
    lo, hi = pre - _ms(90, fs), pre + _ms(120, fs)
    seg = env[lo:hi]
    thr = max(prm["qrs_thr_frac"] * float(seg.max()), prm["qrs_thr_min"])
    reg = _region_around(seg >= thr, pre - lo, _ms(prm["qrs_gap_ms"], fs))
    qrs_fallback = reg is None
    qrs_on, qrs_off = (lo + reg[0], lo + reg[1]) if reg else (pre - _ms(40, fs), pre + _ms(40, fs))
    qrs_dur = float(np.clip((qrs_off - qrs_on) * 1000.0 / fs, 50.0, 240.0))

    # ---- P wave on a 15 Hz low-passed beat: peak = largest deviation from the search-window start within the
    #      300 ms before QRS onset (excluding the last 50 ms); onset = first sample, searching backwards from the
    #      peak, where the absolute slope falls below p_thr_frac x the window's maximum slope
    bl, al = butter(2, 15.0 / (fs / 2.0), btype="low")
    lp = filtfilt(bl, al, mb)
    p_lo = max(1, qrs_on - _ms(prm["p_search_ms"], fs)); p_hi = max(p_lo + 2, qrs_on - _ms(prm["p_min_gap_ms"], fs))
    p_peak = int(np.argmax(np.abs(lp[p_lo:p_hi] - lp[p_lo])))
    slope = np.abs(np.diff(lp, prepend=lp[0]))[p_lo:p_hi]
    p_thr = prm["p_thr_frac"] * float(slope.max())
    k = p_peak
    while k > 0 and slope[k] > p_thr: k -= 1
    p_fallback = (k == 0)
    pr_int = float(np.clip((qrs_on - (p_lo + k)) * 1000.0 / fs, 80.0, 360.0))

    # ---- amplitudes (v1 definitions, in ms relative to R)
    p_win = mb[pre - _ms(180, fs):pre - _ms(60, fs)]
    p_amp = float(p_win[int(np.argmax(np.abs(p_win)))])
    q_win = mb[max(pre - _ms(64, fs), qrs_on - 5):pre]
    q_amp = float(q_win.min()) if len(q_win) and q_win.min() < 0 else 0.0
    r_amp = float(np.max(mb[pre - 8:pre + 9]))
    s_win = mb[pre:min(pre + _ms(90, fs), qrs_off + 8)]
    s_amp = float(s_win.min()) if len(s_win) and s_win.min() < 0 else 0.0
    j_idx = min(qrs_off, len(mb) - 1); j_amp = float(mb[j_idx])
    st60_amp = float(mb[min(qrs_off + _ms(60, fs), len(mb) - 1)]); st_slope = float((st60_amp - j_amp) / 0.06)

    # ---- T wave (peak window as v1: R+100..R+360 ms); T end searched up to R+440 ms
    t_lo, t_hi = pre + _ms(100, fs), pre + _ms(360, fs)
    t_idx = t_lo + int(np.argmax(np.abs(mb[t_lo:t_hi]))); t_amp = float(mb[t_idx])
    t_end_lim = pre + _ms(440, fs)
    below = np.where(np.abs(mb[t_idx:t_end_lim]) <= max(0.15 * abs(t_amp), 0.012))[0]
    t_end = t_idx + int(below[0]) if len(below) else min(t_idx + 35, t_end_lim - 1)
    qt_int = float(np.clip((t_end - qrs_on) * 1000.0 / fs, 220.0, 620.0))
    qtc_frid = float(qt_int / (max(rr_mean / 1000.0, 0.30) ** (1.0 / 3.0)))
    tpte_int = float(np.clip((t_end - t_idx) * 1000.0 / fs, 20.0, 220.0))

    # ---- spectral & statistical (identical to v1)
    fr, psd = welch(x, fs=fs, nperseg=min(1024, len(x))); tot = float(psd.sum() + 1e-12)
    band = lambda a_, b_, inc=False: float(psd[(fr >= a_) & ((fr <= b_) if inc else (fr < b_))].sum() / tot)
    pn = psd / tot
    out = dict(zip([f"{lead_prefix}_{s}" for s in SUFFIXES], [
        pr_int, qrs_dur, qt_int, qtc_frid, tpte_int, p_amp, q_amp, r_amp, s_amp, t_amp, j_amp, st60_amp, st_slope,
        rr_mean, rr_sdnn, rr_rmssd, rr_pnn50, rr_ent, band(0.5, 5.0), band(5.0, 15.0), band(15.0, 40.0, True),
        float(-np.sum(pn * np.log2(pn + 1e-12))), float(np.mean(np.diff(np.sign(x)) != 0)), float(np.sqrt(np.mean(x ** 2))),
        float(np.clip(kurtosis(x), -5.0, 50.0))]))
    if return_debug:
        return out, {"n_beats": len(beats), "qrs_fallback": qrs_fallback, "p_fallback": p_fallback}
    return out


def extract_feat50_v2(sig_I: np.ndarray, sig_II: np.ndarray, fs: float = 500.0, params: dict | None = None,
                      return_debug: bool = False):
    """FEAT_50 v2 for leads I and II with R peaks detected on the combined filtered signal (as in v1)."""
    b, a = butter(3, [0.5 / (fs / 2.0), 40.0 / (fs / 2.0)], btype="band")
    x1 = filtfilt(b, a, np.nan_to_num(sig_I[:int(10 * fs)].astype(np.float64)))
    x2 = filtfilt(b, a, np.nan_to_num(sig_II[:int(10 * fs)].astype(np.float64)))
    peaks = detect_r_peaks_pan_tompkins(x1 + x2, fs=fs)
    r1 = extract_single_lead_v2(sig_I, fs, "L1", peaks, params, return_debug)
    r2 = extract_single_lead_v2(sig_II, fs, "L2", peaks, params, return_debug)
    if return_debug:
        return {**r1[0], **r2[0]}, {"L1": r1[1], "L2": r2[1]}
    return {**r1, **r2}
