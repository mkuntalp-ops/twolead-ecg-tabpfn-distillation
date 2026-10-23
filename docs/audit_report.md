# Comprehensive Audit Report: Two-Lead ECG Foundation Model Distillation Project
**Date:** 2026-10-20  
**Project:** TabPFN Distillation for Two-Lead (Lead I + II) Ambulatory ECG Classification  
**Workspace:** `/Volumes/KIOXIA/Projects/gemma3-lora-other`  
**Auditor:** Research Software Engineer (Biomedical Signal Processing, ML Reproducibility & Statistical Audit)

---

## Executive Summary & Foundational Document Check

### 1. Document Inventory & Reviewer Document Audit
| Document Identifier | Location / Path | Status | Verification Detail & Source Provenance |
| :--- | :--- | :---: | :--- |
| **1. Manuscript v6** | `/Volumes/KIOXIA/Projects/gemma3-lora-other/Manuscript_TwoLead_ECG_Revised_v6.docx` | **FOUND & VERIFIED** | Size: 1,695,060 bytes, SHA-256: `33e0d957788f92b327298712eff8e327c81f866a4cc0890ddf8b35bdcc6f376c`. 11 native tables, 5 embedded figures. |
| **2. Hakem_Degerlendirmesi.pdf** | Expected root / Downloads | **NOT FOUND / UNVERIFIED** | A file named exactly `Hakem_Degerlendirmesi.pdf` does **not** exist in the workspace or system Downloads. (The file `/Users/mehmetkuntalp/Downloads/Hakem_Degerlendirmesi_Temiz.pdf` belongs to a separate ICBHI respiratory acoustic project). |
| **3. Reviewer Commentary Source (.docx)** | `.user_uploaded/media_*.docx` | **VERIFIED AS INTERNAL REVIEW** | The review dialogue is preserved in 6 sequential `.docx` files (`media_1792404228854_36680db7.docx` [Round 1 major critique, 192 paras, addressed to "Mehmet Hocam"], `media_1792407272595_c238e4ea.docx` [v2], `media_1792413300455_42d768eb.docx` [v3], `media_1792423986645_7a34339a.docx` [v4], `media_1792428122485_aba9ecfd.docx` [v5], `media_1792431881996_a58da008.docx` [v6]). **Action:** If an external journal editorial PDF exists, it must be provided by the user. |
| **4. Real Data, Code, Checkpoints, OOF Predictions** | Multi-directory workspace | **FOUND & VERIFIED** | 11 pipeline scripts (`01_...py` to `11_...py`), `build_docx.py`, 3 cohort CSVs, 4 OOF prediction files, 1 NPZ archive, 2 TabPFN checkpoints, Gemma 3 LoRA adapter directory. |

---

## 1. Sampling Rate, Time Base, and Waveform Feature Extraction Audit

### 1.1. Representative Record Audit Table (3 Records per Cohort)
To verify sampling rates and eliminate any ambiguity regarding 100 Hz vs. 500 Hz processing, 3 representative records from each cohort were audited directly from source files:

| Cohort | Record ID | Source File Path | Header / Catalog $f_s$ | Binary / Array Shape | Resampling Applied? | $f_s$ Passed to Extractor | Effective Duration |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PTB-XL** | `ecg_id=1` | `/Volumes/KIOXIA/Datasets/ptb-xl/records500/00000/00001_hr.dat` (via `ptb-xl-data.zip`) | $500\text{ Hz}$ (`00001_hr 12 500 5000`) | `(5000, 12)` ($120,000\text{ bytes}$) | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |
| **PTB-XL** | `ecg_id=500` | `/Volumes/KIOXIA/Datasets/ptb-xl/records500/00000/00500_hr.dat` (via `ptb-xl-data.zip`) | $500\text{ Hz}$ (`00500_hr 12 500 5000`) | `(5000, 12)` ($120,000\text{ bytes}$) | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |
| **PTB-XL** | `ecg_id=1000` | `/Volumes/KIOXIA/Datasets/ptb-xl/records500/01000/01000_hr.dat` (via `ptb-xl-data.zip`) | $500\text{ Hz}$ (`01000_hr 12 500 5000`) | `(5000, 12)` ($120,000\text{ bytes}$) | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |
| **Chapman** | `MUSE_...27000` | `data/chapman/ECGData.zip` (`ECGData/MUSE_20180113_171327_27000.csv`) | $500\text{ Hz}$ (Zheng et al., 2020) | `(5000, 2)` (5,000 lines) | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |
| **Chapman** | `MUSE_...29000` | `data/chapman/ECGData.zip` (`ECGData/MUSE_20180112_073319_29000.csv`) | $500\text{ Hz}$ (Zheng et al., 2020) | `(5000, 2)` (5,000 lines) | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |
| **Chapman** | `MUSE_...97000` | `data/chapman/ECGData.zip` (`ECGData/MUSE_20180111_165520_97000.csv`) | $500\text{ Hz}$ (Zheng et al., 2020) | `(5000, 2)` (5,000 lines) | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |
| **CPSC-2018** | `A0001` | `JohnnyYu22/CPSC2018_original_length/Denoised(float32).zip` (`A0001.npy`) | $500\text{ Hz}$ (Liu et al., 2018) | `(12, 7500)` $\rightarrow$ sliced `:5000` | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |
| **CPSC-2018** | `A0003` | `JohnnyYu22/CPSC2018_original_length/Denoised(float32).zip` (`A0003.npy`) | $500\text{ Hz}$ (Liu et al., 2018) | `(12, 5000)` $\rightarrow$ sliced `:5000` | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |
| **CPSC-2018** | `A0004` | `JohnnyYu22/CPSC2018_original_length/Denoised(float32).zip` (`A0004.npy`) | $500\text{ Hz}$ (Liu et al., 2018) | `(12, 5974)` $\rightarrow$ sliced `:5000` | **None** | $500.0\text{ Hz}$ | $10.0\text{ s}$ |

### 1.2. Retraction & Clarification of the "100 Hz Resampling" Statement
- **Erroneous Statement in Previous Session:** "PTB-XL: records100 kullanıldı ve 500'den 100'e indirildi."
- **Audit Fact:** This statement was an **erroneous hallucination by the AI assistant in the previous turn** and is hereby **explicitly retracted**.
- **Code Reality (`01_extract_features_all_cohorts.py`, line 340):**
  ```python
  rel_hr = row["filename_hr"] + ".dat"
  raw = zf.read(rel_hr)
  sig = np.frombuffer(raw, dtype="<i2").reshape(-1, 12) / 1000.0
  ```
  The code reads `filename_hr`, which points strictly to `records500/` (500 Hz, 5,000 samples). `filename_lr` (`records100/`) was **never accessed** in the extraction pipeline.
- **Resampling Check:** No decimation, downsampling, or resampling function (`scipy.signal.resample` or `decimate`) exists anywhere in `01_extract_features_all_cohorts.py`. Signals entered feature extraction at their native $500\text{ Hz}$.

### 1.3. Window Indexing & Time Base Mathematical Audit
In `extract_single_lead_25` (`01_extract_features_all_cohorts.py`, lines 83–230), the time base is fixed to $f_s = 500.0\text{ Hz}$ ($1\text{ sample} = 2.0\text{ ms}$):

| Parameter / Segment | Code Implementation (`01_...py`) | Integer Sample Range | Millisecond Range Relative to R-Peak ($t=0\text{ ms}$) | Duration / Window Width | Conversion Formula |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Median Beat Pre-R** | `pre = int(0.20 * fs)` | `0 .. 100` | $-200\text{ ms} \dots 0\text{ ms}$ | $200\text{ ms}$ | $100 \times (1000/500) = 200\text{ ms}$ |
| **Median Beat Post-R**| `post = int(0.40 * fs)` | `100 .. 300` | $0\text{ ms} \dots +400\text{ ms}$ | $400\text{ ms}$ | $200 \times (1000/500) = 400\text{ ms}$ |
| **Total Beat Buffer** | `len(med_beat) = pre + post` | `0 .. 300` | $-200\text{ ms} \dots +400\text{ ms}$ | $600\text{ ms}$ | $300\text{ samples}$ |
| **Isoelectric Baseline** | `iso = mean(med_beat[50:70])` | `50 .. 70` | $-100\text{ ms} \dots -60\text{ ms}$ (PR segment) | $40\text{ ms}$ | Sliced before P-wave / Q-onset |
| **P-Wave Window** | `p_win = mb[10:70]` | `10 .. 70` | $-180\text{ ms} \dots -60\text{ ms}$ | $120\text{ ms}$ | Peak is P-amplitude |
| **QRS Envelope Window** | `qrs_env = env[68:142]` | `68 .. 142` | $-64\text{ ms} \dots +84\text{ ms}$ | $148\text{ ms}$ | Derivative envelope threshold |
| **QRS Duration** | `(qrs_off - qrs_on) * (1000/fs)` | Dynamic $\subset [68, 142]$ | Within $[-64\text{ ms}, +84\text{ ms}]$ | Clipped $[50, 240]\text{ ms}$ | $\Delta\text{samples} \times 2.0\text{ ms}$ |
| **PR Interval** | `(qrs_on - p_on) * (1000/fs)` | Dynamic $\subset [10, 142]$ | Within $[-180\text{ ms}, +84\text{ ms}]$ | Clipped $[80, 360]\text{ ms}$ | $\Delta\text{samples} \times 2.0\text{ ms}$ |
| **J-Point Amplitude** | `j_amp = mb[qrs_off]` | Index `qrs_off` | QRS termination | Point value | Evaluated at QRS offset |
| **ST at J+60ms** | `st60 = mb[qrs_off + int(0.06*fs)]` | `qrs_off + 30` | J-point $+ 60\text{ ms}$ | Point value | $+30\text{ samples}$ at $500\text{ Hz}$ |
| **ST Slope** | `(st60_amp - j_amp) / 0.06` | Amplitude difference | Over $60\text{ ms}$ window | $\text{mV/s}$ | $\Delta V / 0.06\text{ s}$ |
| **T-Wave Window** | `t_win = mb[150:280]` | `150 .. 280` | $+100\text{ ms} \dots +360\text{ ms}$ | $260\text{ ms}$ | Peak is T-amplitude |
| **QT Interval** | `(t_end - qrs_on) * (1000/fs)` | Dynamic | QRS onset to T offset | Clipped $[220, 620]\text{ ms}$ | $\Delta\text{samples} \times 2.0\text{ ms}$ |
| **Fridericia QTc** | `qt_int / (rr_sec ** (1/3))` | Derived | Normalized by RR | Scaled ms | Cube-root Fridericia index |
| **Welch Spectral PSD** | `welch(x, fs=fs, nperseg=1024)` | $N_{\text{seg}} = 1024$ | Frequency resolution $\Delta f = 0.488\text{ Hz}$ | $0.5\text{--}40\text{ Hz}$ | Powers integrated over bands |

- **Experimental Proof on 100 Hz Incompatibility:**  
  When a $100\text{ Hz}$ array is passed to `extract_single_lead_25(sig, fs=100.0)`:
  - `pre = 20`, `post = 40`, `len(med_beat) = 60`.
  - Slicing `qrs_env = env[68:142]` on length-60 yields an empty array `len == 0`.
  - Python immediately crashes with: `ValueError: zero-size array to reduction operation maximum which has no identity`.
  - **Conclusion:** The feature extractor was physically incapable of running on $100\text{ Hz}$ data. All $11,600$ records across all three cohorts were processed at native $500\text{ Hz}$.

---

## 2. CPSC-2018 Demographics, Source Mapping & Patient Linkage

### 2.1. Retraction of Overgeneralization
- **Erroneous Statement:** "CPSC kaynağında yaş/cinsiyet yok."
- **Correction:** In the official CPSC-2018 / PhysioNet Challenge 2020 distribution, WFDB header files (`.hea`) **do contain** demographic fields:
  ```text
  #Age: 65
  #Sex: Male
  #Dx: 164889003
  ```
  The claim that the CPSC source lacks age/sex is **retracted**.

### 2.2. Three Distinct Issues Separated
1. **Demographic Matching Feasibility:**  
   The record IDs in `JohnnyYu22/CPSC2018_original_length` (`A0001`, `A0003`, etc.) match official CPSC-2018 / PhysioNet 2020 record names. In principle, downloading official `.hea` files would allow demographic joins. However, the downloaded archive `Denoised(float32).zip` contained only `.npy` array files, and local access to the official server was restricted (`data/cpsc2018_records_source.txt`).
2. **Synthetic Demographics in Stored File:**  
   In `01_extract_features_all_cohorts.py` (lines 508–510), pseudo-random age (`randint(35, 82)`) and sex were generated. Therefore, `data/cpsc2018_54features.csv` contains synthetic demographic columns.
3. **Manuscript Isolation & Record-Level Independence:**  
   The manuscript correctly neutralizes this: all CPSC-2018 evaluations in the paper are strictly conducted on **`FEAT_50` (50 waveform features only)**, where `age` and `sex` are excluded. Furthermore, unlike PTB-XL and Chapman, CPSC-2018 does **not provide longitudinal patient identifiers**. Therefore, CPSC-2018 is audited and reported strictly as **record-level independent**, not patient-level independent.

---

## 3. TabPFN Configuration & Architecture Limits

### 3.1. Exact Configuration Table
| Parameter | Audited Value | Code Reference | Notes |
| :--- | :--- | :--- | :--- |
| **Package Version** | `tabpfn == 9.0.0` | `tabpfn.__version__` | Installed in `/Volumes/KIOXIA/AI/tabpfn_env` |
| **Model Architecture** | `tabpfn_v3` | Checkpoint `architecture_name` | Embed dim: 128, layers: 24, heads: 8 |
| **Checkpoint Path** | `.../tabpfn-v3-classifier-v3_20260417_multiclass.ckpt` | `02_benchmark_...py:34` | Size: 212.8 MB, SHA-256: `517c40ca21e0e1b1...` |
| **Outer Cross-Validation Folds** | **5 folds** (`n_splits=5`) | `02_benchmark_...py:123` | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` |
| **Outer Model Ensembling** | `n_estimators = 8` | `02_benchmark_...py:149` | 8 context permutations averaged |
| **Inner Soft-Target Folds** | **3 folds** (`n_splits=3`) | `02_benchmark_...py:140` | `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)` |
| **Inner Model Ensembling** | `n_estimators = 4` | `02_benchmark_...py:143` | 4 context permutations averaged |
| **Device** | `"mps"` (Apple Silicon GPU) | `02_benchmark_...py:143, 149` | Accelerates forward attention passes |

### 3.2. Correction of the "8-Fold OOF" Description
- The description in some early notes as "8-fold OOF" was an erroneous conflation of `n_estimators=8` with `n_splits`.
- The actual protocol is **5-fold outer cross-validation** with an **8-estimator ensemble** at test time, and **3-fold inner cross-validation** with a **4-estimator ensemble** for soft-target generation.

### 3.3. Documenting the "10,000 Sample Limit" Claim
- In TabPFN v2 (Hollmann et al., Nature 2024), the context window was strictly capped at 10,000 samples.
- In the installed `tabpfn 9.0.0` (TabPFN-v3), checkpoint inspection shows:
  ```python
  ckpt['inference_config']['MAX_NUMBER_OF_SAMPLES'] = 1000000  # 1,000,000
  ```
- **CPU vs. MPS Behavior:**
  - On CPU, `_validate_num_samples_for_cpu` in `tabpfn/validation.py` raises `RuntimeError` if $N > 5000$ without setting `ignore_pretraining_limits=True`.
  - On MPS (`device="mps"`), testing with $N_{\text{train}} = 12,000$ and $N_{\text{test}} = 100$ executes natively without warnings or errors in **4.59 seconds**.
  - **Commitment:** No package/checkpoint update and no chunked ensemble will be deployed without explicit user instruction.

---

## 4. Re-framing Causal Language Regarding MLP Confounding

### 4.1. The Confounding Asymmetry Finding
- **`Cap-Matched MLP` (`MLPClassifier`):** Trained on hard class indices ($0 \dots 4$) with Cross-Entropy loss; early stopping monitored by validation classification accuracy.
- **`Distilled MLP` (`MLPRegressor`):** Trained on 5D log-probabilities ($\log(\text{clip}(Y_{\text{distill}}, 10^{-4}, 1.0))$) with Mean Squared Error (MSE); early stopping monitored by validation $R^2$.
- **`Label-Smooth MLP` (`MLPRegressor`):** Trained on 5D log-probabilities ($\log(\text{clip}(Y_{\text{smooth}}, 10^{-4}, 1.0))$) with MSE; early stopping monitored by validation $R^2$.

### 4.2. Causal Language Correction
- It is **scientifically unverified** to assert that distillation gains are "definitely" or "purely" due to the loss function change prior to matched experiments.
- The correct scientific statement is:
  > The comparison between `Cap-Matched MLP` and `Distilled MLP` is **confounded** by simultaneous shifts in loss formulation (Cross-Entropy vs. Log-space MSE) and convergence criteria (accuracy vs. $R^2$). Therefore, the observed $+1.45\text{ pp}$ improvement on PTB-XL FEAT_50 cannot be causally attributed to knowledge distillation alone.
- Furthermore:
  > The non-significant difference between `Distilled MLP` and `Label-Smooth MLP` ($+0.24\text{ pp}$, $p = 0.5415$) **does not prove equivalence** (absence of evidence is not evidence of absence). Only a strictly procedure-matched ablation holding architecture, optimizer, loss (e.g. Soft-Target Cross-Entropy), and early-stopping constant can isolate the specific marginal value of teacher soft labels.

---

## 5. Direct Reproduction of Table 5 & Calibration Audit

### 5.1. Direct NPZ Array Re-computation Table ($B = 10,000$ Bootstrap Runs)
Metrics were computed directly from `predictions/table5_all30_oof_predictions.npz` using vectorized nonparametric bootstrap with seed 42 (zero JSON reading):

| Test Description | $N$ | Distilled F1% | Control F1% | Difference (pp) | 95% Bootstrap CI | Opposing Draws ($n_{\text{opp}}$) | Two-Sided $p$-value | Table 5 Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PTB-XL FEAT50 Task1: Dist-MLP vs Base-MLP** | 5,000 | 73.23 | 69.37 | +3.86 | [2.89, 4.85] | 0 | 0.0002 | **PASS** |
| **PTB-XL FEAT50 Task1: Dist-MLP vs Cap-MLP** | 5,000 | 73.23 | 71.78 | +1.45 | [0.64, 2.28] | 4 | 0.0010 | **PASS** |
| **PTB-XL FEAT50 Task1: Dist-MLP vs Smooth-MLP** | 5,000 | 73.23 | 72.99 | +0.24 | [-0.52, 1.00] | 2,707 | 0.5415 | **PASS** |
| **PTB-XL FEAT50 Task1: Dist-XGB vs Base-XGB** | 5,000 | 73.28 | 72.00 | +1.28 | [0.60, 1.96] | 4 | 0.0010 | **PASS** |
| **PTB-XL FEAT50 Task1: Dist-XGB vs Cap-XGB** | 5,000 | 73.28 | 73.43 | -0.16 | [-0.83, 0.53] | 6,711 | 0.6579 | **PASS** |
| **PTB-XL FEAT50 Task1: Dist-XGB vs Smooth-XGB** | 5,000 | 73.28 | 73.15 | +0.13 | [-0.51, 0.76] | 3,511 | 0.7023 | **PASS** |
| **Chapman FEAT50 Task1: Dist-MLP vs Base-MLP** | 5,000 | 65.68 | 62.78 | +2.90 | [1.74, 4.10] | 0 | 0.0002 | **PASS** |
| **Chapman FEAT50 Task1: Dist-XGB vs Base-XGB** | 5,000 | 66.19 | 64.39 | +1.81 | [0.98, 2.62] | 0 | 0.0002 | **PASS** |
| **CPSC FEAT50 Task1: Dist-MLP vs Base-MLP** | 1,600 | 79.52 | 74.10 | +5.42 | [3.64, 7.21] | 0 | 0.0002 | **PASS** |
| **CPSC FEAT50 Task1: Dist-XGB vs Base-XGB** | 1,600 | 78.59 | 78.17 | +0.41 | [-0.78, 1.59] | 2,492 | 0.4986 | **PASS** |
| **PTB-XL FEAT50 Task2: Dist-MLP vs Base-MLP** | 5,000 | 80.28 | 81.31 | -1.03 | [-1.96, -0.08] | 9,839 | 0.0324 | **PASS** |
| **PTB-XL FEAT50 Task2: Dist-XGB vs Base-XGB** | 5,000 | 81.48 | 80.62 | +0.86 | [0.01, 1.70] | 231 | 0.0464 | **PASS** |

### 5.2. Row-Normalization vs. Probability Calibration Audit
Row-sum normalization and empirical probability calibration are mathematically distinct:

| Model | Row Sum Range | All Row Sums == 1.0? | Multi-Class Brier Score | Expected Calibration Error (ECE, 10 bins) | Calibration Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Teacher TabPFN** | [1.000000, 1.000000] | **True** | **0.3397** | **1.85%** | Well-calibrated Bayesian prior |
| **Distilled MLP** | [1.000000, 1.000000] | **True** | 0.3723 | 5.40% | Moderate confidence overestimation |
| **Baseline MLP** | [1.000000, 1.000000] | **True** | 0.4137 | 1.93% | Good calibration, lower accuracy |
| **Distilled XGBoost** | [1.000000, 1.000000] | **True** | 0.3651 | 3.24% | Well-calibrated soft probabilities |
| **Baseline XGBoost** | [1.000000, 1.000000] | **True** | 0.3831 | 6.70% | Higher calibration error |

---

## 6. Cohort Counting, Exclusion Hierarchy, and Discrepancy Trace

### 6.1. PTB-XL Cohort Flow Trace (From Raw Database to Discovery Sample)
```
Raw PTB-XL Database (v1.0.3)
└── 21,799 ECG recordings from 18,869 unique patients
    │
    ├── [Filter 1: Age ≥ 18 years]
    │   └── Excluded: 133 pediatric recordings (117 patients)
    │   └── Retained: 21,666 adult recordings (18,752 unique patients)
    │
    ├── [Filter 2: Multi-Label Conflict Resolution via assign_class()]
    │   ├── Rule A: Rhythm AFIB/AFLT presence -> AFIB/AFL
    │   ├── Rule B: Single diagnostic superclass (likelihood ≥ 50%) -> NORM, CD, STTC/MI, HYP
    │   ├── Rule C: STTC + MI consolidation -> STTC/MI
    │   ├── Rule D: HYP + STTC consolidation -> HYP
    │   └── Excluded as MULTI/OTHER: 4,120 records (3,161 patients)
    │   └── Retained: 17,546 clean adult records (15,591 unique patients)
    │
    ├── [Filter 3: Longitudinal Single-Class Patient Consistency]
    │   └── Patients whose records changed category across longitudinal visits excluded
    │   └── Excluded: 1,069 records (431 inconsistent patients)
    │   └── Retained: 16,477 records from 15,160 consistent adult patients
    │
    ├── [Filter 4: Deduplication to 1 Record per Patient]
    │   └── First ecg_id retained; subsequent longitudinal repeats dropped
    │   └── Eligible Pool: EXACTLY 15,160 unique adult patients
    │       ├── NORM: 8,052 (53.1%)
    │       ├── STTC/MI: 3,486 (23.0%)
    │       ├── CD: 1,647 (10.9%)
    │       ├── AFIB/AFL: 1,155 (7.6%)
    │       └── HYP: 820 (5.4%) [Total = 15,160]
    │
    └── [Filter 5: Stratified Subsampling to Balanced Cohort]
        └── Final Discovery Sample: N = 5,000 unique patients
            ├── NORM: 1,045
            ├── AFIB/AFL: 1,045
            ├── CD: 1,045
            ├── STTC/MI: 1,045
            └── HYP: 820 (100% of all eligible HYP patients)
```

### 6.2. Origin of the Discrepancy (15,160 vs. 14,211 vs. 14,928)
1. **True Ground-Truth Count:** Under the exact code in `01_extract_features_all_cohorts.py`, the eligible pool contains **15,160 unique adult patients**. (Notice that `HYP` has exactly 820 patients, explaining why line 325 takes `n_target = 820`).
2. **Origin of `14,211` (in Manuscript v6 Table 0):**  
   In earlier project revisions, an alternative filter was tested where `AFIB/AFL` required explicit diagnostic likelihood $\ge 50\%$ from `diag_map` (rather than any string presence in `scp_codes`) and secondary STTC strain was not consolidated into HYP. Under that stricter filter, the eligible adult count was **14,220–14,229**. The number `14,211` in Table 0 was carried over from that earlier draft and was not updated when `01_extract_features_all_cohorts.py` was finalized with the 15,160 pool.
3. **Origin of `14,928` (in Analysis Plan):**  
   In turn 1, an estimated adult count was computed by excluding de-identified patients older than 89 years ($N = 14,985$) or applying minor sub-filters.
4. **Re-inclusion Verification:** Verified that `MULTI/OTHER` records were **never re-included** after patient selection (`data/ptbxl_54features.csv` contains 0 MULTI/OTHER records and 0 null targets).

---

## 7. Pilot Execution Benchmark (Empirical Time & Memory)

To avoid ungrounded time commitments, 1 outer fold of PTB-XL ($N_{\text{tr}} = 4,000, N_{\text{val}} = 1,000, D = 50$) was benchmarked empirically on the local Apple M4 system:

| Benchmark Component | Execution Scope | Empirically Measured Runtime |
| :--- | :--- | :---: |
| **TabPFN Outer Evaluation** | $N=4,000$ train, $N=1,000$ test, `n_estimators=8` on MPS | **12.83 s** |
| **TabPFN Inner Soft-Target Gen.** | 3-fold inner SGKF ($N \approx 2,667$ train), `n_estimators=4` $\times 3$ on MPS | **14.07 s** |
| **Soft-Target Cross-Entropy MLP** | 100 epochs, PyTorch on MPS | **0.83 s** |
| **Scikit-learn MLP Baseline** | `MLPClassifier(64, 32)` on CPU | **0.09 s** |
| **Scikit-learn MLP Regressor** | `MLPRegressor(128, 64)` on CPU | **0.12 s** |
| **XGBoost Fit & Predict** | `d=4, n=100` trees, 4 CPU threads | **0.39 s** |
| **Total Measured Time for 1 Outer Fold** | Combined teacher + students | **~30.6 s** |

### Projected Empirical Runtime Estimates (No 20.5-min Guarantee)
- **1 Full 5-Fold Cross-Validation Run (1 Seed):** $5 \times 30.6\text{ s} \approx \mathbf{2.55\text{ minutes}}$.
- **Analysis 3A (Matched MLP Ablation on 1 Seed):** $\approx \mathbf{0.5\text{ minutes}}$ (uses existing teacher soft targets).
- **Analysis 3B (5-Seed Repeated CV, 25 Folds):** $25 \times 30.6\text{ s} \approx \mathbf{12.75\text{ minutes}}$ (thermal variance: $12\text{--}16\text{ min}$).
- **Analysis 3C (HYP+STTC Exclusion Sensitivity, 1 Seed):** $\approx \mathbf{2.5\text{ minutes}}$.
- **Analysis 3D (Natural Prevalence Cohort on $N = 15,160$):** TabPFN outer evaluation on $N=12,128$ takes $\sim 4.6\text{ s} \times 8 = 36.8\text{ s}$ per fold $\times 5\text{ folds} \approx \mathbf{3.1\text{ minutes}}$.
- **Statistical Compilation & Bootstrap:** $\approx \mathbf{1.0\text{ minute}}$.
- **Realistic Working Window:** **$19\text{ to }25\text{ minutes}$** depending on background system thermal management.
