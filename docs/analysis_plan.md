# Comprehensive Analysis Plan & Experimental Roadmap
**Date:** 2026-10-20  
**Project:** Two-Lead (Lead I + II) ECG Foundation Model Distillation & Robustness Evaluation  
**Workspace:** `/Volumes/KIOXIA/Projects/gemma3-lora-other`  
**Target Output Directory:** `/Volumes/KIOXIA/Projects/gemma3-lora-other/experiments_phase3/`  
**Auditor / Lead Engineer:** Research Software Engineer (Biomedical Signal Processing & ML Reproducibility)

---

## 1. Scope, Core Research Question & Non-Confirmation Objective

### 1.1. Core Research Framework
The investigation evaluates:
> *"The diagnostic performance of TabPFN and lightweight student models on strictly two-lead (Lead I + II) derived ECG features; the disentanglement of knowledge distillation gains from neural capacity, training loss formulation, and target smoothing; unsupervised target-domain feature standardization under institutional recorder shifts; and the empirical compute–performance trade-offs in edge-to-cloud cascades."*

### 1.2. Objective Integrity & Negative Results Acceptance
- We operate strictly as objective research evaluators. We do **not** seek to prove that Knowledge Distillation is superior.
- If procedure-matched controls show that TabPFN soft targets provide zero benefit over uniform label smoothing or capacity scaling, that is an essential and valuable scientific finding that will be stated prominently.
- If repeated cross-validation diminishes the statistical significance of student gains, or if natural-prevalence evaluation lowers minority-class F1, these will be reported without post-hoc methodological changes.
- All new analyses will be explicitly designated as **post-hoc reviewer-driven robustness analyses**, keeping them transparently distinct from the initial pre-planned study protocol.

---

## 2. Phase 2: Cohort Selection Flow & Label Mapping

### 2.1. Cohort Inclusion, Exclusion & Diagnostic Priority Criteria
```text
+--------------------------------------------------------------------------------------------------+
|                                COHORT SELECTION & FILTERING FLOW                                 |
+------------------------------------+------------------------------------+------------------------+
| PTB-XL (Discovery, N=21,799)       | Chapman-Shaoxing (Ext-1, N=10,646) | CPSC-2018 (Ext-2, N=5,120)
+------------------------------------+------------------------------------+------------------------+
| 1. Age Filter:                     | 1. Age Filter:                     | 1. Length & Channel:   |
|    Keep age >= 18 (21,666 remain)  |    Keep PatientAge >= 18 (10,646)  |    500 Hz, Lead I + II |
| 2. Diagnostic Mapping:             | 2. Multi-Label Mapping:            | 2. Class Isolation:    |
|    SCP statements >= 50% likelihood|    - AFIB/AF -> AFIB/AFL priority  |    Exclude ECTOPIC     |
|    - AFIB/AFLT -> AFIB/AFL priority|    - SR/SB/ST/SI/SA & 0 beats->NORM|    (PAC/PVC: 910 drops)|
|    - Single class NORM, CD, HYP    |    - CD beat codes -> CD          |    Keep single-label:  |
|    - STTC/MI superclass combine    |    - STTC/MI beat codes -> STTC/MI|    NORM, AFIB, CD, STTC|
|    - HYP+STTC -> Step 4 HYP assign |    - LVH/RVH/RAH -> HYP           | 3. Sampling:           |
|    - Multi-class -> Exclude        |    - Complex multi -> Exclude      |    Exact 400 per class |
| 3. Longitudinal Patient Filter:    | 3. Single-Patient Verification:    |    N = 1,600 records   |
|    Strict single-category patients |    Each row is 1 unique patient    | 4. Demographics:       |
|    (Unique single pats: 14,928)    | 4. Sampling Quota:                 |    Zero synthetic demo |
| 4. Balanced Sampling Quota:        |    All 590 CD, all 820 HYP,        |    Evaluated strictly  |
|    1045 NORM, 1045 AF, 1045 CD,    |    1196 NORM, 1197 AF, 1197 STTC   |    under FEAT_50       |
|    1045 STTC/MI, 820 HYP           |    N = 5,000 adult patients        |    (Record-level indep)|
|    N = 5,000 adult patients        |    (Patient-level independent)     |                        |
|    (Patient-level independent)     |                                    |                        |
+------------------------------------+------------------------------------+------------------------+
```

### 2.2. Clinical Equivalence Caveats & 4-Class vs. 5-Class Target Adaptation
1. **AFIB/AFL Priority Rule:** In both PTB-XL and Chapman-Shaoxing, any recording exhibiting atrial fibrillation or flutter is assigned to `AFIB/AFL` regardless of co-occurring bundle branch block or repolarization abnormalities, reflecting the primary hemodynamic urgency of supraventricular tachyarrhythmia.
2. **NORM Criteria Differences:** PTB-XL requires all SCP statements to be strictly normal with diagnostic likelihood $\ge 50\%$. Chapman-Shaoxing defines NORM as sinus rhythm (SR, SB, ST, SI, SA) with zero abnormal beat morphology flags.
3. **STTC/MI Equivalence:** Both cohorts combine localized ischemic ST-T changes and myocardial infarction into a single composite ischemic category (`STTC/MI`), compensating for limited precordial spatial resolution in Leads I and II.
4. **CPSC-2018 4-Class Target Mapping:** CPSC-2018 lacks a primary ventricular hypertrophy (`HYP`) category. When transferring the 5-class PTB-XL model to CPSC-2018:
   - The 5-class model outputs probabilities across all 5 classes $[p_0, p_1, p_2, p_3, p_4]$.
   - We slice the 4 relevant classes $[p_0, p_1, p_2, p_3]$ (`NORM`, `AFIB/AFL`, `CD`, `STTC/MI`) and **strictly re-normalize** them to sum to $1.0$: $\tilde{p}_c = p_c / \sum_{j=0}^3 p_j$.
   - This exact procedure is reported programmatically and transparently.

---

## 3. Phase 3: Priority Robustness Analyses Design

### 3.1. Analysis 3A: Training-Procedure-Matched MLP Control
- **Objective:** Eliminate the methodological confounding where `Distilled MLP` used `MLPRegressor` (MSE on log-probabilities with $R^2$ early stopping) while `Cap-Matched MLP` used `MLPClassifier` (Cross-Entropy with accuracy early stopping).
- **Unified Architecture:**
  - Identical 2-hidden-layer MLP: $(128, 64)$ units with ReLU activations.
  - Identical input features: `FEAT_50` (50 waveform features standardized by fold-training scaler).
  - Identical training optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$, $\text{lr}=10^{-3}$, weight decay $= 10^{-3}$).
  - Identical loss function: **Soft-Target Cross-Entropy Loss**:
    $$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^{K-1} q_{i,c} \log \hat{p}_{i,c}(\theta)$$
    where $\hat{p}_{i}(\theta) = \text{softmax}(z_i(\theta))$, and $q_{i}$ is the target distribution.
  - Three strictly controlled target distributions:
    1. **Hard-Label Target:** $q_{i} = Y_{\text{one-hot}, i} \in \{0, 1\}^K$.
    2. **Uniform Label-Smoothed Target:** $q_{i} = (1 - \epsilon) Y_{\text{one-hot}, i} + \epsilon / K$, with $\epsilon = 0.15$.
    3. **Blended Teacher Soft Target:** $q_{i} = (1 - \alpha) Y_{\text{one-hot}, i} + \alpha P_{\text{teacher}, i}$, with $\alpha = 0.50$ (where $P_{\text{teacher}}$ is generated via inner 3-fold OOF inside the training fold).
  - Identical validation split & early stopping: $10\%$ stratified split of the training fold, monitoring validation Cross-Entropy loss with patience $= 20$ epochs, max epochs $= 350$.
- **Outcome Metric:** Paired bootstrap difference $\Delta\text{Macro F1}$ on the exact same test folds, isolating the pure effect of the target distribution $q_i$.

### 3.2. Analysis 3B: Repeated Outer 5-Fold Cross-Validation (Multi-Seed CV)
- **Objective:** Quantify model ranking stability across 5 independent outer partition seeds (`seeds = [42, 100, 2024, 777, 999]`).
- **Models Evaluated:**
  1. `Teacher TabPFN` ($n_{\text{estimators}}=8$)
  2. `Baseline XGBoost` ($d=3, n=60$)
  3. `Distilled XGBoost` ($d=4, n=100$)
  4. `Cap-Matched XGBoost` ($d=4, n=100$)
  5. `Baseline MLP` ($64, 32$)
  6. `Procedure-Matched Hard-Label MLP` ($128, 64$)
  7. `Procedure-Matched Label-Smoothed MLP` ($128, 64$)
  8. `Procedure-Matched Distilled MLP` ($128, 64$)
- **Statistical Nuance:** We will explicitly report that folds across seeds overlap in training data and cannot be treated as independent degrees of freedom. We will report Mean $\pm$ SD across repetitions, alongside the paired difference $\Delta\text{F1}$ per repetition and the average paired difference.

### 3.3. Analysis 3C: Label Mapping Sensitivity (Exclusion of HYP + STTC Dual-Labeled Cases)
- **Objective:** In PTB-XL, recordings exhibiting both chamber hypertrophy (`HYP`) and repolarization changes (`STTC`) were assigned to `HYP` under Step 4 of the primary mapping (as secondary strain).
- **Alternative Mapping:** Re-run 5-fold CV on PTB-XL where all $198$ dual-labeled `HYP + STTC` cases are strictly **excluded** rather than mapped to `HYP`.
- **Sample Size Impact:**
  - Primary PTB-XL: $N=5,000$ ($1,045$ NORM, $1,045$ AFIB, $1,045$ CD, $1,045$ STTC, $820$ HYP).
  - Alternative PTB-XL: $N=4,802$ ($1,045$ NORM, $1,045$ AFIB, $1,045$ CD, $1,045$ STTC, $622$ HYP).
- **Hypothesis:** Test whether neural student gains or teacher superiority over compact baselines are sensitive to this secondary strain boundary.

### 3.4. Analysis 3D: Natural Prevalence Cohort Sensitivity
- **Objective:** Evaluate compact models and TabPFN on the full single-class adult PTB-XL population ($N = 15,160$ unique patients) without downsampling to artificial class quotas ($8,052$ NORM [$53.1\%$], $3,486$ STTC/MI [$23.0\%$], $1,647$ CD [$10.9\%$], $1,155$ AFIB/AFL [$7.6\%$], $820$ HYP [$5.4\%$]).
- **Computational Boundary & Feasibility:**
  - TabPFN v3 (`tabpfn 9.0.0`) has an internal sample threshold of $1,000,000$ in its checkpoint config. On MPS (`device="mps"`), testing on $N_{\text{train}} = 12,000$ executes natively in $4.59\text{ s}$ per forward pass.
  - On CPU, datasets exceeding $5,000$ samples trigger a `RuntimeError` by default.
  - **Protocol Constraint:** No package or checkpoint updates and no chunked/subsampled ensembles will be deployed without explicit user approval.
  - Metrics reported: Macro F1, balanced accuracy, class-specific recall and precision, and Task 2 binary PPV and NPV under natural prevalence.

---

## 4. Phase 4: Statistical Reporting & Hypothesis Testing Protocol

1. **Family Isolation:**
   - **Family 1 (Primary Benchmark Suite):** The 30 pre-specified comparisons in Table 5 remain unchanged with Bonferroni $\alpha_{\text{FWER}} = 0.05 / 30 \approx 0.00167$.
   - **Family 2 (Procedure-Matched MLP Suite):** 3 new hypothesis tests (Distilled MLP vs Hard-Label MLP, Distilled MLP vs Uniform-Smoothed MLP, Uniform-Smoothed MLP vs Hard-Label MLP) with Bonferroni threshold $\alpha_{\text{FWER}} = 0.05 / 3 \approx 0.0167$.
2. **Bootstrap Specification:**
   - Resampling count: $B = 10,000$ paired resamples with $+1$ continuity adjustment (`seed = 42`).
   - Stratification: Uniform paired sampling of patient prediction pairs (PTB-XL, Chapman-Shaoxing) and record prediction pairs (CPSC-2018).
   - Monte Carlo floor reporting: $p = 0.0002$ will be explicitly documented as the empirical resolution limit of $B = 10,000$ resamples ($2 / 10,001$), not a proof of zero probability mass across zero.

---

## 5. Phase 5 & 6: Domain Adaptation, Cascade, and Feature Protocols

1. **Target Feature Standardization Chain:**
   - Documented exactly as:
     $$\tilde{x}_{\text{target}} = \frac{x_{\text{target}} - \mu_{\text{cal}}}{\sigma_{\text{cal}}} \cdot \sigma_{\text{PTB-XL}} + \mu_{\text{PTB-XL}}$$
   - Binary indicator flags remain strictly unscaled.
   - Assert statement: $\text{Intersection}(\text{Buffer IDs}, \text{Evaluation IDs}) = \emptyset$.
2. **Cascade Evaluation:**
   - Anchor strictly on the **Strictly Isolated Inner-Fold Nested Cascade**, where routing thresholds $\gamma_k^*$ are selected on inner 3-fold OOF predictions without test fold contamination.
3. **Feature Extraction Limits:**
   - Reiterate that QRS duration has a $146.0\text{ ms}$ search-window ceiling and $80.0\text{ ms}$ derivative fallback upper bound.
4. **Clinical LLM Reporting Audit:**
   - Explicitly designate the 20-case Gemma 3 270M LoRA evaluation as a non-blinded, proof-of-concept audit conducted jointly by the authors.

---

## 6. Computational Resource & Execution Budget (Pilot Benchmarked)

Based on empirical single-fold benchmarking on the local Apple M4 system (~$30.6\text{ s}$ per outer fold including TabPFN outer evaluation [$12.8\text{ s}$], inner 3-fold soft-target generation [$14.1\text{ s}$], MLP variants [$0.8\text{ s}$], and XGBoost [$0.4\text{ s}$]):

| Planned Analysis | Target Script | Hardware Target | Empirically Benchmarked Runtime | Memory Footprint | Output Artifact |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Aşama 2: Cohort Selection & Mapping** | `experiments_phase3/01_cohort_flow_audit.py` | CPU (1 thread) | $\approx 15\text{ seconds}$ | $< 500\text{ MB}$ | `results/cohort_selection_flow.json` |
| **Analysis 3A: Procedure-Matched MLP** | `experiments_phase3/02_procedure_matched_mlp.py` | CPU / MPS | $\approx 30\text{ seconds}$ | $< 1\text{ GB}$ | `results/procedure_matched_mlp_results.json` |
| **Analysis 3B: Multi-Seed Repeated CV (5 Seeds)** | `experiments_phase3/03_repeated_cv_multiseed.py` | Apple M4 MPS | $\approx 12\text{--}16\text{ minutes}$ (25 folds) | $\approx 3.5\text{ GB}$ | `results/repeated_cv_5seeds_results.json` |
| **Analysis 3C: HYP+STTC Dual-Label Exclusion** | `experiments_phase3/04_mapping_sensitivity_hyp_sttc.py` | Apple M4 MPS | $\approx 2.5\text{ minutes}$ (5 folds) | $\approx 2.5\text{ GB}$ | `results/hyp_sttc_sensitivity_results.json` |
| **Analysis 3D: Natural Prevalence Cohort** | `experiments_phase3/05_natural_prevalence_cohort.py` | CPU + MPS | $\approx 3.5\text{ minutes}$ ($N=15,160$) | $\approx 4.0\text{ GB}$ | `results/natural_prevalence_results.json` |
| **Phase 4: Statistical Testing & Tables** | `experiments_phase3/06_compile_robustness_tables.py` | CPU | $\approx 1.0\text{ minute}$ | $< 500\text{ MB}$ | `results/robustness_tables_unified.json` |
| **REALISTIC WORKING WINDOW** | — | **Apple M4 SoC** | **$\approx 19\text{ to }25\text{ minutes}$** | **Peak $\approx 4\text{ GB}$** | Clean versioned artifacts |

---

## 7. Approval Request & Next Operational Step
In accordance with your explicit instructions:
> *"Şimdi yalnız Aşama 1 denetimini yap ve ardından öncelikli analiz planını, dosya eksiklerini ve hesaplama maliyetini sun. Ağır deneyleri, başlık değişikliğini veya makaledeki bilimsel sonuçları değiştirmeyi bu ilk rapor tamamlanmadan başlatma."*

We submit this initial audit and plan. Upon your review and approval, we will proceed immediately to execute the versioned scripts in `experiments_phase3/` without modifying or overwriting any existing project files.
