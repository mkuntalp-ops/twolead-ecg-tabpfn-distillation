**Supplementary Table.** Post hoc summary, added after the results had been seen: partition-averaged paired differences in Macro F1 (pp) with 95% patient-blocked bootstrap intervals (B = 10,000; in five-partition cohorts the same resampled patients are used in every partition). The point estimates equal the means in Table 4; the intervals reflect patient sampling only, not partitioning or training variability, and are not adjusted for multiplicity.

| Student | Contrast | PTB-XL quota-5000 | PTB-XL natural-5000 | PTB-XL eligible-15,160 | Chapman quota-5000 | Chapman eligible-9,315 |
|---|---|---|---|---|---|---|
| MLP | Distillation − smoothing (tuned) | +1.21 [+0.81, +1.61] | +0.90 [+0.33, +1.48] | +0.51 [-0.08, +1.10] | +0.97 [+0.52, +1.43] | +1.56 [+0.74, +2.34] |
| MLP | Distillation − entropy-matched (tuned) | +0.90 [+0.53, +1.27] | +1.08 [+0.46, +1.70] | +1.01 [+0.38, +1.63] | +0.75 [+0.32, +1.20] | +0.82 [+0.01, +1.60] |
| MLP | Distillation (tuned) − hard labels | +1.44 [+1.04, +1.85] | +1.27 [+0.72, +1.83] | +0.65 [+0.09, +1.20] | +1.25 [+0.78, +1.73] | +1.21 [+0.35, +2.04] |
| MLP | Smoothing (tuned) − hard labels | +0.24 [-0.09, +0.56] | +0.37 [-0.15, +0.90] | +0.15 [-0.49, +0.76] | +0.29 [-0.12, +0.69] | -0.34 [-1.09, +0.40] |
| MLP | Entropy-matched (tuned) − hard labels | +0.54 [+0.19, +0.91] | +0.19 [-0.40, +0.81] | -0.35 [-1.00, +0.29] | +0.50 [+0.06, +0.94] | +0.40 [-0.45, +1.25] |
| MLP | Distillation − smoothing (fixed α = 0.5, ε = 0.15) | +1.01 [+0.66, +1.36] | +0.93 [+0.39, +1.45] | +0.49 [-0.14, +1.11] | +0.69 [+0.25, +1.12] | +0.73 [-0.05, +1.52] |
| MLP | Distillation (fixed) − hard labels | +1.31 [+0.96, +1.67] | +1.21 [+0.74, +1.72] | +0.82 [+0.25, +1.38] | +0.85 [+0.42, +1.28] | +0.46 [-0.37, +1.30] |
| XGBoost | Distillation − smoothing (tuned) | +0.43 [+0.09, +0.76] | -0.10 [-0.63, +0.44] | -0.05 [-0.55, +0.45] | -0.01 [-0.40, +0.37] | +0.85 [+0.19, +1.51] |
| XGBoost | Distillation − entropy-matched (tuned) | +0.38 [+0.04, +0.71] | +0.07 [-0.45, +0.59] | -0.21 [-0.71, +0.28] | -0.08 [-0.48, +0.32] | +0.82 [+0.15, +1.49] |
| XGBoost | Distillation (tuned) − hard labels | +0.26 [-0.05, +0.58] | +0.25 [-0.26, +0.75] | -0.05 [-0.58, +0.48] | -0.09 [-0.46, +0.27] | +1.05 [+0.36, +1.76] |
| XGBoost | Smoothing (tuned) − hard labels | -0.16 [-0.44, +0.12] | +0.34 [-0.10, +0.78] | -0.00 [-0.45, +0.45] | -0.08 [-0.39, +0.23] | +0.21 [-0.51, +0.93] |
| XGBoost | Entropy-matched (tuned) − hard labels | -0.12 [-0.42, +0.19] | +0.17 [-0.28, +0.63] | +0.16 [-0.29, +0.61] | -0.02 [-0.37, +0.34] | +0.23 [-0.47, +0.93] |
| XGBoost | Distillation − smoothing (fixed α = 0.5, ε = 0.15) | +0.49 [+0.12, +0.86] | +0.16 [-0.36, +0.68] | -0.26 [-0.76, +0.24] | +0.43 [+0.03, +0.85] | -0.14 [-0.91, +0.62] |
| XGBoost | Distillation (fixed) − hard labels | +0.51 [+0.19, +0.82] | +0.26 [-0.18, +0.72] | -0.08 [-0.56, +0.41] | +0.29 [-0.09, +0.67] | +0.15 [-0.55, +0.84] |
