**Table 4.** Paired differences in Macro F1 (percentage points), summarized as specified in the analysis plan; all contrasts are exploratory. Five-partition cohorts: mean ± SD of the five partition-level differences, followed by (number of partitions with a positive difference, number whose 95% patient-blocked bootstrap interval lay above zero, number whose interval lay below zero). Single-partition cohorts: difference with its 95% patient-blocked bootstrap interval. Bootstrap B = 2,000 per partition. Intervals are conditional on the fitted models and fixed partitions and are not adjusted for multiplicity.

| Student | Contrast | PTB-XL quota-5000 | PTB-XL natural-5000 | PTB-XL eligible-15,160 | Chapman quota-5000 | Chapman eligible-9,315 |
|---|---|---|---|---|---|---|
| MLP | Distillation − smoothing (tuned) | +1.21 ± 0.35 (5, 4, 0) | +0.90 ± 0.57 (5, 1, 0) | +0.51 [-0.04, +1.09] | +0.97 ± 0.46 (5, 3, 0) | +1.56 [+0.74, +2.41] |
| MLP | Distillation − entropy-matched (tuned) | +0.90 ± 0.35 (5, 3, 0) | +1.08 ± 1.11 (4, 4, 0) | +1.01 [+0.37, +1.62] | +0.75 ± 0.34 (5, 3, 0) | +0.82 [+0.01, +1.64] |
| MLP | Distillation (tuned) − hard labels | +1.44 ± 0.23 (5, 5, 0) | +1.27 ± 0.90 (5, 3, 0) | +0.65 [+0.11, +1.22] | +1.25 ± 0.47 (5, 3, 0) | +1.21 [+0.37, +2.07] |
| MLP | Smoothing (tuned) − hard labels | +0.24 ± 0.32 (3, 0, 0) | +0.37 ± 0.94 (3, 1, 0) | +0.15 [-0.45, +0.76] | +0.29 ± 0.30 (4, 0, 0) | -0.34 [-1.12, +0.40] |
| MLP | Entropy-matched (tuned) − hard labels | +0.54 ± 0.48 (5, 1, 0) | +0.19 ± 0.84 (4, 0, 0) | -0.35 [-0.98, +0.26] | +0.50 ± 0.30 (5, 0, 0) | +0.40 [-0.43, +1.18] |
| MLP | Distillation − smoothing (fixed α = 0.5, ε = 0.15) | +1.01 ± 0.28 (5, 4, 0) | +0.93 ± 0.54 (5, 1, 0) | +0.49 [-0.14, +1.10] | +0.69 ± 0.59 (4, 2, 0) | +0.73 [-0.06, +1.51] |
| MLP | Distillation (fixed) − hard labels | +1.31 ± 0.40 (5, 5, 0) | +1.21 ± 0.77 (5, 3, 0) | +0.82 [+0.24, +1.38] | +0.85 ± 0.37 (5, 2, 0) | +0.46 [-0.41, +1.27] |
| XGBoost | Distillation − smoothing (tuned) | +0.43 ± 0.22 (5, 1, 0) | -0.10 ± 0.35 (2, 0, 0) | -0.05 [-0.58, +0.49] | -0.01 ± 0.36 (3, 0, 0) | +0.85 [+0.17, +1.50] |
| XGBoost | Distillation − entropy-matched (tuned) | +0.38 ± 0.17 (5, 0, 0) | +0.07 ± 0.60 (3, 0, 0) | -0.21 [-0.70, +0.26] | -0.08 ± 0.41 (3, 0, 0) | +0.82 [+0.17, +1.44] |
| XGBoost | Distillation (tuned) − hard labels | +0.26 ± 0.27 (4, 0, 0) | +0.25 ± 0.30 (4, 0, 0) | -0.05 [-0.57, +0.49] | -0.09 ± 0.38 (3, 0, 0) | +1.05 [+0.33, +1.74] |
| XGBoost | Smoothing (tuned) − hard labels | -0.16 ± 0.27 (1, 0, 0) | +0.34 ± 0.19 (5, 0, 0) | -0.00 [-0.46, +0.45] | -0.08 ± 0.35 (2, 0, 0) | +0.21 [-0.51, +0.96] |
| XGBoost | Entropy-matched (tuned) − hard labels | -0.12 ± 0.36 (2, 0, 1) | +0.17 ± 0.49 (3, 0, 0) | +0.16 [-0.28, +0.58] | -0.02 ± 0.34 (2, 0, 0) | +0.23 [-0.48, +0.95] |
| XGBoost | Distillation − smoothing (fixed α = 0.5, ε = 0.15) | +0.49 ± 0.43 (5, 2, 0) | +0.16 ± 0.45 (3, 0, 0) | -0.26 [-0.73, +0.23] | +0.43 ± 0.27 (5, 1, 0) | -0.14 [-0.91, +0.59] |
| XGBoost | Distillation (fixed) − hard labels | +0.51 ± 0.28 (5, 2, 0) | +0.26 ± 0.30 (4, 0, 0) | -0.08 [-0.55, +0.42] | +0.29 ± 0.36 (4, 0, 0) | +0.15 [-0.57, +0.85] |
