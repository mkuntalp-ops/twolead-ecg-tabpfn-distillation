**Table 2.** Outer-test Macro F1 (%) by cohort. Five-partition cohorts: mean ± SD over the five partition-level scores (each computed on all N pooled out-of-fold predictions); single-partition cohorts: point estimate. 'Tuned' rows use the configuration chosen in each outer fold on the student-validation set.

| Model | PTB-XL quota-5000 | PTB-XL natural-5000 | PTB-XL eligible-15,160 | Chapman quota-5000 | Chapman eligible-9,315 |
|---|---|---|---|---|---|
| TabPFN teacher | 74.91 ± 0.34 | 70.80 ± 0.42 | 73.05 | 68.90 ± 0.45 | 64.63 |
| MLP, hard labels | 71.54 ± 0.46 | 66.47 ± 0.63 | 70.93 | 65.28 ± 0.40 | 64.43 |
| MLP, smoothing ε = 0.15 | 71.85 ± 0.32 | 66.76 ± 0.62 | 71.26 | 65.44 ± 0.49 | 64.16 |
| MLP, distillation α = 0.5, T = 1 | 72.86 ± 0.52 | 67.69 ± 0.61 | 71.75 | 66.13 ± 0.34 | 64.89 |
| MLP, smoothing (tuned) | 71.78 ± 0.27 | 66.84 ± 0.36 | 71.08 | 65.57 ± 0.30 | 64.09 |
| MLP, entropy-matched (tuned) | 72.09 ± 0.32 | 66.66 ± 0.75 | 70.58 | 65.78 ± 0.10 | 64.83 |
| MLP, distillation (tuned) | 72.99 ± 0.35 | 67.74 ± 0.45 | 71.58 | 66.54 ± 0.34 | 65.65 |
| XGBoost, hard labels | 73.60 ± 0.23 | 68.54 ± 0.18 | 71.65 | 67.69 ± 0.39 | 63.76 |
| XGBoost, smoothing (tuned) | 73.44 ± 0.21 | 68.88 ± 0.31 | 71.65 | 67.61 ± 0.23 | 63.97 |
| XGBoost, entropy-matched (tuned) | 73.49 ± 0.46 | 68.71 ± 0.54 | 71.82 | 67.67 ± 0.28 | 63.99 |
| XGBoost, distillation α = 0.5, T = 1 | 74.11 ± 0.26 | 68.80 ± 0.15 | 71.57 | 67.97 ± 0.47 | 63.91 |
| XGBoost, distillation (tuned) | 73.87 ± 0.32 | 68.78 ± 0.25 | 71.60 | 67.59 ± 0.28 | 64.82 |
