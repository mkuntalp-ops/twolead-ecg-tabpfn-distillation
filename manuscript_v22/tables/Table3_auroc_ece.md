**Table 3.** Macro one-versus-rest AUROC (%) and top-label expected calibration error (ECE, %, 15 equal-width bins) in three cohorts; all cohorts and further metrics are in Supplementary Table {{ST:full}}. Format as in Table 2.

| Model | PTB-XL quota-5000 AUROC | PTB-XL quota-5000 ECE | PTB-XL eligible-15,160 AUROC | PTB-XL eligible-15,160 ECE | Chapman eligible-9,315 AUROC | Chapman eligible-9,315 ECE |
|---|---|---|---|---|---|---|
| TabPFN teacher | 93.75 ± 0.05 | 1.55 ± 0.24 | 93.85 | 1.14 | 92.71 | 1.47 |
| MLP, hard labels | 91.99 ± 0.26 | 4.78 ± 1.09 | 92.35 | 2.14 | 90.35 | 2.33 |
| MLP, smoothing ε = 0.15 | 92.00 ± 0.08 | 6.16 ± 0.84 | 92.11 | 7.93 | 89.80 | 8.03 |
| MLP, distillation α = 0.5, T = 1 | 92.60 ± 0.10 | 2.74 ± 0.30 | 92.94 | 1.47 | 91.15 | 1.53 |
| MLP, smoothing (tuned) | 91.71 ± 0.13 | 7.22 ± 2.85 | 91.92 | 16.16 | 89.40 | 7.81 |
| MLP, entropy-matched (tuned) | 91.37 ± 0.07 | 13.67 ± 6.59 | 90.93 | 22.74 | 89.79 | 14.46 |
| MLP, distillation (tuned) | 92.11 ± 0.36 | 11.47 ± 6.01 | 91.91 | 5.13 | 89.88 | 10.15 |
| XGBoost, hard labels | 93.01 ± 0.09 | 1.77 ± 0.29 | 92.99 | 0.64 | 91.86 | 1.79 |
| XGBoost, smoothing (tuned) | 92.55 ± 0.19 | 9.90 ± 2.82 | 92.19 | 7.54 | 90.74 | 11.71 |
| XGBoost, entropy-matched (tuned) | 92.18 ± 0.32 | 16.12 ± 2.53 | 92.15 | 7.21 | 91.39 | 8.65 |
| XGBoost, distillation α = 0.5, T = 1 | 93.19 ± 0.05 | 1.61 ± 0.29 | 93.23 | 1.05 | 92.17 | 1.42 |
| XGBoost, distillation (tuned) | 92.60 ± 0.45 | 8.36 ± 2.74 | 92.24 | 10.78 | 90.74 | 13.18 |
