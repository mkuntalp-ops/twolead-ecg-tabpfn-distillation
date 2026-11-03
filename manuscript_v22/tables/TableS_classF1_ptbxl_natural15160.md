**Supplementary Table.** Class-specific F1 (%), PTB-XL eligible-15,160 (mean over partitions).

| Model | NORM | AFIB/AFL | CD | STTC/MI | HYP |
|---|---|---|---|---|---|
| TabPFN teacher | 88.0 | 91.9 | 71.0 | 70.8 | 43.4 |
| MLP, hard labels | 86.8 | 87.4 | 67.4 | 67.8 | 45.3 |
| MLP, smoothing ε = 0.15 | 87.0 | 88.1 | 67.6 | 68.3 | 45.4 |
| MLP, distillation α = 0.5, T = 1 | 87.5 | 88.1 | 68.7 | 68.9 | 45.6 |
| MLP, distillation (tuned) | 87.4 | 88.5 | 68.9 | 68.8 | 44.3 |
| XGBoost, hard labels | 87.5 | 91.3 | 69.8 | 69.2 | 40.6 |
