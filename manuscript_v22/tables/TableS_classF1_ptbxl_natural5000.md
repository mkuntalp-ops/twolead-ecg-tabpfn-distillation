**Supplementary Table.** Class-specific F1 (%), PTB-XL natural-5000 (mean over partitions).

| Model | NORM | AFIB/AFL | CD | STTC/MI | HYP |
|---|---|---|---|---|---|
| TabPFN teacher | 87.6 | 89.9 | 66.4 | 68.8 | 41.3 |
| MLP, hard labels | 85.9 | 83.2 | 59.6 | 65.1 | 38.5 |
| MLP, smoothing ε = 0.15 | 86.0 | 84.1 | 60.0 | 64.7 | 39.0 |
| MLP, distillation α = 0.5, T = 1 | 86.4 | 85.2 | 60.5 | 65.5 | 40.8 |
| MLP, distillation (tuned) | 86.3 | 85.3 | 60.9 | 65.3 | 40.8 |
| XGBoost, hard labels | 86.6 | 89.3 | 63.4 | 68.0 | 35.3 |
