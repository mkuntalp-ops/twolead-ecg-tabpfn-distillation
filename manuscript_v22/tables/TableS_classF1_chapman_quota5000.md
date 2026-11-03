**Supplementary Table.** Class-specific F1 (%), Chapman quota-5000 (mean over partitions).

| Model | NORM | AFIB/AFL | CD | STTC/MI | HYP |
|---|---|---|---|---|---|
| TabPFN teacher | 72.0 | 92.2 | 68.9 | 65.6 | 45.8 |
| MLP, hard labels | 68.9 | 88.9 | 61.7 | 60.5 | 46.3 |
| MLP, smoothing ε = 0.15 | 68.7 | 89.0 | 61.8 | 61.0 | 46.7 |
| MLP, distillation α = 0.5, T = 1 | 69.8 | 89.5 | 62.9 | 62.1 | 46.2 |
| MLP, distillation (tuned) | 70.0 | 89.8 | 63.3 | 62.4 | 47.1 |
| XGBoost, hard labels | 71.1 | 90.7 | 66.7 | 65.0 | 44.9 |
