**Supplementary Table.** Class-specific F1 (%), Chapman eligible-9,315 (mean over partitions).

| Model | NORM | AFIB/AFL | CD | STTC/MI | HYP |
|---|---|---|---|---|---|
| TabPFN teacher | 86.7 | 93.9 | 61.5 | 61.4 | 19.6 |
| MLP, hard labels | 85.4 | 92.0 | 58.1 | 55.8 | 30.9 |
| MLP, smoothing ε = 0.15 | 85.3 | 91.8 | 57.0 | 55.5 | 31.2 |
| MLP, distillation α = 0.5, T = 1 | 86.2 | 92.2 | 58.1 | 57.8 | 30.2 |
| MLP, distillation (tuned) | 85.9 | 92.4 | 57.5 | 58.7 | 33.7 |
| XGBoost, hard labels | 86.1 | 92.8 | 57.8 | 59.7 | 22.4 |
