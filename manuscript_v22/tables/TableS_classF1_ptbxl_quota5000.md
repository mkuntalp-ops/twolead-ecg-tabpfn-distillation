**Supplementary Table.** Class-specific F1 (%), PTB-XL quota-5000 (mean over partitions).

| Model | NORM | AFIB/AFL | CD | STTC/MI | HYP |
|---|---|---|---|---|---|
| TabPFN teacher | 74.3 | 93.8 | 76.1 | 65.2 | 65.3 |
| MLP, hard labels | 71.5 | 90.9 | 71.9 | 61.4 | 62.1 |
| MLP, smoothing ε = 0.15 | 71.6 | 91.1 | 72.4 | 61.5 | 62.6 |
| MLP, distillation α = 0.5, T = 1 | 72.4 | 91.6 | 73.3 | 63.0 | 64.0 |
| MLP, distillation (tuned) | 72.6 | 91.9 | 73.5 | 63.1 | 63.9 |
| XGBoost, hard labels | 72.5 | 93.3 | 75.1 | 64.0 | 63.2 |
