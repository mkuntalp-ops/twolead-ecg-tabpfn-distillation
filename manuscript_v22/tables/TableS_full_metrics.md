**Supplementary Table.** All discrimination and calibration metrics (mean over partitions). AUROC and AUPRC are macro one-versus-rest; NLL, negative log-likelihood; Brier, multiclass Brier score; ECE, expected calibration error with 15 equal-width bins; sensitivity and specificity for NORM versus any other class, with abnormal as the positive class.

| Cohort | Model | Macro F1 (%) | Macro AUROC (%) | Macro AUPRC (%) | NLL (nats) | Brier | ECE, top-label (%) | ECE, class-wise (%) | Sensitivity, abnormal (%) | Specificity, abnormal (%) | HYP recall (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PTB-XL quota-5000 | TabPFN teacher | 74.91 | 93.75 | 81.60 | 0.648 | 0.342 | 1.55 | 1.16 | 91.26 | 78.66 | 61.85 |
| PTB-XL quota-5000 | MLP, hard labels | 71.54 | 91.99 | 77.07 | 0.764 | 0.392 | 4.78 | 2.32 | 90.70 | 75.20 | 59.98 |
| PTB-XL quota-5000 | MLP, smoothing ε = 0.15 | 71.85 | 92.00 | 77.39 | 0.769 | 0.391 | 6.16 | 3.01 | 90.62 | 75.62 | 60.93 |
| PTB-XL quota-5000 | MLP, distillation α = 0.5, T = 1 | 72.86 | 92.60 | 78.58 | 0.718 | 0.375 | 2.74 | 1.60 | 91.20 | 75.62 | 61.90 |
| PTB-XL quota-5000 | MLP, smoothing (tuned) | 71.78 | 91.71 | 76.53 | 0.788 | 0.400 | 7.22 | 3.50 | 91.25 | 74.05 | 61.83 |
| PTB-XL quota-5000 | MLP, entropy-matched (tuned) | 72.09 | 91.37 | 75.87 | 0.863 | 0.432 | 13.67 | 5.98 | 91.47 | 74.41 | 60.68 |
| PTB-XL quota-5000 | MLP, distillation (tuned) | 72.99 | 92.11 | 77.23 | 0.786 | 0.404 | 11.47 | 4.70 | 91.30 | 75.73 | 61.54 |
| PTB-XL quota-5000 | XGBoost, hard labels | 73.60 | 93.01 | 79.52 | 0.687 | 0.360 | 1.77 | 1.13 | 90.96 | 76.27 | 58.80 |
| PTB-XL quota-5000 | XGBoost, smoothing ε = 0.15 | 73.62 | 92.86 | 79.27 | 0.744 | 0.374 | 9.26 | 3.98 | 90.66 | 76.57 | 58.00 |
| PTB-XL quota-5000 | XGBoost, smoothing (tuned) | 73.44 | 92.55 | 78.43 | 0.760 | 0.382 | 9.90 | 4.35 | 90.73 | 76.56 | 57.88 |
| PTB-XL quota-5000 | XGBoost, entropy-matched (tuned) | 73.49 | 92.18 | 77.63 | 0.837 | 0.415 | 16.12 | 7.00 | 90.48 | 77.11 | 58.02 |
| PTB-XL quota-5000 | XGBoost, distillation α = 0.5, T = 1 | 74.11 | 93.19 | 79.96 | 0.677 | 0.356 | 1.61 | 1.08 | 91.07 | 76.94 | 59.34 |
| PTB-XL quota-5000 | XGBoost, distillation (tuned) | 73.87 | 92.60 | 78.44 | 0.733 | 0.380 | 8.36 | 3.50 | 91.01 | 76.75 | 59.05 |
| PTB-XL natural-5000 | TabPFN teacher | 70.80 | 92.94 | 77.24 | 0.563 | 0.294 | 1.50 | 1.01 | 78.52 | 92.70 | 30.15 |
| PTB-XL natural-5000 | MLP, hard labels | 66.47 | 90.61 | 70.87 | 0.682 | 0.346 | 4.83 | 2.41 | 79.30 | 89.11 | 31.70 |
| PTB-XL natural-5000 | MLP, smoothing ε = 0.15 | 66.76 | 90.42 | 71.18 | 0.694 | 0.347 | 6.24 | 3.29 | 77.35 | 90.47 | 31.63 |
| PTB-XL natural-5000 | MLP, distillation α = 0.5, T = 1 | 67.69 | 91.49 | 72.94 | 0.632 | 0.329 | 2.92 | 1.63 | 77.83 | 91.00 | 32.59 |
| PTB-XL natural-5000 | MLP, smoothing (tuned) | 66.84 | 90.04 | 70.56 | 0.733 | 0.361 | 10.66 | 4.96 | 77.20 | 90.41 | 31.19 |
| PTB-XL natural-5000 | MLP, entropy-matched (tuned) | 66.66 | 89.71 | 70.48 | 0.839 | 0.410 | 19.00 | 8.50 | 78.03 | 89.82 | 30.74 |
| PTB-XL natural-5000 | MLP, distillation (tuned) | 67.74 | 90.07 | 70.65 | 0.711 | 0.363 | 10.27 | 4.35 | 77.61 | 90.99 | 32.81 |
| PTB-XL natural-5000 | XGBoost, hard labels | 68.54 | 91.93 | 74.42 | 0.605 | 0.315 | 1.87 | 1.32 | 76.70 | 92.08 | 24.37 |
| PTB-XL natural-5000 | XGBoost, smoothing ε = 0.15 | 68.64 | 91.69 | 74.36 | 0.658 | 0.326 | 9.05 | 3.96 | 76.59 | 92.36 | 23.85 |
| PTB-XL natural-5000 | XGBoost, smoothing (tuned) | 68.88 | 91.33 | 74.00 | 0.651 | 0.325 | 7.73 | 3.42 | 76.70 | 92.43 | 24.67 |
| PTB-XL natural-5000 | XGBoost, entropy-matched (tuned) | 68.71 | 90.80 | 73.79 | 0.693 | 0.341 | 10.94 | 4.82 | 76.55 | 92.36 | 24.07 |
| PTB-XL natural-5000 | XGBoost, distillation α = 0.5, T = 1 | 68.80 | 92.23 | 75.13 | 0.593 | 0.311 | 1.59 | 1.24 | 77.06 | 92.26 | 24.44 |
| PTB-XL natural-5000 | XGBoost, distillation (tuned) | 68.78 | 91.05 | 73.26 | 0.659 | 0.338 | 9.64 | 3.98 | 77.22 | 92.19 | 24.74 |
| PTB-XL eligible-15,160 | TabPFN teacher | 73.05 | 93.85 | 80.10 | 0.523 | 0.275 | 1.14 | 0.71 | 79.59 | 92.80 | 31.95 |
| PTB-XL eligible-15,160 | MLP, hard labels | 70.93 | 92.35 | 76.07 | 0.599 | 0.311 | 2.14 | 1.44 | 80.57 | 89.77 | 38.78 |
| PTB-XL eligible-15,160 | MLP, smoothing ε = 0.15 | 71.26 | 92.11 | 76.06 | 0.637 | 0.317 | 7.93 | 3.48 | 79.77 | 90.81 | 36.59 |
| PTB-XL eligible-15,160 | MLP, distillation α = 0.5, T = 1 | 71.75 | 92.94 | 77.51 | 0.569 | 0.298 | 1.47 | 0.98 | 79.83 | 91.63 | 36.46 |
| PTB-XL eligible-15,160 | MLP, smoothing (tuned) | 71.08 | 91.92 | 75.78 | 0.715 | 0.346 | 16.16 | 6.92 | 80.64 | 89.80 | 36.59 |
| PTB-XL eligible-15,160 | MLP, entropy-matched (tuned) | 70.58 | 90.93 | 74.13 | 0.816 | 0.396 | 22.74 | 9.75 | 79.57 | 90.52 | 34.15 |
| PTB-XL eligible-15,160 | MLP, distillation (tuned) | 71.58 | 91.91 | 76.14 | 0.602 | 0.310 | 5.13 | 2.15 | 79.36 | 91.74 | 35.12 |
| PTB-XL eligible-15,160 | XGBoost, hard labels | 71.65 | 92.99 | 78.02 | 0.555 | 0.290 | 0.64 | 0.66 | 78.83 | 92.24 | 29.88 |
| PTB-XL eligible-15,160 | XGBoost, smoothing ε = 0.15 | 71.83 | 92.66 | 77.71 | 0.617 | 0.304 | 9.62 | 4.03 | 78.63 | 92.47 | 28.90 |
| PTB-XL eligible-15,160 | XGBoost, smoothing (tuned) | 71.65 | 92.19 | 77.25 | 0.606 | 0.303 | 7.54 | 3.23 | 78.47 | 92.47 | 29.02 |
| PTB-XL eligible-15,160 | XGBoost, entropy-matched (tuned) | 71.82 | 92.15 | 76.99 | 0.609 | 0.302 | 7.21 | 3.20 | 78.85 | 92.23 | 29.51 |
| PTB-XL eligible-15,160 | XGBoost, distillation α = 0.5, T = 1 | 71.57 | 93.23 | 78.56 | 0.547 | 0.287 | 1.05 | 0.72 | 78.94 | 92.51 | 28.54 |
| PTB-XL eligible-15,160 | XGBoost, distillation (tuned) | 71.60 | 92.24 | 76.93 | 0.618 | 0.313 | 10.78 | 4.50 | 78.76 | 92.25 | 29.63 |
| Chapman quota-5000 | TabPFN teacher | 68.90 | 91.76 | 74.84 | 0.716 | 0.388 | 1.83 | 1.26 | 88.27 | 77.17 | 38.15 |
| Chapman quota-5000 | MLP, hard labels | 65.28 | 89.09 | 68.33 | 0.864 | 0.446 | 4.66 | 2.41 | 88.89 | 71.12 | 43.54 |
| Chapman quota-5000 | MLP, smoothing ε = 0.15 | 65.44 | 88.95 | 68.50 | 0.872 | 0.448 | 5.51 | 3.18 | 88.43 | 71.59 | 43.85 |
| Chapman quota-5000 | MLP, distillation α = 0.5, T = 1 | 66.13 | 89.91 | 70.25 | 0.811 | 0.429 | 2.63 | 1.67 | 88.51 | 73.28 | 41.80 |
| Chapman quota-5000 | MLP, smoothing (tuned) | 65.57 | 88.73 | 68.29 | 0.890 | 0.454 | 7.19 | 3.78 | 88.53 | 71.76 | 43.85 |
| Chapman quota-5000 | MLP, entropy-matched (tuned) | 65.78 | 88.33 | 67.22 | 0.946 | 0.477 | 12.13 | 5.64 | 88.85 | 71.47 | 43.88 |
| Chapman quota-5000 | MLP, distillation (tuned) | 66.54 | 89.29 | 68.62 | 0.899 | 0.466 | 13.68 | 5.75 | 88.76 | 73.11 | 42.68 |
| Chapman quota-5000 | XGBoost, hard labels | 67.69 | 90.94 | 72.66 | 0.762 | 0.408 | 2.52 | 1.50 | 88.32 | 75.72 | 39.34 |
| Chapman quota-5000 | XGBoost, smoothing ε = 0.15 | 67.54 | 90.80 | 72.51 | 0.814 | 0.418 | 7.44 | 3.68 | 88.22 | 75.85 | 38.20 |
| Chapman quota-5000 | XGBoost, smoothing (tuned) | 67.61 | 90.49 | 71.90 | 0.817 | 0.421 | 6.77 | 3.47 | 88.33 | 76.20 | 38.80 |
| Chapman quota-5000 | XGBoost, entropy-matched (tuned) | 67.67 | 90.62 | 72.12 | 0.817 | 0.420 | 6.64 | 3.53 | 87.95 | 76.89 | 37.59 |
| Chapman quota-5000 | XGBoost, distillation α = 0.5, T = 1 | 67.97 | 91.22 | 73.20 | 0.749 | 0.402 | 1.76 | 1.27 | 88.37 | 77.11 | 38.39 |
| Chapman quota-5000 | XGBoost, distillation (tuned) | 67.59 | 90.51 | 71.59 | 0.797 | 0.422 | 6.15 | 2.97 | 88.23 | 76.51 | 38.71 |
| Chapman eligible-9,315 | TabPFN teacher | 64.63 | 92.71 | 72.00 | 0.547 | 0.288 | 1.47 | 0.90 | 80.56 | 93.93 | 12.20 |
| Chapman eligible-9,315 | MLP, hard labels | 64.43 | 90.35 | 66.49 | 0.657 | 0.330 | 2.33 | 2.04 | 82.60 | 89.60 | 25.12 |
| Chapman eligible-9,315 | MLP, smoothing ε = 0.15 | 64.16 | 89.80 | 66.13 | 0.692 | 0.342 | 8.03 | 4.02 | 82.44 | 89.60 | 25.73 |
| Chapman eligible-9,315 | MLP, distillation α = 0.5, T = 1 | 64.89 | 91.15 | 68.00 | 0.614 | 0.316 | 1.53 | 1.13 | 81.20 | 92.37 | 22.32 |
| Chapman eligible-9,315 | MLP, smoothing (tuned) | 64.09 | 89.40 | 65.58 | 0.699 | 0.345 | 7.81 | 3.92 | 82.02 | 90.11 | 24.15 |
| Chapman eligible-9,315 | MLP, entropy-matched (tuned) | 64.83 | 89.79 | 65.87 | 0.756 | 0.370 | 14.46 | 6.62 | 83.65 | 88.16 | 27.56 |
| Chapman eligible-9,315 | MLP, distillation (tuned) | 65.65 | 89.88 | 66.69 | 0.683 | 0.348 | 10.15 | 4.34 | 82.76 | 90.44 | 26.71 |
| Chapman eligible-9,315 | XGBoost, hard labels | 63.76 | 91.86 | 69.42 | 0.586 | 0.306 | 1.79 | 1.03 | 80.02 | 93.18 | 15.12 |
| Chapman eligible-9,315 | XGBoost, smoothing ε = 0.15 | 64.05 | 91.53 | 69.32 | 0.647 | 0.318 | 8.70 | 3.86 | 79.98 | 93.16 | 14.02 |
| Chapman eligible-9,315 | XGBoost, smoothing (tuned) | 63.97 | 90.74 | 68.21 | 0.682 | 0.333 | 11.71 | 5.09 | 79.92 | 93.25 | 14.39 |
| Chapman eligible-9,315 | XGBoost, entropy-matched (tuned) | 63.99 | 91.39 | 69.06 | 0.650 | 0.320 | 8.65 | 3.76 | 79.69 | 93.67 | 13.90 |
| Chapman eligible-9,315 | XGBoost, distillation α = 0.5, T = 1 | 63.91 | 92.17 | 70.03 | 0.574 | 0.301 | 1.42 | 1.00 | 80.32 | 93.84 | 13.54 |
| Chapman eligible-9,315 | XGBoost, distillation (tuned) | 64.82 | 90.74 | 68.19 | 0.675 | 0.339 | 13.18 | 5.40 | 80.54 | 93.28 | 15.49 |
