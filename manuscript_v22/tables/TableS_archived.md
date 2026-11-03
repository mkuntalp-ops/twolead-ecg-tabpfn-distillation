**Supplementary Table.** Earlier fixed-configuration analyses (ε = 0.15, α = 0.5, T = 1) re-run with the original (v1) and revised (v2) features; scripts, partitions and seeds unchanged. The eligible-15,160 analysis used a different MLP procedure from the other two (batch normalization and dropout 0.1, AdamW with weight decay 1e-4, at most 100 epochs, early stopping on validation cross-entropy, inner teacher folds not grouped by patient, which is equivalent here because each patient has one record); this was not stated correctly in the earlier report. The main analyses of the present paper use one procedure throughout.

| Features | Analysis | Distilled / smoothed / hard (Macro F1, %) | Distilled − smoothed (pp) | Distilled − hard (pp) |
|---|---|---|---|---|
| v1 | PTB-XL quota-5000, 5 partitions (inner teacher 4 estimators) | 73.29 / 71.94 / 71.96 | +1.35 ± 0.71 (mean ± SD of partitions) | +1.33 ± 0.30 |
| v1 | PTB-XL quota-5000 without HYP+STTC co-occurrence (N = 4,504), retrained, partition 42 | 70.58 / 69.44 / 69.22 | +1.14 [+0.18, +2.12] | +1.36 [+0.45, +2.27] |
| v1 | PTB-XL eligible-15,160, partition 42 (different MLP procedure, see note) | 71.34 / 71.19 / 70.38 | +0.15 (p = 0.641) | +0.96 (p = 0.002) |
| v2 | PTB-XL quota-5000, 5 partitions (inner teacher 4 estimators) | 72.76 / 71.87 / 71.63 | +0.88 ± 0.26 (mean ± SD of partitions) | +1.13 ± 0.31 |
| v2 | PTB-XL quota-5000 without HYP+STTC co-occurrence (N = 4,504), retrained, partition 42 | 70.30 / 69.30 / 69.64 | +0.99 [+0.14, +1.86] | +0.66 [-0.20, +1.50] |
| v2 | PTB-XL eligible-15,160, partition 42 (different MLP procedure, see note) | 71.04 / 70.98 / 70.31 | +0.05 (p = 0.869) | +0.73 (p = 0.018) |
