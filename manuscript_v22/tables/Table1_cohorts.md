**Table 1.** Cohorts. Each patient contributes one recording, so patient-grouped and record-level splits coincide. The PTB-XL quota-5000 and natural-5000 cohorts are subsets of the PTB-XL eligible pool, and Chapman quota-5000 is a subset of the Chapman eligible pool; overlapping cohorts are not independent samples.

| Cohort | Source | Selection | N (patients) | NORM / AFIB/AFL / CD / STTC/MI / HYP | Partition seeds |
|---|---|---|---|---|---|
| PTB-XL quota-5000 | PTB-XL eligible pool | Class-quota subsample (seed 42): 1,045 per class, all 820 HYP | 5,000 | 1,045 / 1,045 / 1,045 / 1,045 / 820 | 42–46 |
| PTB-XL natural-5000 | PTB-XL eligible pool | Class-stratified random subsample (seed 2026), natural proportions | 5,000 | 2,656 / 381 / 543 / 1,150 / 270 | 42–46 |
| PTB-XL eligible-15,160 | PTB-XL, 21,799 records / 18,869 patients | All eligible adults, first eligible record per patient | 15,160 | 8,052 / 1,155 / 1,647 / 3,486 / 820 | 42 |
| Chapman quota-5000 | Chapman-Shaoxing eligible pool | Class-quota subsample (seed 42): all 590 CD and 820 HYP; 1,196 NORM, 1,197 AFIB/AFL and 1,197 STTC/MI drawn at random | 5,000 | 1,196 / 1,197 / 590 / 1,197 / 820 | 42–46 |
| Chapman eligible-9,315 | Chapman-Shaoxing, 10,646 records | All eligible adults (≥ 18 years) with one assignable class | 9,315 | 4,299 / 2,225 / 590 / 1,381 / 820 | 42 |
