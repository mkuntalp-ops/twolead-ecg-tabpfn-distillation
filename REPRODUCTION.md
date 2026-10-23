# Re-execution record (23 October 2026)

The primary analyses were re-executed from the archived scripts in the original analysis environment
(`tabpfn 9.0.0`, `torch 2.14.0`, Python 3.12.14, Apple-silicon MPS, macOS) and compared with the stored
outputs in this repository. Each rerun used an isolated working directory so that no stored file could be
overwritten; the only script edits were the output root (`BASE_DIR`) for Phases 3B and 3C. The Phase 3D
script resolves paths relative to itself and ran unmodified.

| Check | Input | Result |
|---|---|---|
| FEAT_50 re-extraction from raw PTB-XL 500 Hz records | 300 patients (60 per class, random seed 2026) from the 15,160-patient table | 15,000/15,000 values identical (max abs. difference 2.3×10⁻¹³) |
| Phase 3B vs Phase 3D feature tables | 5,000 overlapping patients | identical for all 50 features |
| Phase 3B, all five partition seeds (42–46) | `data/ptbxl_54features.csv` | all OOF probabilities and predictions identical to `predictions/phase3b_repeated_oof_predictions.npz` (13/13 arrays, max abs. difference 0); 48/48 regenerated initial weight files identical |
| Phase 3C (HYP+STTC exclusion, evaluations A and B) | same | all 11 arrays of `phase3c_oof_predictions.npz` identical; excluded-record list identical (496 records; only the print order of Python sets in the `superclasses` column differs) |
| Phase 3D (N = 15,160) | `ptbxl_eligible_cohort_15160_features.csv` | all 13 arrays of `phase3d_full_cohort_oof_predictions.npz` identical (max abs. difference 0); 10/10 initial weight files identical; printed bootstrap tail estimates match Table 6 |

Wall-clock time on the original workstation: Phase 3B seed 42 ≈ 3 min; Phase 3D ≈ 20 min.

Not re-executed: the Phase 3B 10,000-replicate bootstrap script
(`experiments_phase3/04_statistical_closing_and_blocked_bootstrap.py`; `verify_reported_metrics.py` gives an
independent approximation), the full feature extraction of all 15,160 records, and the legacy analyses in
`legacy/`. Bit-identical results were obtained on the same hardware and software; other platforms or library
versions may produce small numerical differences.
