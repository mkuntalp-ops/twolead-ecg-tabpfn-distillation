# Teacher–Student Distillation of TabPFN for Retrospective Two-Lead ECG Features

Code, derived feature tables, fold manifests and out-of-fold (OOF) predictions for the manuscript

> Kuntalp M. *Teacher–Student Distillation of TabPFN for Retrospective Two-Lead ECG Features:
> Procedure-Matched Evaluation and Cohort-Composition Sensitivity.* Submitted to *Physiological Measurement*.

The study compares a TabPFN-v3 teacher with multilayer-perceptron students trained on hard labels,
uniform label smoothing (ε = 0.15) or mixed hard/inner-OOF teacher targets (α = 0.5), using 50
Lead I/II features extracted from PTB-XL. All splits are patient-grouped (`StratifiedGroupKFold`).

## Repository layout

| Path | Content | Manuscript |
|---|---|---|
| `experiments_phase3/03_procedure_matched_repeated_eval.py` | Phase 3B: five outer partition seeds (42–46) × five folds, 5,000 quota-selected patients | Table 3 |
| `experiments_phase3/04_statistical_closing_and_blocked_bootstrap.py` | Phase 3B paired patient-blocked bootstrap (B = 10,000) | Table 3 |
| `experiments_phase3/phase3c_sensitivity/` | Phase 3C: HYP+STTC exclusion (N = 4,504), evaluations A and B, excluded-record list, fold-role manifest | Table 4 |
| `experiments_phase3/phase3d_full_cohort/` | Phase 3D: eligibility-filtered cohort without quotas (N = 15,160): feature extraction, evaluation, cohort manifest, confusion matrices, bootstrap table | Tables 5–6, S4 |
| `experiments_phase3/phase3b_weights/`, `.../phase3d_weights/` | Shared initial student weights per seed/fold | Methods 2.4 |
| `experiments_phase3/archive_phase3a_pilot/` | Phase 3A pilot of the matched protocol (archived; not a pre-registration) | Results 3.1 |
| `predictions/phase3b_repeated_oof_predictions.npz` | Phase 3B OOF probabilities/predictions for all five models and seeds | Table 3 |
| `data/ptbxl_54features.csv`, `data/chapman_54features.csv` | Derived per-patient feature tables (selected cohorts) | Table 1 |
| `legacy/` | Earlier multi-cohort benchmarks, external transfer, Gemma-3 270M reporting interface and reviewer-round scripts | Supplementary S5, S7 |
| `predictions/oof_task*`, `predictions/boot10k_*`, `predictions/table5_oof_predictions_ptbxl_chapman_only.npz` | Legacy OOF predictions and bootstrap replicates | Supplementary S7 |
| `results/`, `figures/` | Legacy result JSONs and the five legacy figures (Supplementary Figures S1–S5) | Supplementary |
| `data_ecg_clinical/` | Legacy LoRA instruction-tuning data generated from PTB-XL features | Supplementary S7 |
| `docs/` | Analysis plan and internal audit report | — |
| `figures_main/` | `make_main_figures.py` and the main-text Figures 1–3 (PDF/PNG), generated from the stored outputs in this repository | Figures 1–3 |
| `verify_reported_metrics.py` | Recomputes the headline metrics from the stored OOF files | — |
| `SHA256SUMS` | SHA-256 of every file in this repository | — |
| `REPRODUCTION.md` | Record of the 23 October 2026 re-execution (bit-identical OOF outputs) | — |

## Quick verification (no retraining)

```bash
pip install numpy scikit-learn
python verify_reported_metrics.py            # 2,000 bootstrap replicates, ~minutes on a laptop
python verify_reported_metrics.py --bootstrap 10000
shasum -a 256 -c SHA256SUMS
```

Point estimates (per-seed and mean Macro F1, Phase 3C/3D class metrics, binary sensitivity/specificity)
must match the manuscript exactly. Bootstrap intervals use an independent random seed and may differ in
the second decimal.

## Re-running the experiments

Scripts are archived **byte-identical to the versions that produced the reported results** and therefore
contain absolute paths from the original workstation (e.g. `/Volumes/KIOXIA/Projects/gemma3-lora-other`,
`/Volumes/KIOXIA/Datasets/ptb-xl/`, `/Volumes/KIOXIA/AI Models/tabpfn/`). To re-run them, edit the
`BASE_DIR`, dataset and checkpoint constants at the top of each script.

Requirements:

- Python 3.12; package versions in `requirements.txt` (recorded from the analysis environment on
  23 October 2026; `tabpfn 9.0.0` was installed on 21 September 2026, before the experiments).
- PTB-XL raw data (`ptbxl_database.csv`, `scp_statements.csv`, 500 Hz records) from PhysioNet. The exact
  PTB-XL release used was not recorded; the current release is v1.0.3.
- TabPFN-v3 checkpoint `tabpfn-v3-classifier-v3_20260417_multiclass.ckpt` (not redistributed; obtain from
  Prior Labs under its licence). Standalone evaluation used 8 estimators; inner target generation 4.
- Seeds: outer partitions 42–46 (Phase 3B), 42 (Phases 3C/3D); quota subsampling seed 42.

`experiments_phase3/phase3d_full_cohort/extracted_remaining_10160_checkpoint.csv` (an intermediate
extraction checkpoint) is omitted; the merged table `ptbxl_eligible_cohort_15160_features.csv` is included.

## What is not included

- Raw ECG waveforms (obtain from the original repositories below).
- **CPSC-2018-derived files.** The CPSC analysis used a third-party denoised copy whose licence could not be
  verified (repository licence "other", empty LICENSE file, checked 23 October 2026). CPSC feature/index
  files are therefore omitted, and the five CPSC arrays were removed from the legacy Table 5 OOF archive
  (`table5_oof_predictions_ptbxl_chapman_only.npz`; all other arrays are unchanged). Aggregate CPSC metrics
  remain in the legacy result JSONs.
- TabPFN checkpoints, Gemma-3 base model and LoRA adapters (third-party licences).
- Manuscript files.

## Data sources and licences

| Dataset | Source | Licence |
|---|---|---|
| PTB-XL | Wagner et al. 2020, *Sci Data* 7:154, https://physionet.org/content/ptb-xl/1.0.3/ | CC BY 4.0 |
| Chapman-Shaoxing | Zheng et al. 2020, *Sci Data* 7:48, https://figshare.com/collections/ChapmanECG/4560497 | CC0 |
| CPSC-2018 (not redistributed) | PhysioNet/CinC Challenge 2020 v1.0.2 | CC BY 4.0 (upstream) |

## Licence

- Code: MIT (`LICENSE`).
- Derived data, OOF predictions and figures: CC BY 4.0 (`DATA_LICENSE`). Please also cite the original datasets.

## Citation

See `CITATION.cff`. Contact: Mehmet Kuntalp, Department of Electrical and Electronics Engineering,
Dokuz Eylül University, İzmir, Turkey — mehmet.kuntalp@deu.edu.tr — ORCID 0000-0002-3381-9026.

## AI assistance

Antigravity (Google) and Claude Code (Anthropic) assisted with code, analyses, figures and text; all code was executed
locally and reviewed by the author. Perplexity was used as an LLM-as-a-judge to critically review successive manuscript drafts
(methods, statistics, numerical consistency, references); each critique was checked against the stored outputs in this
repository before any change was accepted. It also assisted with drafting and editing. See the manuscript's AI disclosure statement.
