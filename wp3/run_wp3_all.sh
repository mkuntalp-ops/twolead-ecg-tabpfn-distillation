#!/usr/bin/env bash
# Run WP3 for all cohorts: TabPFN teacher + MLP students (GPU/MPS), then XGBoost students in a separate process.
# The XGBoost stage of a cohort runs in the background while the next cohort's MLP stage runs. Finished seeds are skipped,
# so the script can be restarted. Logs: revizyon_v22/results/wp3/<cohort>/{mlp,xgb}.log
set -euo pipefail
ROOT=/Volumes/KIOXIA/Projects/tabpfn-ecg-distillation
PY=/Volumes/KIOXIA/AI/tabpfn_env/bin/python3
W=$ROOT/revizyon_v22/wp3
R=$ROOT/revizyon_v22/results/wp3
cd $ROOT
pids=()
for c in chapman_quota5000 ptbxl_quota5000 ptbxl_natural5000 ptbxl_natural15160 chapman_natural; do
  mkdir -p $R/$c
  $PY -u $W/wp3_teacher_mlp.py --cohort $c >> $R/$c/mlp.log 2>&1
  echo "$c MLP done $(date)"
  $PY -u $W/wp3_xgb.py --cohort $c >> $R/$c/xgb.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "all XGBoost stages done $(date)"
$PY -u $W/wp3_analyze.py > $R/analyze.log 2>&1
echo "analysis done $(date)"
