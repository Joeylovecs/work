#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export START_INDEX=0
export END_INDEX=30
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-30_joint_sweep_v1}"
source "$SCRIPT_DIR/_common_50.sh"

run_joint_pair() {
  local dataset="$1"
  local left="$2"
  local right="$3"
  local label_left="$4"
  local label_right="$5"
  local output_name="$6"
  python "$PROJECT_DIR/scripts/run_joint_from_results.py" \
    --dataset "$dataset" \
    --source-a "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$dataset/$left/result.jsonl" \
    --source-b "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$dataset/$right/result.jsonl" \
    --label-a "$label_left" \
    --label-b "$label_right" \
    --experiment "$EXPERIMENT_ROOT/$dataset/$output_name" \
    --temperature 0.0 \
    --timeout "${REQUEST_TIMEOUT:-60}" \
    --overwrite
}

for dataset in wtq tabfact; do
  run_experiment "$dataset" baseline "$dataset/python_baseline"
  run_experiment "$dataset" dp "$dataset/dp_baseline"
  run_experiment "$dataset" audit "$dataset/optimized_python"
  run_experiment "$dataset" dp_audit "$dataset/optimized_dp"

  run_joint_pair "$dataset" python_baseline dp_baseline \
    python_baseline dp_baseline joint_python_baseline_dp_baseline
  run_joint_pair "$dataset" optimized_python dp_baseline \
    optimized_python dp_baseline joint_optimized_python_dp_baseline
  run_joint_pair "$dataset" python_baseline optimized_dp \
    python_baseline optimized_dp joint_python_baseline_optimized_dp
  run_joint_pair "$dataset" optimized_python optimized_dp \
    optimized_python optimized_dp joint_optimized_python_optimized_dp

  python "$PROJECT_DIR/scripts/run_conservative_fusion.py" \
    --dataset "$dataset" \
    --preferred "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$dataset/dp_baseline/result.jsonl" \
    --fallback "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$dataset/optimized_dp/result.jsonl" \
    --preferred-label dp_baseline \
    --fallback-label optimized_dp_iterative \
    --experiment "$EXPERIMENT_ROOT/$dataset/optimized_dp_guarded" \
    --overwrite

  python "$PROJECT_DIR/scripts/run_conservative_fusion.py" \
    --dataset "$dataset" \
    --preferred "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$dataset/optimized_dp_guarded/result.jsonl" \
    --fallback "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$dataset/joint_optimized_python_optimized_dp/result.jsonl" \
    --preferred-label optimized_dp_guarded \
    --fallback-label joint_optimized_python_optimized_dp_open \
    --experiment "$EXPERIMENT_ROOT/$dataset/joint_safe_optimized_python_optimized_dp" \
    --overwrite

done
