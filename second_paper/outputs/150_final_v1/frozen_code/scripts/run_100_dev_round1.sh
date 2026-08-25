#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "wtq" && "$1" != "tabfact" ) ]]; then
  echo "usage: $0 wtq|tabfact" >&2
  exit 2
fi

DATASET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export START_INDEX=0
export END_INDEX=100
export EXPERIMENT_ROOT="100_dev_v1"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"
source "$SCRIPT_DIR/_common_50.sh"

run_experiment "$DATASET" baseline "$DATASET/python_baseline"
run_experiment "$DATASET" dp "$DATASET/dp_baseline"
run_experiment "$DATASET" audit "$DATASET/optimized_python"
MAX_DP_REPAIRS=0 run_experiment "$DATASET" dp_audit "$DATASET/dp_optimized_initial"

python "$PROJECT_DIR/scripts/run_structured_dp.py" \
  --dataset "$DATASET" \
  --experiment "$EXPERIMENT_ROOT/$DATASET/structured_dp" \
  --start "$START_INDEX" \
  --end "$END_INDEX" \
  --temperature 0 \
  --timeout "$REQUEST_TIMEOUT" \
  --overwrite

python "$PROJECT_DIR/scripts/run_consensus_joint.py" \
  --dataset "$DATASET" \
  --dp-baseline "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/dp_baseline/result.jsonl" \
  --dp-optimized "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/dp_optimized_initial/result.jsonl" \
  --dp-structured "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/structured_dp/result.jsonl" \
  --python-source "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/optimized_python/result.jsonl" \
  --experiment "$EXPERIMENT_ROOT/$DATASET/consensus_joint_v1" \
  --overwrite
