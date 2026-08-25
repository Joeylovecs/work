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
export END_INDEX=150
export EXPERIMENT_ROOT="150_final_v1"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common_50.sh"

seed_cache() {
  local source_dir="$1"
  local target_rel="$2"
  local target_dir="$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$target_rel/cache"
  mkdir -p "$target_dir"
  if [[ -d "$source_dir/cache" ]]; then
    cp -an "$source_dir/cache/." "$target_dir/"
  fi
}

if [[ "$DATASET" == "wtq" ]]; then
  OPT_PY_DEV="$PROJECT_DIR/outputs/100_dev_v4/wtq/optimized_python_count_semantics"
  BLIND_DEV="$PROJECT_DIR/outputs/100_dev_v6/wtq/blind_final_v1"
else
  OPT_PY_DEV="$PROJECT_DIR/outputs/100_dev_v5/tabfact/optimized_python_semantic_v2"
  BLIND_DEV="$PROJECT_DIR/outputs/100_dev_v6/tabfact/blind_final_v1"
fi
OPT_DP_DEV="$PROJECT_DIR/outputs/100_dev_v5/$DATASET/optimized_dp_semantic_v2"

seed_cache "$PROJECT_DIR/outputs/100_dev_v1/$DATASET/python_baseline" "$DATASET/python_baseline"
seed_cache "$PROJECT_DIR/outputs/100_dev_v1/$DATASET/dp_baseline" "$DATASET/dp_baseline"
seed_cache "$OPT_PY_DEV" "$DATASET/optimized_python"
seed_cache "$OPT_DP_DEV" "$DATASET/optimized_dp_raw"

run_experiment "$DATASET" baseline "$DATASET/python_baseline"
run_experiment "$DATASET" dp "$DATASET/dp_baseline"
run_experiment "$DATASET" audit "$DATASET/optimized_python"
MAX_DP_REPAIRS=0 run_experiment "$DATASET" dp_audit "$DATASET/optimized_dp_raw"

python "$PROJECT_DIR/scripts/run_conservative_fusion.py" \
  --dataset "$DATASET" \
  --preferred "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/dp_baseline/result.jsonl" \
  --fallback "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/optimized_dp_raw/result.jsonl" \
  --preferred-label dp_baseline \
  --fallback-label optimized_dp_semantic \
  --experiment "$EXPERIMENT_ROOT/$DATASET/optimized_dp_guarded" \
  --overwrite

seed_cache "$BLIND_DEV" "$DATASET/blind_final_v1"
python "$PROJECT_DIR/scripts/run_double_verifier.py" \
  --dataset "$DATASET" \
  --primary "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/dp_baseline/result.jsonl" \
  --primary-label dp_baseline \
  --candidate "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/optimized_python/result.jsonl" \
  --label optimized_python \
  --hide-candidates \
  --required-candidate-support 1 \
  --confidence-threshold 0.9 \
  --experiment "$EXPERIMENT_ROOT/$DATASET/blind_final_v1" \
  --overwrite

python "$PROJECT_DIR/scripts/run_guarded_joint.py" \
  --dataset "$DATASET" \
  --primary "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/dp_baseline/result.jsonl" \
  --blind "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/blind_final_v1/result.jsonl" \
  --python "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/optimized_python/result.jsonl" \
  --optimized-dp "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/optimized_dp_raw/result.jsonl" \
  --output "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/guarded_joint/result.jsonl"

echo "FINAL_150_COMPLETE dataset=$DATASET"
