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
export END_INDEX=5
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-5_nofrozen_v1}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"

DATASET_OUTPUT_DIR="$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET"
if [[ -e "$DATASET_OUTPUT_DIR" ]]; then
  echo "dataset output already exists: $DATASET_OUTPUT_DIR" >&2
  echo "choose a new EXPERIMENT_ROOT to guarantee a fresh run" >&2
  exit 3
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common_50.sh"
export PYTHONUNBUFFERED=1
echo "开始 dataset=$DATASET 区间=[$START_INDEX,$END_INDEX) model=${MODEL_ID:-DeepSeek-V3.2} 输出=$DATASET_OUTPUT_DIR"
trap 'echo "阶段执行失败 exit=$? line=$LINENO；请查看上方错误和阶段 progress.log" >&2' ERR

echo "[阶段 1/7] Python baseline"
run_experiment "$DATASET" baseline "$DATASET/python_baseline"
echo "[阶段 2/7] DP baseline"
run_experiment "$DATASET" dp "$DATASET/dp_baseline"
echo "[阶段 3/7] 优化 Python（审计与有限修复）"
run_experiment "$DATASET" audit "$DATASET/optimized_python"
echo "[阶段 4/7] 优化 DP raw"
MAX_DP_REPAIRS=0 run_experiment "$DATASET" dp_audit "$DATASET/optimized_dp_raw"

echo "[阶段 5/7] DP 保守融合"
python "$PROJECT_DIR/scripts/run_conservative_fusion.py" \
  --dataset "$DATASET" \
  --preferred "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/dp_baseline/result.jsonl" \
  --fallback "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/optimized_dp_raw/result.jsonl" \
  --preferred-label dp_baseline \
  --fallback-label optimized_dp_semantic \
  --experiment "$EXPERIMENT_ROOT/$DATASET/optimized_dp_guarded" \
  --overwrite

echo "[阶段 6/7] 双验证器"
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

echo "[阶段 7/7] 最终 guarded joint"
python "$PROJECT_DIR/scripts/run_guarded_joint.py" \
  --dataset "$DATASET" \
  --primary "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/dp_baseline/result.jsonl" \
  --blind "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/blind_final_v1/result.jsonl" \
  --python "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/optimized_python/result.jsonl" \
  --optimized-dp "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/optimized_dp_raw/result.jsonl" \
  --output "$PROJECT_DIR/outputs/$EXPERIMENT_ROOT/$DATASET/guarded_joint/result.jsonl"

echo "FINAL_150_NOFROZEN_COMPLETE dataset=$DATASET experiment_root=$EXPERIMENT_ROOT"
