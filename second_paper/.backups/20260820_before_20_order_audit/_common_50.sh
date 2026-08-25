#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
cd "$PROJECT_DIR"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate work
  fi
fi
export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"

run_experiment() {
  local dataset="$1"
  local mode="$2"
  local experiment_root="${EXPERIMENT_ROOT:-50}"
  local start="${START_INDEX:-0}"
  local end="${END_INDEX:-50}"
  local experiment="$experiment_root/$3"
  local audit_level="${4:-full}"
  python "$PROJECT_DIR/scripts/run_paper1.py" \
    --dataset "$dataset" \
    --mode "$mode" \
    --experiment "$experiment" \
    --start "$start" \
    --end "$end" \
    --audit-level "$audit_level" \
    --audit-mode hybrid \
    --max-repairs 1 \
    --max-execution-repairs 1 \
    --dp-votes 3 \
    --model "${MODEL_ID:-DeepSeek-V3.2}" \
    --temperature 0.1 \
    --timeout "${REQUEST_TIMEOUT:-60}" \
    --overwrite
}
