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
if [[ -f "$PROJECT_DIR/.secrets/paratera.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.secrets/paratera.env"
  set +a
fi

run_experiment() {
  local dataset="$1"
  local mode="$2"
  local experiment_root="${EXPERIMENT_ROOT:-50}"
  local start="${START_INDEX:-0}"
  local end="${END_INDEX:-50}"
  local experiment="$experiment_root/$3"
  local audit_level="${4:-full}"
  local semantic_repairs=0
  local execution_repairs=0
  local dp_votes=1
  if [[ "$mode" == "audit" || "$mode" == "joint" ]]; then
    semantic_repairs=2
    execution_repairs=1
  fi
  python "$PROJECT_DIR/scripts/run_paper1.py" \
    --dataset "$dataset" \
    --mode "$mode" \
    --experiment "$experiment" \
    --start "$start" \
    --end "$end" \
    --all-questions \
    --audit-level "$audit_level" \
    --audit-mode hybrid \
    --max-repairs "$semantic_repairs" \
    --max-execution-repairs "$execution_repairs" \
    --dp-votes "$dp_votes" \
    --model "${MODEL_ID:-DeepSeek-V3.2}" \
    --temperature 0.0 \
    --timeout "${REQUEST_TIMEOUT:-60}" \
    --overwrite
}
