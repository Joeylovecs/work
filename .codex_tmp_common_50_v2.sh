#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
cd "$PROJECT_DIR"

CONDA_BASE="/data/amax/home/E23101002/anaconda3"
WORK2_PREFIX="/data/amax/home/E23101002/.conda/envs/work2"

if [[ ! -x "$WORK2_PREFIX/bin/python" ]]; then
  echo "work2 python not found: $WORK2_PREFIX/bin/python" >&2
  exit 4
fi

# Conda's initialization code expects interactive-shell variables such as PS1.
# Temporarily disable nounset while sourcing it, then restore strict mode.
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$WORK2_PREFIX"
set -u

# A stale shell can claim work2 is active while still pointing at base Python.
# Force this project's interpreter to the front even when Conda activation is a no-op.
export PATH="$WORK2_PREFIX/bin:$PATH"
hash -r

# Do not inherit Python paths from the caller (for example, base Python 3.8).
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE_DIR"

if [[ "$(python -c 'import sys; print(sys.prefix)')" != "$WORK2_PREFIX" ]]; then
  echo "failed to activate work2; actual python: $(command -v python)" >&2
  exit 5
fi
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
    --dp-votes "$dp_votes"     --max-dp-repairs "${MAX_DP_REPAIRS:-2}"     --dp-revise-threshold "${DP_REVISE_THRESHOLD:-0.8}" \
    --model "${MODEL_ID:-DeepSeek-V3.2}" \
    --temperature 0.0 \
    --timeout "${REQUEST_TIMEOUT:-60}" \
    --overwrite
}
