#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:-100}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-100_run}"
# Each wrapper has one fixed dataset/method/interval, matching the first-project shell workflow.
source "$SCRIPT_DIR/_common_50.sh"
run_experiment "tabfact" "baseline" "tabfact/baseline"
