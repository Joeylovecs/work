#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export START_INDEX=0
export END_INDEX=20
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-20_recheck_v3}"
source "$SCRIPT_DIR/_common_50.sh"

run_experiment wtq baseline "wtq/python_baseline"
run_experiment wtq dp "wtq/text_baseline"
run_experiment wtq audit "wtq/optimized_python"
run_experiment tabfact baseline "tabfact/python_baseline"
run_experiment tabfact dp "tabfact/text_baseline"
run_experiment tabfact audit "tabfact/optimized_python"
