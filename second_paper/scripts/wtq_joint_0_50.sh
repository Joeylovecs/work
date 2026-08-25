#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Each wrapper has one fixed dataset/method/interval, matching the first-project shell workflow.
source "$SCRIPT_DIR/_common_50.sh"
run_experiment "wtq" "joint" "wtq/joint"
