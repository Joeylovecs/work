#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 5 ]]; then
  echo "用法: $0 <wtq|tabfact> <baseline|dp|audit|joint> <start> <end> <experiment_root>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export START_INDEX="$3"
export END_INDEX="$4"
export EXPERIMENT_ROOT="$5"
source "$SCRIPT_DIR/_common_50.sh"
run_experiment "$1" "$2" "$1/$2"
