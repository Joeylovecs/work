#!/usr/bin/env bash
set -euo pipefail
export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:-100}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-100_run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/wtq_baseline_0_100.sh"
"$SCRIPT_DIR/wtq_audit_0_100.sh"
"$SCRIPT_DIR/wtq_joint_0_100.sh"
"$SCRIPT_DIR/tabfact_baseline_0_100.sh"
"$SCRIPT_DIR/tabfact_audit_0_100.sh"
"$SCRIPT_DIR/tabfact_joint_0_100.sh"
