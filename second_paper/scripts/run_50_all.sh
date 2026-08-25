#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/wtq_baseline_0_50.sh"
"$SCRIPT_DIR/wtq_audit_0_50.sh"
"$SCRIPT_DIR/wtq_joint_0_50.sh"
"$SCRIPT_DIR/tabfact_baseline_0_50.sh"
"$SCRIPT_DIR/tabfact_audit_0_50.sh"
"$SCRIPT_DIR/tabfact_joint_0_50.sh"
