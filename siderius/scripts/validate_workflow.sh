#!/bin/bash
# Regenerate a workflow and validate it offline (no model calls, no training).
#   bash project/scripts/validate_workflow.sh NAME ITERATIONS ROUNDS [ADVICE_FILE]
set -euo pipefail
P="$HOME/work/cell-signalling/project"
SID="$HOME/SIDERIUS-wsl"
cd "$SID"
ARGS=(--name "$1" --iterations "$2" --rounds "$3")
[ -n "${4:-}" ] && ARGS+=(--advice-file "$4")
"$SID/.venv/bin/python" "$P/scripts/make_workflow.py" "${ARGS[@]}"
"$SID/.venv/bin/python" -m workflows.definition to-json --input "$P/experiment/workflow-$1.py" > "$P/experiment/workflow-$1.json"
grep -o '"allowed_output_types": *\[[^]]*\]' "$P/experiment/workflow-$1.json" || true
echo "workflow-$1 validates offline"
