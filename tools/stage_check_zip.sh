#!/bin/bash
# Check, reproduce and zip a staged cell-signalling submission.
#   bash tools/stage_check_zip.sh submissions/NAME
set -euo pipefail
cd "$HOME/work/cell-signalling"
PY="$HOME/SIDERIUS/.venv/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SUB="$1"; NAME="$(basename "$SUB")"
"$PY" tools/package.py check "$SUB"
"$PY" tools/repro_check.py "$SUB" --data-dir data --package-dir package/directed_lr_coupling --cs-eval \
  --report "runs/repro-$NAME.json"
"$PY" tools/package.py zip "$SUB" --out "uploads/$NAME-$(date +%H%M%S).zip"
