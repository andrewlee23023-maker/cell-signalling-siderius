#!/bin/bash
set -euo pipefail
cd "$HOME/work/cell-signalling"
PY="$HOME/SIDERIUS/.venv/bin/python"
SUB=submissions/ensemble-noskip-explore2-v2
find "$SUB" -name __pycache__ -prune -exec rm -rf {} +
"$PY" tools/package.py check "$SUB"
OUT=uploads/cell-signalling-ensemble-noskip-explore2-v2.zip
rm -f "$OUT"
"$PY" tools/package.py zip "$SUB" --out "$OUT"
cp "$OUT" /mnt/c/Users/andre/Downloads/
sha256sum /mnt/c/Users/andre/Downloads/cell-signalling-ensemble-noskip-explore2-v2.zip
unzip -l "$OUT"
