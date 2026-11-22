#!/bin/bash
cd "$HOME/work/cell-signalling"
PY="$HOME/SIDERIUS/.venv/bin/python"
export PATH="$(dirname "$PY"):$PATH"
for SUB in ensemble-noskip-explore2-v2 ensemble-aug-both; do
  echo "== $SUB $(date +%H:%M)"
  "$PY" tools/model_eval.py submissions/$SUB 20261121 1 2 2>&1 | grep -vE "epoch [0-9]+/" | tail -8
done
echo "ALL DONE $(date +%H:%M)"
