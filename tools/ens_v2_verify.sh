#!/bin/bash
cd "$HOME/work/cell-signalling"
PY="$HOME/SIDERIUS/.venv/bin/python"
export PATH="$(dirname "$PY"):$PATH"
"$PY" tools/verify_ensemble_package.py submissions/ensemble-noskip-explore2-v2 2>&1 | tail -3
"$PY" tools/repro_check.py submissions/ensemble-noskip-explore2-v2 --data-dir data --package-dir package/directed_lr_coupling --cs-eval --report runs/repro-ensemble-noskip-explore2-v2.json 2>&1 | tail -15
