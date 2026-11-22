#!/bin/bash
set -euo pipefail
cd "$HOME/work/cell-signalling"
PY="$HOME/SIDERIUS/.venv/bin/python"
"$PY" tools/make_bce_variant.py
"$PY" tools/model_eval.py submissions/iter1-noskip-bce 20261121 1 2
