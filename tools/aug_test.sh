#!/bin/bash
set -euo pipefail
cd "$HOME/work/cell-signalling"
PY="$HOME/SIDERIUS/.venv/bin/python"
"$PY" tools/make_aug_variant.py
grep -n "KEEP_LO\|aug_gen\|masked_fill" submissions/iter1-noskip-aug/train/train.py
"$PY" tools/model_eval.py submissions/iter1-noskip-aug 20261121 1 2
