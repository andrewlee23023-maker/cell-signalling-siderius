#!/bin/bash
# Option 2: iter1-noskip retrained on train + 60 generator realisations of the TRAIN tiles (3 seeds).
set -euo pipefail
cd "$HOME/work/cell-signalling"
rm -rf submissions/iter1-noskip-gen && cp -r submissions/iter1-noskip submissions/iter1-noskip-gen
rm -f submissions/iter1-noskip-gen/weights.pt submissions/iter1-noskip-gen/initial.pt
TRAIN_NPZ="$HOME/work/cell-signalling/data_aug/train.npz" "$HOME/SIDERIUS/.venv/bin/python" tools/model_eval.py submissions/iter1-noskip-gen 20261121 1 2
