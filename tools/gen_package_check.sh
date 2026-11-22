#!/bin/bash
# Option-2 package: the staged weights.pt came from training on data_aug/train.npz (made by tools/gen_extra.py).
# The repro check runs the submission's own train_command (generate + train from train.npz only) and compares.
set -euo pipefail
cd "$HOME/work/cell-signalling"
ls -la submissions/iter1-noskip-gen submissions/iter1-noskip-gen/train
bash tools/stage_check_zip.sh submissions/iter1-noskip-gen
