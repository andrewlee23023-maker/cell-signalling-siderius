#!/bin/bash
set -euo pipefail
cd "$HOME/work/cell-signalling"
SUB=submissions/ensemble-noskip-explore2-v2
PY="$HOME/SIDERIUS/.venv/bin/python"
export PATH="$(dirname "$PY"):$PATH"
WD=$(mktemp -d /tmp/ensv2.XXXX)
cp -r "$SUB"/. "$WD"/
cd "$WD"
mkdir -p /tmp/ensv2data && cp "$HOME/work/cell-signalling/data/train.npz" /tmp/ensv2data/
/usr/bin/time -v python train/train.py --data_dir /tmp/ensv2data --seed 20261121 --initial-out initial.pt --out weights.pt 2>&1 | grep -v '^\s*\(Page\|Minor\|Major\|Swaps\|File\|Socket\|Signals\|Voluntary\|Involuntary\|Average\|Exit\)'
cp initial.pt weights.pt "$HOME/work/cell-signalling/$SUB/"
echo "STAGED from $WD"
