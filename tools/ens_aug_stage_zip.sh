#!/bin/bash
set -euo pipefail
cd "$HOME/work/cell-signalling"
SUB=submissions/ensemble-aug-both
PY="$HOME/SIDERIUS/.venv/bin/python"
export PATH="$(dirname "$PY"):$PATH"
WD=$(mktemp -d /tmp/ensaug.XXXX)
cp -r "$SUB"/. "$WD"/
mkdir -p /tmp/ensv2data && cp data/train.npz /tmp/ensv2data/
( cd "$WD" && python train/train.py --data_dir /tmp/ensv2data --seed 20261121 --initial-out initial.pt --out weights.pt 2>&1 | grep --line-buffered -E "epoch (1|12|24|30|60)/|saved|tensors|augmented|Error|Traceback" )
cp "$WD/initial.pt" "$WD/weights.pt" "$SUB/"
find "$SUB" -name __pycache__ -prune -exec rm -rf {} +
"$PY" tools/package.py check "$SUB"
OUT=uploads/cell-signalling-ensemble-aug-both.zip
rm -f "$OUT"
"$PY" tools/package.py zip "$SUB" --out "$OUT" | tail -3
cp "$OUT" /mnt/c/Users/andre/Downloads/
echo "ZIPPED"
