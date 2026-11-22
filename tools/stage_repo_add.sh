#!/bin/bash
# Add the ensemble packages, new evidence and updated report to the existing public repo (no wipe).
set -euo pipefail
W="$HOME/work/cell-signalling"
R="/mnt/c/Users/andre/workspace/cell-signalling-siderius"
mkdir -p "$R/submissions" "$R/submission_zip"
for S in ensemble-noskip-explore2-v2 ensemble-aug-both; do
  rm -rf "$R/submissions/$S"; cp -r "$W/submissions/$S" "$R/submissions/$S"
  find "$R/submissions/$S" \( -name __pycache__ -o -name '*.npz' \) -prune -exec rm -rf {} +
done
cp "$W/uploads/cell-signalling-ensemble-noskip-explore2-v2.zip" "$W/uploads/cell-signalling-ensemble-aug-both.zip" "$R/submission_zip/"
cp "$W"/tools/*.py "$W"/tools/*.sh "$R/tools/"
cp -r "$W/runs/loopholes/." "$R/results/loopholes/"
cp "$W"/runs/repro-ensemble-*.json "$R/results/" 2>/dev/null || true
find "$R/results" \( -name '*.npy' -o -name '*.npz' \) -delete
cp "$W/report/main.tex" "$R/report/"
cp "$W"/report/figures/*.pdf "$W"/report/figures/*.png "$R/report/figures/"
echo "added; large files:"; find "$R" -path "$R/.git" -prune -o -type f -size +5M -printf 'LARGE %s %p\n'
