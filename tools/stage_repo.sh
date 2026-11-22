#!/bin/bash
# Stage the public GitHub repo: uploaded model + evidence. No data (.npz), no credentials, no caches.
set -euo pipefail
W="$HOME/work/cell-signalling"
R="/mnt/c/Users/andre/workspace/cell-signalling-siderius"
rm -rf "$R"; mkdir -p "$R"
cp -r "$W/submissions/iter1-noskip-gen" "$R/submission"
rm -f "$R/submission/train_augmented.npz"
mkdir -p "$R/submission_zip" && cp "$W/uploads/iter1-noskip-gen-175133.zip" "$R/submission_zip/cell-signalling-iter1-noskip-gen.zip"
mkdir -p "$R/report/figures" && cp "$W/report/main.tex" "$W/report/neurips_2026.sty" "$R/report/" && cp "$W"/report/figures/*.pdf "$W"/report/figures/*.png "$R/report/figures/"
mkdir -p "$R/docs" && cp "$W/data_and_methods.md" "$W/decode_report.md" "$W/decode_report.json" \
  "$W/reward_hacking_explorations.md" "$W/reward_hacking_part2_results.md" "$R/docs/"
mkdir -p "$R/advice" && cp "$W"/project/advice/*.md "$R/advice/"
mkdir -p "$R/siderius/config" "$R/siderius/experiment" "$R/siderius/scripts"
cp "$W"/project/config/*.json "$R/siderius/config/"
cp "$W"/project/experiment/*.py "$R/siderius/experiment/"
cp "$W"/project/scripts/*.py "$W"/project/scripts/*.sh "$R/siderius/scripts/"
cp "$W/project/SIDERIUS-RUN.md" "$R/siderius/"
mkdir -p "$R/tools" && cp "$W"/tools/*.py "$W"/tools/*.sh "$R/tools/"
mkdir -p "$R/results" && cp -r "$W/runs/loopholes" "$W/runs/m1" "$W/runs/nb" "$W/runs/baseline" "$R/results/"
find "$R/results" \( -name '*.npy' -o -name '*.npz' \) -delete
# SIDERIUS run records: logs, records, outputs, proposals/interpretations, generated plugins, token ledgers.
cd "$W/project/results/runs"
for env in */; do
  env="${env%/}"
  [ -d "$env/workspace" ] || continue
  dst="$R/siderius/runs/$env"; mkdir -p "$dst"
  (cd "$env/workspace" && find . -type f \( -name 'worker.log' -o -name 'token_usage.jsonl' -o -name 'run_output_*.json' \
     -o -name 'summary_*.json' -o -name 'proposal_*.json' -o -name 'interpretation_*.json' -o -name 'validation_*.json' \
     -o -path '*/records/*.json' -o -path '*/plugins/*.py' -o -name 'receipt.json' \) -size -3M \
     -not -path '*/.pytest_cache/*' -print0 | xargs -0 -I{} cp --parents {} "$dst/")
done
cp "$W/project/results/launches.jsonl" "$R/siderius/runs/"
echo "staged: $(find "$R" -type f | wc -l) files, $(du -sh "$R" | cut -f1)"
find "$R" -type f -size +5M -printf 'LARGE %s %p\n'
