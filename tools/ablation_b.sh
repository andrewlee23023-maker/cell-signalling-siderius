#!/bin/bash
# Ablation (b): iter1 with the skip connection removed (use_skip_zscore=false), retrained on train, 3 seeds.
set -euo pipefail
cd "$HOME/work/cell-signalling"
rm -rf submissions/iter1-noskip
cp -r submissions/iter1-mlp submissions/iter1-noskip
python3 -I - <<'EOF'
import json
p = "submissions/iter1-noskip/cfg.json"
c = json.load(open(p)); c["use_skip_zscore"] = False
json.dump(c, open(p, "w")); print(c)
EOF
"$HOME/SIDERIUS/.venv/bin/python" tools/model_eval.py submissions/iter1-noskip 20261121 1 2
