#!/bin/bash
# Smoke test: 1 iteration, 1 tuning round, all Gemini Flash-Lite. Validates the workflow offline, then launches.
set -euo pipefail
P="$HOME/work/cell-signalling/project"
SID="$HOME/SIDERIUS-wsl"
cd "$SID"
"$SID/.venv/bin/python" "$P/scripts/make_workflow.py" --name smoke --iterations 1 --rounds 1
"$SID/.venv/bin/python" -m workflows.definition to-json --input "$P/experiment/workflow-smoke.py" > /dev/null && echo "workflow validates offline"
bash "$HOME/work/cell-signalling/setup-inputs/with_keys.sh" \
  env PYTHONPATH="$SID/src" "$SID/.venv/bin/python" "$P/scripts/launch.py" \
  --workflow experiment/workflow-smoke.py --profile config/llm-smoke.json --run-name smoke --estimate-usd 0.05 --timeout-s 2400
