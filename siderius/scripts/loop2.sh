#!/bin/bash
# 2-iteration research loop with the team's advice (Claude Sonnet 5.5; validation on Gemini).
set -uo pipefail
P="$HOME/work/cell-signalling/project"
SID="$HOME/SIDERIUS-wsl"
cd "$SID"
"$SID/.venv/bin/python" "$P/scripts/make_workflow.py" --name loop2 --iterations 2 --rounds 2 \
  --advice-file "$P/advice/decode_findings.md"
"$SID/.venv/bin/python" -m workflows.definition to-json --input "$P/experiment/workflow-loop2.py" > /dev/null || exit 1
OUT=$(bash "$HOME/work/cell-signalling/setup-inputs/with_keys.sh" env PYTHONPATH="$SID/src" "$SID/.venv/bin/python" \
  "$P/scripts/launch.py" --workflow experiment/workflow-loop2.py --profile config/llm-moderate.json \
  --run-name "${RUN:-loop2}" --estimate-usd 1.2 --timeout-s 3600)
echo "$OUT" | tee "$P/results/launch-${RUN:-loop2}.log"
OP=$(echo "$OUT" | python3 -I -c 'import json,sys; t=sys.stdin.read(); print(json.loads(t[t.find("{"):])["operation_id"])')
echo "${RUN:-loop2} operation $OP"
bash "$P/scripts/wait_run.sh" "$OP" 30
