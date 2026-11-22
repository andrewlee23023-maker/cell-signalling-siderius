#!/bin/bash
# Single-iteration run with the team's round-2 advice (Claude Sonnet 5.5; validation on Gemini).
set -uo pipefail
P="$HOME/work/cell-signalling/project"
SID="$HOME/SIDERIUS-wsl"
cd "$SID"
"$SID/.venv/bin/python" "$P/scripts/make_workflow.py" --name round2 --iterations 1 --rounds 2 \
  --advice-file "$P/advice/round2.md" || exit 1
"$SID/.venv/bin/python" -m workflows.definition to-json --input "$P/experiment/workflow-round2.py" > /dev/null || exit 1
python3 -I "$P/scripts/show_advice.py" "$P/experiment/workflow-round2.py" "$P/advice/round2.md" | tail -1
OUT=$(bash "$HOME/work/cell-signalling/setup-inputs/with_keys.sh" env PYTHONPATH="$SID/src" "$SID/.venv/bin/python" \
  "$P/scripts/launch.py" --workflow experiment/workflow-round2.py --profile config/llm-moderate.json \
  --run-name round2 --estimate-usd 0.75 --timeout-s 3600)
echo "$OUT" | tee "$P/results/launch-round2.log"
OP=$(echo "$OUT" | python3 -I -c 'import json,sys; t=sys.stdin.read(); print(json.loads(t[t.find("{"):])["operation_id"])')
echo "round2 operation $OP"
bash "$P/scripts/wait_run.sh" "$OP" 30
