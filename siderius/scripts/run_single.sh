#!/bin/bash
# One single-iteration SIDERIUS run (Claude Sonnet 5.5; validation on Gemini).
#   bash run_single.sh NAME [ADVICE_FILE]     (no advice file = agents choose freely)
set -uo pipefail
NAME="$1"; ADVICE="${2:-}"
P="$HOME/work/cell-signalling/project"
SID="$HOME/SIDERIUS-wsl"
cd "$SID"
PROFILE="${PROFILE:-config/llm-moderate.json}"
ITER="${ITER:-1}"
DATA="${DATA:-$HOME/work/cell-signalling/data}"
ARGS=(--name "$NAME" --iterations "$ITER" --rounds 2 --data-root "$DATA")
[ -n "$ADVICE" ] && ARGS+=(--advice-file "$ADVICE")
"$SID/.venv/bin/python" "$P/scripts/make_workflow.py" "${ARGS[@]}" || exit 1
"$SID/.venv/bin/python" -m workflows.definition to-json --input "$P/experiment/workflow-$NAME.py" > /dev/null || exit 1
[ -n "$ADVICE" ] && python3 -I "$P/scripts/show_advice.py" "$P/experiment/workflow-$NAME.py" "$ADVICE" | tail -1
OUT=$(bash "$HOME/work/cell-signalling/setup-inputs/with_keys.sh" env PYTHONPATH="$SID/src" "$SID/.venv/bin/python" \
  "$P/scripts/launch.py" --workflow "experiment/workflow-$NAME.py" --profile "$PROFILE" \
  --run-name "$NAME" --estimate-usd "$(awk "BEGIN{print 0.75*$ITER}")" --timeout-s 3600 --data-dir "$DATA") || { echo "$NAME launch refused/failed"; echo "$OUT"; exit 1; }
echo "$OUT" > "$P/results/launch-$NAME.log"
OP=$(echo "$OUT" | python3 -I -c 'import json,sys; t=sys.stdin.read(); print(json.loads(t[t.find("{"):])["operation_id"])')
ENV=$(echo "$OUT" | python3 -I -c 'import json,sys; t=sys.stdin.read(); print(json.loads(t[t.find("{"):])["envelope"])')
echo "$NAME operation $OP envelope $ENV"
bash "$P/scripts/wait_run.sh" "$OP" 30 | grep -v 'state=active'
python3 -I "$P/scripts/run_summary.py" "$ENV" 2>/dev/null || bash "$P/scripts/why_failed.sh" "$ENV" | tail -6
