#!/bin/bash
# At 13:00 PST (21:00 UTC): smoke run (Flash-Lite) -> if it completes AND validation passed -> iter1 (Claude + team advice).
set -uo pipefail
P="$HOME/work/cell-signalling/project"
SID="$HOME/SIDERIUS-wsl"
KEYS="$HOME/work/cell-signalling/setup-inputs/with_keys.sh"
TARGET=$(date -u -d "$(date -u +%Y-%m-%d) 21:00:00" +%s)
while [ "$(date -u +%s)" -lt "$TARGET" ]; do sleep 15; done
echo "$(date -u +%H:%M:%S) UTC: 13:00 PST reached"

launch() {  # name workflow profile estimate timeout
  bash "$KEYS" env PYTHONPATH="$SID/src" "$SID/.venv/bin/python" "$P/scripts/launch.py" \
    --workflow "$2" --profile "$3" --run-name "$1" --estimate-usd "$4" --timeout-s "$5" \
    | tee "$P/results/launch-$1.log" | python3 -I -c 'import json,sys
t=sys.stdin.read(); i=t.find("{"); print(json.loads(t[i:])["operation_id"])'
}
outcome() {  # operation id -> "state receipt_status validation_passed"
  cd "$SID" && "$SID/.venv/bin/python" -m tools.candidate_operations status --store-config "$P/results/store.json" --operation-id "$1" \
    | python3 -I -c 'import json,sys; d=json.load(sys.stdin); r=d.get("receipt") or {}; print(d.get("state"), r.get("status"), r.get("validation_passed"))'
}

cd "$SID"
"$SID/.venv/bin/python" "$P/scripts/make_workflow.py" --name smoke --iterations 1 --rounds 1
OP1=$(launch smoke2 experiment/workflow-smoke.py config/llm-smoke.json 0.05 2400) || { echo "smoke launch failed"; exit 1; }
echo "smoke2 operation $OP1"
bash "$P/scripts/wait_run.sh" "$OP1" 20
read -r STATE STATUS VALID <<< "$(outcome "$OP1")"
echo "smoke2 outcome: state=$STATE receipt=$STATUS validation_passed=$VALID"
if [ "$STATUS" != "completed" ] || [ "$VALID" != "True" ]; then
  echo "SMOKE DID NOT PASS — iter1 NOT launched (no Claude spend)."; exit 2
fi

OP2=$(launch iter1 experiment/workflow-iter1.py config/llm-moderate.json 0.75 3600) || { echo "iter1 launch failed"; exit 1; }
echo "iter1 operation $OP2"
bash "$P/scripts/wait_run.sh" "$OP2" 30
read -r STATE STATUS VALID <<< "$(outcome "$OP2")"
echo "iter1 outcome: state=$STATE receipt=$STATUS validation_passed=$VALID"
