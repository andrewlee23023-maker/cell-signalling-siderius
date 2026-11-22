#!/bin/bash
# Poll a SIDERIUS operation until it leaves starting/active/queued; print the receipt summary and spend.
#   bash project/scripts/wait_run.sh OPERATION_ID [interval_s]
OP="$1"; INT="${2:-30}"
P="$HOME/work/cell-signalling/project"
PY="$HOME/SIDERIUS-wsl/.venv/bin/python"
cd "$HOME/SIDERIUS-wsl" || exit 1
for i in $(seq 1 400); do
  S=$("$PY" -m tools.candidate_operations status --store-config "$P/results/${STORE_NAME:-store}.json" --operation-id "$OP" 2>&1)
  STATE=$(printf '%s' "$S" | python3 -I -c 'import json,sys
try: print(json.load(sys.stdin).get("state"))
except Exception: print("unparsed")')
  echo "$(date +%T) state=$STATE"
  case "$STATE" in starting|active|queued) sleep "$INT" ;; *) break ;; esac
done
printf '%s' "$S" | python3 -I -c 'import json,sys
d=json.load(sys.stdin); r=d.get("receipt") or {}
print("state:", d.get("state"), "| receipt:", r.get("status"), "| error:", r.get("error"), "| validation_passed:", r.get("validation_passed"))
print("workspace:", r.get("workspace")); print("log:", r.get("log_path"))'
"$HOME/SIDERIUS/.venv/bin/python" "$P/scripts/spend_guard.py" | head -8
