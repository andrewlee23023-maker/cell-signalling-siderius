#!/bin/bash
# Status of the most recent launches, plus the organizers' OpenAI research profile.
P="$HOME/work/cell-signalling/project"
cd "$HOME/SIDERIUS-wsl" || exit 1
tail -n 3 "$P/results/launches.jsonl" | python3 -I -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line); print(r["run_name"], r["operation_id"])' | while read -r name op; do
  st=$(.venv/bin/python -m tools.candidate_operations status --store-config "$P/results/store.json" --operation-id "$op" 2>/dev/null \
       | python3 -I -c 'import json,sys; d=json.load(sys.stdin); r=d.get("receipt") or {}; print(d.get("state"), r.get("status"))')
  echo "$name: $st"
done
echo "==== openai-moderate-v1.json"
cat "$HOME/SIDERIUS/configs/llm/openai-moderate-v1.json"
