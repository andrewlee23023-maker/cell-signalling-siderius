#!/bin/bash
# Summarize why a run failed: validation verdicts, errors and the log tail.
#   bash project/scripts/why_failed.sh RUN_ENVELOPE_DIR
E="$1"
L="$E/workspace/candidate_run/worker.log"
echo "== log lines: $(wc -l < "$L")"
echo "== validation / failure lines"
grep -niE 'valid|reject|fail|error|issue|verdict|passed|refus' "$L" | cut -c1-320 | tail -40
echo "== validator outputs"
find "$E/workspace" -iname '*valid*' -name '*.json' | head -5 | while read -r f; do
  echo "-- $f"; python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d, indent=1)[:2500])' "$f"
done
echo "== log tail"; tail -n 25 "$L" | cut -c1-320
