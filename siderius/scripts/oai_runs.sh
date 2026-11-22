#!/bin/bash
# Wait for iter3 to finish (one SIDERIUS run on the GPU at a time), report it, then run with OpenAI implementing:
#   explore2 — no advice, 1 iteration;  loop2c — team decode advice, 2 iterations.
P="$HOME/work/cell-signalling/project"
SID="$HOME/SIDERIUS-wsl"
OP3=$(grep '"run_name": "iter3"' "$P/results/launches.jsonl" | tail -1 | python3 -I -c 'import json,sys; print(json.loads(sys.stdin.read())["operation_id"])')
ENV3=$(grep '"run_name": "iter3"' "$P/results/launches.jsonl" | tail -1 | python3 -I -c 'import json,sys; print(json.loads(sys.stdin.read())["envelope"])')
echo "########## iter3 ($OP3)"
bash "$P/scripts/wait_run.sh" "$OP3" 30 | grep -v 'state=active'
python3 -I "$P/scripts/run_summary.py" "$ENV3" 2>/dev/null || bash "$P/scripts/why_failed.sh" "$ENV3" | tail -4
echo; echo "########## RUN E: explore2 (OpenAI implement, no advice)"
PROFILE=config/llm-moderate-oai-impl.json bash "$P/scripts/run_single.sh" explore2
echo; echo "########## RUN F: loop2c (OpenAI implement, decode advice, 2 iterations)"
PROFILE=config/llm-moderate-oai-impl.json ITER=2 bash "$P/scripts/run_single.sh" loop2c "$P/advice/decode_findings.md"
