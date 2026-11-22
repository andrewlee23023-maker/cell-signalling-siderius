#!/bin/bash
# After the current OpenAI-implement queue finishes: one run on the organizers' full OpenAI profile
# (openai-moderate-v1: gpt-6-astra for propose/validate, gpt-5.6-luna elsewhere) with the team's iter3b advice.
P="$HOME/work/cell-signalling/project"
cp -n "$HOME/SIDERIUS/configs/llm/openai-moderate-v1.json" "$P/config/llm-openai-moderate.json"
while pgrep -f oai_runs.sh > /dev/null; do sleep 15; done
echo "########## RUN G: oaifull (organizers' openai-moderate-v1, iter3b advice)"
PROFILE=config/llm-openai-moderate.json bash "$P/scripts/run_single.sh" oaifull "$P/advice/iter3b.md"
