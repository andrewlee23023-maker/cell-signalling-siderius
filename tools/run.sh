#!/bin/bash
# Run one or more tools with the official SIDERIUS interpreter: bash tools/run.sh a.py b.py ...
cd "$HOME/work/cell-signalling" || exit 1
for t in "$@"; do
  echo "################ $t"
  "$HOME/SIDERIUS/.venv/bin/python" "tools/$t" || echo "!! $t exited $?"
  echo
done
