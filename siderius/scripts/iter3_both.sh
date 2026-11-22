#!/bin/bash
# Wait for the explore/round3 job to finish (one SIDERIUS run on the GPU at a time), then run
# iter3 (team advice verbatim) and iter3b (point 2 -> focal_cw), one after the other.
P="$HOME/work/cell-signalling/project"
while pgrep -f round_ab.sh > /dev/null; do sleep 15; done
echo "$(date +%T) explore/round3 job finished; starting iter3"
echo "########## RUN C: iter3 (verbatim)"; bash "$P/scripts/run_single.sh" iter3 "$P/advice/iter3.md"
echo; echo "########## RUN D: iter3b (focal_cw)"; bash "$P/scripts/run_single.sh" iter3b "$P/advice/iter3b.md"
