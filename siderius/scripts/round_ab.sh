#!/bin/bash
# Run A (explore, no advice) then run B (round3, team advice with focal_cw).
P="$HOME/work/cell-signalling/project"
echo "########## RUN A: explore"; bash "$P/scripts/run_single.sh" explore
echo; echo "########## RUN B: round3"; bash "$P/scripts/run_single.sh" round3 "$P/advice/round3.md"
