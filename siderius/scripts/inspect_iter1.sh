#!/bin/bash
R="$HOME/work/cell-signalling/project/results/runs/8575efd3-4afe-4dfc-8745-d200fabb3f78/workspace"
python3 -I "$HOME/work/cell-signalling/project/scripts/inspect_record.py" \
  "$R/iter1/graph_instances/train/iteration_001/attempt_001/records/iter1/lr_coupling_mlp_iter1_002.json" | head -60
echo "==== large files"; find "$R" -type f -size +100k | head
echo "==== python files"; find "$R" -name '*.py' -not -path '*pytest*' | head -20
