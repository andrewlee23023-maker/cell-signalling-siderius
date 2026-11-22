#!/bin/bash
# Idea 1: SIDERIUS on generator-augmented TRAIN data (120 datasets, train tiles only; validation unchanged),
#         team iter3b advice, OpenAI implementing.
# Idea 3: explore2's design + team advice (idea3.md), OpenAI implementing, original data.
P="$HOME/work/cell-signalling/project"
W="$HOME/work/cell-signalling"
export STORE_NAME=store2  # own operations store, so this queue runs alongside the main one
D="$W/data_aug"
cp -n "$W/data/validation.npz" "$W/data/manifest.json" "$D/"
python3 -I - "$D" <<'EOF'
import sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1])
for f in ("train.npz", "validation.npz"):
    with np.load(d / f, allow_pickle=False) as z:
        print(f, "datasets:", len(z["datasets"]), "| object arrays:", sum(z[k].dtype == object for k in z.files))
EOF
echo "########## IDEA 1: augdata"
DATA="$D" PROFILE=config/llm-moderate-oai-impl.json bash "$P/scripts/run_single.sh" augdata "$P/advice/iter3b.md"
echo; echo "########## IDEA 3: idea3"
PROFILE=config/llm-moderate-oai-impl.json bash "$P/scripts/run_single.sh" idea3 "$P/advice/idea3.md"
