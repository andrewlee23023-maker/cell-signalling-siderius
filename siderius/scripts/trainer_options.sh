#!/bin/bash
# Which loss types / class weighting / checkpoint policies does the SIDERIUS trainer accept?
cd "$HOME/SIDERIUS-wsl" || exit 1
echo "== loss_type literals"; grep -rhoE 'loss_type[^=]*=[^#]{0,120}' src/agent/schemas src/core 2>/dev/null | grep -i literal | head -5
grep -rhoE 'Literal\[[^]]*"smooth_l1"[^]]*\]' src --include=*.py | sort -u | head -3
echo "== pos_weight / class weighting"; grep -rlE 'pos_weight|class_weight' src --include=*.py | head -5
grep -rhoE '"(bce|bce_logits|bce_with_logits|weighted_bce|binary_cross_entropy)"' src --include=*.py | sort -u | head
echo "== checkpoint_selection literals"; grep -rhoE 'Literal\[[^]]*"best_validation_loss"[^]]*\]' src --include=*.py | sort -u | head -3
