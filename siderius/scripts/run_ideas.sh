#!/bin/bash
# Runner script for newly drafted SIDERIUS ideas
# Profile: config/llm-moderate-oai-impl.json (Claude Sonnet 5.5 propose, GPT-5.6-luna implement, Gemini validate)
set -euo pipefail

P="$HOME/work/cell-signalling/project"
SID="$HOME/SIDERIUS-wsl"
PROFILE="${PROFILE:-config/llm-moderate-oai-impl.json}"

run_idea() {
    local NAME="$1"
    local ADVICE="$P/advice/${NAME}.md"
    echo "=========================================================="
    echo "Launching SIDERIUS run: $NAME"
    echo "Advice file: $ADVICE"
    echo "=========================================================="
    PROFILE="$PROFILE" bash "$P/scripts/run_single.sh" "$NAME" "$ADVICE"
}

case "${1:-}" in
    threshold_coupling)
        run_idea "threshold_coupling"
        ;;
    pairwise_ranking)
        run_idea "pairwise_ranking"
        ;;
    soft_copula)
        run_idea "soft_copula"
        ;;
    pairwise_gen)
        # Use the generator-augmented data directory (120 datasets)
        DATA="$HOME/work/cell-signalling/data_aug" run_idea "pairwise_gen"
        ;;
    topk_contacts)
        run_idea "topk_contacts"
        ;;
    explore2_focal)
        run_idea "explore2_focal"
        ;;
    explore2_gen)
        DATA="$HOME/work/cell-signalling/data_aug" run_idea "explore2_gen"
        ;;
    all)
        run_idea "threshold_coupling"
        run_idea "pairwise_ranking"
        run_idea "soft_copula"
        ;;
    *)
        echo "Usage: bash run_ideas.sh [threshold_coupling | pairwise_ranking | soft_copula | pairwise_gen | topk_contacts | explore2_focal | explore2_gen | all]"
        exit 1
        ;;
esac
