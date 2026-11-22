#!/bin/bash
# Stop ONLY the explore2-gbm model_eval run and its training child (cwd /tmp/seed*-*), not the option-2
# reproduction check (cwd /tmp/repro-cell-signalling-*/work).
for pid in $(pgrep -f '[m]odel_eval.py submissions/explore2-gbm'); do
  echo "stopping model_eval $pid"; kill "$pid"
done
for pid in $(pgrep -f '[t]rain/train.py'); do
  cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
  case "$cwd" in
    /tmp/seed*) echo "stopping explore2 training $pid ($cwd)"; kill "$pid" ;;
    *) echo "keeping $pid ($cwd)" ;;
  esac
done
