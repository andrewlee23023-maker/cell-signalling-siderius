#!/bin/bash
# Pause (SIGSTOP, resumable with SIGCONT) the train/train.py jobs NOT started by the cell-signalling work:
# any train/train.py whose cwd is outside /tmp/repro-cell-signalling-* and /tmp/seed*. Children paused too.
#   bash tools/pause_others.sh          pause
#   bash tools/pause_others.sh resume   resume
SIG=STOP; [ "${1:-}" = "resume" ] && SIG=CONT
for pid in $(pgrep -f '[t]rain/train.py'); do
  cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
  case "$cwd" in
    /tmp/repro-cell-signalling-*|/tmp/seed*) continue ;;
  esac
  kids=$(pgrep -P "$pid" | tr '\n' ' ')
  echo "$SIG pid $pid cwd=$cwd children=[$kids] cmd=$(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-160)"
  kill -"$SIG" "$pid" $kids
done
echo "--- states (T = paused)"
for pid in $(pgrep -f '[t]rain/train.py'); do echo "$pid $(awk '{print $3}' /proc/$pid/stat) $(readlink /proc/$pid/cwd)"; done
