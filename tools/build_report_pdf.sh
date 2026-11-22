#!/bin/bash
cd "$HOME/work/cell-signalling/report"
D=/mnt/c/Users/andre/AppData/Local/Temp/claude/315ea23220429eff/20a1bfcd-45f8-4067-b707-ee8a5d6b2a60/scratchpad/reportbuild
rm -rf "$D"; mkdir -p "$D"; cp -r main.tex neurips_2026.sty figures "$D"/
echo "$D"
