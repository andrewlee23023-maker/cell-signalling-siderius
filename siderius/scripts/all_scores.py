#!/usr/bin/env python3
"""Print all SIDERIUS run scores ranked highest to lowest."""
import json, pathlib, sys

results_dir = pathlib.Path("/home/andre/work/cell-signalling/project/results/runs")
rows = []
for f in results_dir.rglob("run_output_*.json"):
    try:
        d = json.loads(f.read_text())
        score = d.get("best_valid_formal_denoising_score") or d.get("best_formal_denoising_score")
        name = f.stem.replace("run_output_", "")
        if score:
            rows.append((float(score), name))
    except Exception:
        pass

rows.sort(reverse=True)
print(f"{'Score':>8}  Run")
print("-" * 40)
for score, name in rows:
    print(f"{score:>8.4f}  {name}")
