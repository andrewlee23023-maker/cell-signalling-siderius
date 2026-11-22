"""Fact check: can integer receptor counts be recovered from the [5,1024] input?

count_i = expm1(row1_i) * exp(row2_i) * C, with C = exp(tile mean log library) / 1e4 unknown but shared by
all cells of a tile. Estimate C per triple from the smallest non-zero value (assumed count 1), then measure
how close all recovered counts are to integers, and whether C agrees across triples of the same tile.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import load  # noqa: E402

va = load("validation")
near, total, cs_by_tile, max_counts, n_nz = 0, 0, defaultdict(list), [], []
for x, ds in zip(va["X"], va["dataset"]):
    m = (x[4] > 0) & (x[1] > 0)
    if m.sum() < 2:
        continue
    u = np.expm1(x[1, m].astype(np.float64)) * np.exp(x[2, m].astype(np.float64))
    c = 1.0 / u.min()
    cnt = u * c
    near += int(np.sum(np.abs(cnt - np.round(cnt)) < 0.05))
    total += cnt.size
    cs_by_tile[ds.rsplit("_r", 1)[0]].append(np.log(c))
    max_counts.append(cnt.max())
    n_nz.append(m.sum())
print(f"recovered counts within 0.05 of an integer: {near / total:.1%} of {total:,} non-zero cells")
spread = [np.ptp(v) for v in cs_by_tile.values()]
mode_frac = []
for v in cs_by_tile.values():
    v = np.round(np.array(v), 3)
    vals, cnts = np.unique(v, return_counts=True)
    mode_frac.append(cnts.max() / len(v))
print(f"log C per tile: most common value shared by {np.mean(mode_frac):.0%} of a tile's triples on average "
      f"(others = triples whose smallest non-zero count is >1)")
print(f"max recovered count per triple: median {np.median(max_counts):.0f}, p90 {np.percentile(max_counts, 90):.0f}; "
      f"non-zero cells per triple: median {int(np.median(n_nz))}")
