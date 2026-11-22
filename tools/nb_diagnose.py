"""Descriptive diagnostic: what drives the NB delta-deviance score among NON-injected triples (validation)?
Spearman correlation of the NB score (and of the baseline score, for comparison) with per-triple quantities.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import load, official_baseline_scores  # noqa: E402
from nb_delta_dev import recover_counts, signed_delta_dev  # noqa: E402

va = load("validation")
neg = ~va["y"]
nb = np.array([signed_delta_dev(x, True) for x in va["X"]])
base = official_baseline_scores(va)
q = {"contact cells": [], "total receptor counts": [], "max receptor count": [], "receptor zero fraction": []}
for x in va["X"]:
    m = x[4] > 0
    y = recover_counts(x[1, m].astype(float), x[2, m].astype(float))
    q["contact cells"].append(m.sum())
    q["total receptor counts"].append(y.sum())
    q["max receptor count"].append(y.max() if y.size else 0)
    q["receptor zero fraction"].append(np.mean(y == 0) if y.size else 1)
print("Spearman with the score, among non-injected triples (validation):")
print(f"{'quantity':26} {'NB Δdev':>9} {'baseline':>9}")
for k, v in q.items():
    v = np.array(v, float)
    print(f"{k:26} {spearmanr(nb[neg], v[neg]).statistic:>+9.3f} {spearmanr(base[neg], v[neg]).statistic:>+9.3f}")
print(f"spread of scores among negatives: NB Δdev sd {nb[neg].std():.2f} (null expectation ~1), baseline sd {base[neg].std():.3f}")
