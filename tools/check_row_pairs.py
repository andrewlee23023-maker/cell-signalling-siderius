"""Fact check: lr_auprc of a contact-cell Spearman correlation for each pair of input rows (validation)."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import load, score  # noqa: E402

va = load("validation")
names = {0: "ligand", 1: "receptor", 2: "library", 3: "sender-nbr frac"}
for i, j in combinations(range(4), 2):
    s = np.zeros(len(va["y"]))
    for k, x in enumerate(va["X"]):
        m = x[4] > 0
        a, b = x[i, m], x[j, m]
        if m.sum() >= 3 and a.std() > 0 and b.std() > 0:
            s[k] = spearmanr(a, b).statistic
    agg = score(va, s)["aggregate"]
    print(f"rows {i}x{j} ({names[i]} x {names[j]}): lr_auprc {agg['lr_auprc']:.3f}, direction_accuracy {agg['direction_accuracy']:.3f}")
