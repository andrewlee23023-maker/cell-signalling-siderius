"""Baseline algorithm (framework "classic"): directional contact detector.

For one triple (L, R, A->B), given the task input x of shape [5, max_contacts]
(row 0 the mean log-normalised ligand over each contact B cell's type-A
neighbours, row 1 the B cell's log-normalised receptor, row 4 the mask), return
the Spearman rank correlation between rows 0 and 1 over the contact cells
(average ranks for ties; 0 if either row is constant or fewer than 3 cells).
A coupling A->B raises R in B cells with the neighbouring A cells' ligand
signal, so injected triples get positive correlations; the reverse direction
and nuisance perturbations (driven by an unobserved ligand) do not. This is the
positive-control detector of our label-audit work (Zenodo 10.5281/zenodo.24023398).
No parameters are fitted on training data; the method is deterministic.
"""

from __future__ import annotations

import numpy as np

NAME = "contact_spearman"


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    r = np.empty(v.size)
    sv = v[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def predict(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x[4] > 0
    if m.sum() < 3:
        return 0.0
    a, b = _ranks(x[0, m]), _ranks(x[1, m])
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
