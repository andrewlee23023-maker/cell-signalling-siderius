"""L6 · calibrated direction transform. FLAGGED; validation; official metric; no uploads.

Family A: s' = s + lambda * (s - s_rev), lambda in 21 values on [0, 1], with dataset-bootstrap bands.
Family B: p = sigmoid((s - s_rev) / tau); score = (1 - w) * rank_ds(s) + w * rank_ds(p), tau x w grid.
Holdout estimate for the best lambda with the part-1 §4 protocol (70 tile splits, 3 or 4 selection tiles).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import FastAuprc, base, boot_ci, data, dataset_index, reverse_index, save, verify  # noqa: E402

verify()
va = data("validation")
s = base("validation")
rev = reverse_index(va)
srev = np.where(rev >= 0, s[np.maximum(rev, 0)], s)
fa = FastAuprc(va)
idx = dataset_index(va)

lams = np.round(np.linspace(0, 1, 21), 3)
curve = []
for lam in lams:
    v = s + lam * (s - srev)
    m, lo, hi = boot_ci(fa.per_dataset(v))
    curve.append({"lambda": float(lam), "lr_auprc": m, "ci95": [lo, hi]})
print("Family A (lambda: lr_auprc [95% CI]):")
for c in curve:
    print(f"  {c['lambda']:.2f}: {c['lr_auprc']:.4f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]")


def rank_ds(v):
    out = np.empty(len(v))
    for ix in idx.values():
        out[ix] = np.argsort(np.argsort(v[ix], kind="mergesort"), kind="mergesort") / max(len(ix) - 1, 1)
    return out


grid = []
rs = rank_ds(s)
for tau in (0.05, 0.1, 0.2, 0.5):
    p = 1.0 / (1.0 + np.exp(-(s - srev) / tau))
    rp = rank_ds(p)
    for w in np.round(np.linspace(0, 1, 11), 2):
        grid.append({"tau": tau, "w": float(w), "lr_auprc": fa((1 - w) * rs + w * rp)})
print("Family B grid (rows tau, columns w = 0.0 .. 1.0):")
for tau in (0.05, 0.1, 0.2, 0.5):
    print(f"  tau {tau:<4}: " + " ".join(f"{g['lr_auprc']:.3f}" for g in grid if g["tau"] == tau))

tiles = sorted({ds.rsplit("_r", 1)[0] for ds in idx})
ds_tile = np.array([ds.rsplit("_r", 1)[0] for ds in idx])
A = np.array([fa.per_dataset(s + lam * (s - srev)) for lam in lams])
sel, hold = [], []
for k in (3, 4):
    for st in combinations(tiles, k):
        m = np.isin(ds_tile, st)
        j = int(np.argmax(np.nanmean(A[:, m], axis=1)))
        sel.append(np.nanmean(A[j, m]))
        hold.append(np.nanmean(A[j, ~m]) - np.nanmean(A[0, ~m]))
best = int(np.argmax([c["lr_auprc"] for c in curve]))
holdout = {"best_lambda_full_validation": float(lams[best]), "best_full_validation": curve[best]["lr_auprc"],
           "mean_holdout_gain_vs_lambda0": float(np.mean(hold)), "sd_holdout_gain": float(np.std(hold))}
print(f"best lambda on all validation: {lams[best]:.2f} ({curve[best]['lr_auprc']:.4f}); tile-split holdout gain of "
      f"the selected lambda vs lambda=0: {np.mean(hold):+.4f} ± {np.std(hold):.4f}")
save("direction_sweep.json", {"family_A": curve, "family_B": grid, "holdout": holdout})
