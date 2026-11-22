"""L2 · selection on a hidden test via the upload quota — SIMULATION on validation only; no uploads.

(1) sigma_diff: SD of paired score differences between comparable-quality variants (the §4 family, top 12 by
    validation lr_auprc), from a 2,000-rep dataset bootstrap; uncertainty = spread over variant pairs.
(2) best-of-k for k = 1..10 with the §4 tile-split protocol (selection tiles stand in for the leaderboard,
    holdout tiles for true performance): optimism mean with 5-95% bands.
(3) analytic E[max of k std normals] * sigma_diff overlay; crossing with |M1 - baseline| = 0.045.
(4) correlated subset: repeat (2) on variants with mutual score correlation > 0.9.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import FastAuprc, OUT, data, save, verify  # noqa: E402
from selection_optimism import family_scores  # noqa: E402

verify()
va = data("validation")
cache = OUT / "family_validation.npz"
if cache.is_file():
    z = np.load(cache, allow_pickle=False)
    names, S = z["names"].tolist(), z["S"]
else:
    names, S = family_scores(va)
    np.savez(cache, names=np.array(names), S=S)
keep, seen = [], set()
for j in range(S.shape[1]):
    key = np.round(S[:, j], 10).tobytes()
    if key not in seen:
        seen.add(key)
        keep.append(j)
names, S = [names[j] for j in keep], S[:, keep]
fa = FastAuprc(va)
A = np.array([fa.per_dataset(S[:, j]) for j in range(S.shape[1])])  # variants x datasets
full = np.nanmean(A, axis=1)
top = np.argsort(-full)[:12]
At = A[top]

rng = np.random.default_rng(0)
nds = At.shape[1]
B = rng.integers(0, nds, size=(2000, nds))
boot_means = np.nanmean(At[:, B], axis=2)  # variants x reps
pair_sd = [float(np.std(boot_means[i] - boot_means[j])) for i, j in combinations(range(len(top)), 2)]
sigma = float(np.mean(pair_sd))
print(f"(1) sigma_diff over {len(pair_sd)} pairs of the top-12 variants: {sigma:.4f} "
      f"(pair spread {np.percentile(pair_sd, 5):.4f}-{np.percentile(pair_sd, 95):.4f}); "
      f"top-12 validation range {full[top].min():.3f}-{full[top].max():.3f}")

ds_names = sorted({d for d in va["dataset"]})
tile_of = np.array([d.rsplit("_r", 1)[0] for d in ds_names])
tiles = sorted(set(tile_of))
splits = [c for k in (3, 4) for c in combinations(tiles, k)]
EMAX = {1: 0.0, 2: 0.5642, 3: 0.8463, 4: 1.0294, 5: 1.1630, 6: 1.2672, 7: 1.3522, 8: 1.4236, 9: 1.4850, 10: 1.5388}


def best_of_k(pool):
    out = {}
    for k in range(1, 11):
        opt = []
        for st in splits:
            m = np.isin(tile_of, st)
            for _ in range(20):
                ch = rng.choice(pool, size=min(k, len(pool)), replace=False)
                j = ch[np.argmax(np.nanmean(A[ch][:, m], axis=1))]
                opt.append(np.nanmean(A[j, m]) - np.nanmean(A[j, ~m]) - (np.nanmean(A[ch][:, m]) - np.nanmean(A[ch][:, ~m])).mean())
        out[k] = {"mean": float(np.mean(opt)), "p05": float(np.percentile(opt, 5)), "p95": float(np.percentile(opt, 95))}
    return out


curve = best_of_k(top)
C = np.corrcoef(S[:, top].T)
corr_pool = [top[i] for i in range(len(top)) if np.sum(C[i] > 0.9) >= 4]
curve_corr = best_of_k(np.array(corr_pool)) if len(corr_pool) >= 4 else None
M1 = 0.045
print(f"{'k':>3} {'simulated optimism [5%,95%]':>32} {'analytic E[max]*sigma':>22}" + ("   correlated subset" if curve_corr else ""))
cross = None
for k in range(1, 11):
    a = EMAX[k] * sigma
    c = curve[k]
    if cross is None and c["mean"] >= M1:
        cross = k
    extra = f"   {curve_corr[k]['mean']:+.4f}" if curve_corr else ""
    print(f"{k:>3} {c['mean']:>+12.4f} [{c['p05']:+.3f},{c['p95']:+.3f}] {a:>+22.4f}{extra}")
print(f"crossing with |M1 - baseline| = {M1}: simulated k = {cross}; analytic k = "
      f"{next((k for k in range(1, 11) if EMAX[k] * sigma >= M1), None)}")
save("selection_hidden.json", {"sigma_diff": sigma, "pair_sd_p05_p95": [float(np.percentile(pair_sd, 5)), float(np.percentile(pair_sd, 95))],
                               "top12": [names[j] for j in top], "best_of_k": curve,
                               "analytic": {k: EMAX[k] * sigma for k in EMAX}, "crossing_k_simulated": cross,
                               "correlated_subset": [names[j] for j in corr_pool], "best_of_k_correlated": curve_corr})
