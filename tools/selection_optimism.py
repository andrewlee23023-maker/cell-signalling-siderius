"""FLAGGED evaluation-process probe: how much does picking the best of many variants on the OPEN validation
split inflate the reported score? (No submission; validation labels are used only to measure this effect.)

1. A family of fixed (unfitted) statistics: association x ligand transform; identical score vectors deduplicated.
2. For each split of the 7 validation tiles into selection tiles (3 or 4) and holdout tiles (the rest):
   pick the best variant by mean per-dataset AP on the selection tiles; record its selection and holdout
   scores. Optimism = selection - holdout. The baseline (fixed in advance) gives the no-selection reference.
3. Repeat choosing among N random variants (N = 1, 5, 10, 25, all) to show growth with search size.
4. Best variant on ALL validation, scored on the 60 train datasets (independent tiles; nothing is fitted).
Writes runs/loopholes/selection_optimism.json.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import W, load, official_baseline_scores, score  # noqa: E402

MIN = 5


def zclip(v, c):
    sd = v.std()
    return np.clip((v - v.mean()) / sd, -c, c) if sd > 0 else np.zeros_like(v)


def rank(v):
    return np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort").astype(float)


TRANSFORMS = {"raw": lambda v: v, "clip2": lambda v: zclip(v, 2), "clip4": lambda v: zclip(v, 4), "rank": rank}


def slope_t(yv, xv, covs):
    n = yv.size
    X = np.column_stack([np.ones(n), xv] + [c for c in covs if c.std() > 0])
    p = X.shape[1]
    if n <= p + 1 or xv.std() == 0 or yv.std() == 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ beta
    s2 = float(res @ res) / (n - p)
    se = np.sqrt(max(s2 * np.linalg.pinv(X.T @ X)[1, 1], 0.0))
    t = beta[1] / se if se > 0 else 0.0
    return float(t) if np.isfinite(t) else 0.0


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def assoc_all(l, r, lib, na):
    """All association statistics for one triple (ligand already transformed)."""
    out = {}
    if l.size < MIN:
        return None
    out["spearman"] = corr(rank(l), rank(r))
    out["pearson"] = corr(l, r)
    for cov_name, covs in (("nocov", ()), ("cov", (lib, na))):
        C = np.column_stack([np.ones(l.size)] + [c for c in covs if c.std() > 0])
        rl = l - C @ np.linalg.lstsq(C, l, rcond=None)[0]
        rr = r - C @ np.linalg.lstsq(C, r, rcond=None)[0]
        out[f"partial_{cov_name}"] = corr(rl, rr)
        out[f"slope_t_{cov_name}"] = slope_t(r, l, covs)
        det = (r > 0).astype(float)
        t_det = slope_t(det, l, covs)
        on = r > 0
        t_int = slope_t(r[on], l[on], tuple(c[on] for c in covs)) if on.sum() >= MIN else 0.0
        p = float(det.mean())
        out[f"detection_t_{cov_name}"] = t_det
        out[f"intensity_t_{cov_name}"] = t_int
        out[f"two_part_t_{cov_name}"] = p * t_int + (1 - p) * t_det
    return out


def family_scores(d):
    names, cols = None, []
    for x in d["X"]:
        m = x[4] > 0
        l0, r, lib, na = (x[k, m].astype(np.float64) for k in range(4))
        row = {}
        for tname, tf in TRANSFORMS.items():
            a = assoc_all(tf(l0), r, lib, na) if l0.size >= MIN else None
            for k in (a or {}):
                row[f"{k}|{tname}"] = a[k]
        if names is None:
            names = sorted(row)
        cols.append([row.get(n, 0.0) for n in names])
    return names, np.array(cols)


def per_dataset_ap(d, s):
    return {k: v["lr_auprc"] for k, v in score(d, s)["per_dataset"].items()}


def main() -> int:
    va, tr = load("validation"), load("train")
    names, S = family_scores(va)
    # dedupe identical score vectors (e.g. Spearman is invariant to monotone ligand transforms)
    keep, seen = [], set()
    for j in range(S.shape[1]):
        key = np.round(S[:, j], 10).tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(j)
    names = [names[j] for j in keep]
    S = S[:, keep]
    ap = [per_dataset_ap(va, S[:, j]) for j in range(S.shape[1])]
    base_ap = per_dataset_ap(va, official_baseline_scores(va))
    datasets = sorted(base_ap)
    tiles = sorted({ds.rsplit("_r", 1)[0] for ds in datasets})
    A = np.array([[ap[j][ds] for ds in datasets] for j in range(len(names))])  # variants x datasets
    b = np.array([base_ap[ds] for ds in datasets])
    tile_of = np.array([ds.rsplit("_r", 1)[0] for ds in datasets])

    rng = np.random.default_rng(0)
    results = {"n_variants": len(names), "tiles": tiles, "by_N": {}}
    splits = [c for k in (3, 4) for c in combinations(tiles, k)]
    for N in (1, 5, 10, 25, len(names)):
        opt, sel_scores, hold_scores, base_diff = [], [], [], []
        for sel_tiles in splits:
            sm = np.isin(tile_of, sel_tiles)
            for rep in range(20 if N < len(names) else 1):
                pool = rng.choice(len(names), size=min(N, len(names)), replace=False)
                j = pool[np.argmax(A[pool][:, sm].mean(axis=1))]
                sel, hold = A[j, sm].mean(), A[j, ~sm].mean()
                opt.append(sel - hold)
                sel_scores.append(sel)
                hold_scores.append(hold)
                base_diff.append(b[sm].mean() - b[~sm].mean())
        results["by_N"][N] = {"optimism_mean": float(np.mean(opt)), "optimism_sd": float(np.std(opt)),
                              "selected_on_sel_tiles": float(np.mean(sel_scores)),
                              "selected_on_holdout": float(np.mean(hold_scores)),
                              "baseline_sel_minus_holdout": float(np.mean(base_diff))}

    full = A.mean(axis=1)
    jbest = int(np.argmax(full))
    best_name = names[jbest]
    # score the best-on-validation variant on train (independent tiles; nothing fitted)
    tnames, ST = family_scores(tr)
    tcol = ST[:, tnames.index(best_name)]
    tr_best = score(tr, tcol)["aggregate"]["lr_auprc"]
    tr_base = score(tr, official_baseline_scores(tr))["aggregate"]["lr_auprc"]
    results["best_on_validation"] = {"variant": best_name, "validation": float(full[jbest]),
                                     "train": tr_best, "baseline_validation": float(b.mean()),
                                     "baseline_train": tr_base}
    results["top5_on_validation"] = [(names[j], float(full[j])) for j in np.argsort(-full)[:5]]
    path = W / "runs/loopholes/selection_optimism.json"
    path.write_text(json.dumps(results, indent=2))

    print(f"{len(names)} distinct fixed variants; {len(splits)} tile splits (3 or 4 selection tiles of 7)")
    print(f"{'N searched':>10} {'score on selection tiles':>25} {'on holdout tiles':>17} {'optimism':>17} {'baseline sel-hold':>18}")
    for N, r in results["by_N"].items():
        print(f"{N:>10} {r['selected_on_sel_tiles']:>25.4f} {r['selected_on_holdout']:>17.4f} "
              f"{r['optimism_mean']:>+9.4f} ± {r['optimism_sd']:.3f} {r['baseline_sel_minus_holdout']:>+18.4f}")
    bb = results["best_on_validation"]
    print(f"\nbest on ALL validation: {bb['variant']}  validation {bb['validation']:.4f} (baseline {bb['baseline_validation']:.4f})"
          f"  ->  train {bb['train']:.4f} (baseline {bb['baseline_train']:.4f})")
    print("top 5 on validation:", ", ".join(f"{n} {v:.3f}" for n, v in results["top5_on_validation"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
