"""Score the M1 candidates with the official metric on validation (comparison/selection only, nothing fitted).

Also reports train-split scores (same fixed statistics) as a stability check, per-beta recall in top-k,
the blocking Health condition (>= 5 occupied within-dataset deciles) and a dataset-bootstrap interval
for each candidate's difference from the baseline. Writes runs/m1/compare.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import W, load, official_baseline_scores, score  # noqa: E402
from m1_stats import CANDIDATES  # noqa: E402


def deciles_ok(d, s) -> bool:
    for ds in np.unique(d["dataset"]):
        v = s[d["dataset"] == ds]
        ranks = np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort")
        if len(np.unique((ranks * 10) // len(v))) < 5 or len(np.unique(v)) < 5:
            return False
    return True


def topk_recall(d, s):
    out = {}
    for beta in (0.5, 1.0, 2.0):
        found = tot = 0
        for ds in np.unique(d["dataset"]):
            idx = np.flatnonzero(d["dataset"] == ds)
            k = int(d["y"][idx].sum())
            top = set(idx[np.argsort(-s[idx], kind="mergesort")[:k]])
            for i in idx[np.isclose(d["beta"][idx], beta)]:
                tot += 1
                found += i in top
        out[str(beta)] = found / max(tot, 1)
    return out


def boot_diff(d, s, b, reps=2000, seed=0):
    per_s = score(d, s)["per_dataset"]
    per_b = score(d, b)["per_dataset"]
    keys = [k for k in per_s if np.isfinite(per_s[k]["lr_auprc"])]
    diff = np.array([per_s[k]["lr_auprc"] - per_b[k]["lr_auprc"] for k in keys])
    rng = np.random.default_rng(seed)
    bs = rng.choice(diff, size=(reps, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    va, tr = load("validation"), load("train")
    base_va = official_baseline_scores(va)
    base_tr = official_baseline_scores(tr)
    rows = []
    for name, fn in [("C0 baseline (contact Spearman)", None), *CANDIDATES.items()]:
        t0 = time.time()
        sv = base_va if fn is None else np.array([fn(x) for x in va["X"]])
        sec = time.time() - t0
        st = base_tr if fn is None else np.array([fn(x) for x in tr["X"]])
        av, at = score(va, sv)["aggregate"], score(tr, st)["aggregate"]
        diff = (0.0, 0.0, 0.0) if fn is None else boot_diff(va, sv, base_va)
        rows.append({"candidate": name, "val": av, "train": at, "val_minus_baseline_ci95": diff,
                     "recall_topk_by_beta": topk_recall(va, sv), "health_5_deciles": deciles_ok(va, sv),
                     "finite": bool(np.isfinite(sv).all()), "val_seconds": round(sec, 1)})
    out = W / "runs/m1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(rows, indent=2))
    print("M1 candidates — official metric. val = scored split (selection only); train = stability check")
    print(f"{'candidate':36} {'val AUPRC':>9} {'Δ vs base [95% CI]':>24} {'dirAcc':>7} {'train AUPRC':>11}  recall@k β.5/1/2   health")
    for r in rows:
        m, lo, hi = r["val_minus_baseline_ci95"]
        rc = r["recall_topk_by_beta"]
        print(f"{r['candidate']:36} {r['val']['lr_auprc']:>9.4f} {m:>+8.3f} [{lo:+.3f},{hi:+.3f}] {r['val']['direction_accuracy']:>7.3f} "
              f"{r['train']['lr_auprc']:>11.4f}  {rc['0.5']:.2f}/{rc['1.0']:.2f}/{rc['2.0']:.2f}   {'ok' if r['health_5_deciles'] and r['finite'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
