"""L3 · the leaderboard as a label oracle — SIMULATION on validation only. The live leaderboard is not probed.

The 21 validation datasets stand in for a hidden set; the oracle returns only lr_auprc (official AP code).
Attack 1 (Blum & Hardt boosting): q random Gaussian score vectors; keep those whose oracle answer exceeds
the median answer; final ranking = sum of kept vectors. Attack 2 (hill-climb from the baseline): each query
adds noise to the incumbent and keeps it if the oracle score rises. Both measured by the oracle score of the
final ranking (information extracted from the hidden labels), q = 1..1000, 10 seeds.
Ladder (Blum & Hardt 2015): the oracle reports a new value only if it beats the best reported value by eta
(0.01), rounded to 0.01; attacks then act on reported values.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import FastAuprc, base, data, save, verify  # noqa: E402

verify()
va = data("validation")
fa = FastAuprc(va)
s0 = base("validation")
N = len(s0)
QS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


class Oracle:
    def __init__(self, ladder: bool, eta: float = 0.01):
        self.ladder, self.eta, self.best = ladder, eta, -np.inf

    def __call__(self, v):
        a = fa(v)
        if not self.ladder:
            return a
        if a > self.best + self.eta:
            self.best = round(a, 2)
        return self.best


def boosting(q, seed, ladder):
    rng = np.random.default_rng(seed)
    orc = Oracle(ladder)
    V = rng.normal(size=(q, N))
    ans = np.array([orc(v) for v in V])
    keep = ans > np.median(ans) if q > 1 else ans > 0.114
    agg_v = V[keep].sum(axis=0) if keep.any() else rng.normal(size=N)
    return fa(agg_v)


def hill_climb(q, seed, ladder):
    rng = np.random.default_rng(seed)
    orc = Oracle(ladder)
    inc = s0.copy()
    best = orc(inc)
    for _ in range(q - 1):
        cand = inc + rng.normal(scale=0.05, size=N)
        a = orc(cand)
        if a > best:
            inc, best = cand, a
    return fa(inc)


res = {}
for name, fn in (("boosting_random", boosting), ("hill_climb_from_baseline", hill_climb)):
    for ladder in (False, True):
        key = f"{name}{'_ladder' if ladder else ''}"
        res[key] = {}
        for q in QS:
            vals = [fn(q, seed, ladder) for seed in range(10)]
            res[key][q] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
print("oracle score of the attacker's final ranking (true hidden-set lr_auprc), mean ± sd over 10 seeds")
print(f"{'queries':>8} " + " ".join(f"{k:>26}" for k in res))
for q in QS:
    print(f"{q:>8} " + " ".join(f"{res[k][q]['mean']:>18.4f} ± {res[k][q]['sd']:.3f}" for k in res))
save("oracle.json", {"queries": QS, "results": res, "reference_random": 0.114, "baseline": 0.5603341514})
