"""L5 · global assignment under group exclusivity. FLAGGED; validation; official metric; no uploads.

(1) per-dataset injected count: constant? (2) per dataset select <= 1 triple per group maximising total
(positive) baseline score — groups (L,A,B), (R,A,B), or both jointly (bipartite matching via
linear_sum_assignment) — optionally exactly k_hat = round(0.065 * n) (generator's injection probability;
greedy). (3) selected block ranked above unselected, raw score inside blocks. Reports AP, Health and
per-group accuracy. Saves the best variant's selection mask for L8.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import OUT, agg, base, data, dataset_index, health, parse, save, verify  # noqa: E402

verify()
va = data("validation")
s = base("validation")
y = va["y"]
trip = [parse(x)[3] for x in va["sample_id"].tolist()]
idx = dataset_index(va)
counts = [int(y[ix].sum()) for ix in idx.values()]
ns = [len(ix) for ix in idx.values()]
res = {"injected_per_dataset": counts, "triples_per_dataset": ns,
       "count_constant": len(set(counts)) == 1,
       "count_vs_expected_0.065n": [round(c - 0.065 * n, 2) for c, n in zip(counts, ns)]}
print(f"(1) injected per dataset: {counts} -> constant: {res['count_constant']}")

GL = lambda t: (t[0], t[2], t[3])  # noqa: E731  same ligand, cell types
GR = lambda t: (t[1], t[2], t[3])  # noqa: E731  same receptor, cell types


def select(ix, mode, k_hat=None):
    sel = np.zeros(len(ix), bool)
    if mode in ("L", "R"):
        g = defaultdict(list)
        for p, i in enumerate(ix):
            g[(GL if mode == "L" else GR)(trip[i])].append(p)
        if k_hat is None:
            for members in g.values():
                p = max(members, key=lambda q: s[ix[q]])
                sel[p] = s[ix[p]] > 0
        else:  # greedy is optimal for a partition matroid with a cardinality bound
            used = set()
            for p in np.argsort(-s[ix], kind="mergesort"):
                key = (GL if mode == "L" else GR)(trip[ix[p]])
                if key not in used and sel.sum() < k_hat:
                    sel[p], _ = True, used.add(key)
    else:  # joint: bipartite matching between (L,A,B) groups and (R,A,B) groups
        rows = {k: r for r, k in enumerate(sorted({GL(trip[i]) for i in ix}))}
        cols = {k: c for c, k in enumerate(sorted({GR(trip[i]) for i in ix}))}
        W = np.zeros((len(rows), len(cols)))
        where = {}
        for p, i in enumerate(ix):
            r, c = rows[GL(trip[i])], cols[GR(trip[i])]
            W[r, c] = max(s[i], 0.0)
            where[(r, c)] = p
        rr, cc = linear_sum_assignment(-W)
        for r, c in zip(rr, cc):
            if (r, c) in where and W[r, c] > 0:
                sel[where[(r, c)]] = True
        if k_hat is not None and sel.sum() > k_hat:
            keep = np.argsort(-np.where(sel, s[ix], -np.inf), kind="mergesort")[:k_hat]
            sel[:] = False
            sel[keep] = True
    return sel


def group_accuracy(mask, keyfn):
    hit = tot = 0
    for ix in idx.values():
        g = defaultdict(list)
        for i in ix:
            g[keyfn(trip[i])].append(i)
        for members in g.values():
            pos = [i for i in members if y[i]]
            if len(pos) == 1 and len(members) > 1:
                tot += 1
                hit += int(mask[pos[0]] and mask[members].sum() == 1)
    return hit, tot


res["variants"] = {}
best = None
print(f"{'variant':22} {'lr_auprc':>9} {'dir_acc':>8} {'selected':>9} {'precision':>9}  group acc (L / R)   health")
for mode in ("L", "R", "joint"):
    for use_k in (False, True):
        mask = np.zeros(len(s), bool)
        for ix in idx.values():
            mask[ix] = select(ix, mode, int(round(0.065 * len(ix))) if use_k else None)
        v = s + 10.0 * mask
        a = agg(va, v)
        h = health(va, v)
        gl, gr = group_accuracy(mask, GL), group_accuracy(mask, GR)
        name = f"{mode}{' + k_hat' if use_k else ''}"
        prec = float(y[mask].mean()) if mask.any() else float("nan")
        res["variants"][name] = {**a, "selected": int(mask.sum()), "precision": prec,
                                 "group_acc_L": gl, "group_acc_R": gr, "health": h}
        print(f"{name:22} {a['lr_auprc']:>9.4f} {a['direction_accuracy']:>8.3f} {int(mask.sum()):>9} {prec:>9.3f}  "
              f"{gl[0]}/{gl[1]} , {gr[0]}/{gr[1]}   {'pass' if h['blocking_pass'] else 'REFUSED'}")
        if best is None or a["lr_auprc"] > best[0]:
            best = (a["lr_auprc"], name, mask.copy())
np.save(OUT / "assignment_best_mask.npy", best[2])
res["best_variant"] = best[1]
save("assignment.json", res)
