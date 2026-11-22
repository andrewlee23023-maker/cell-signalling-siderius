"""L9-L12 tier-3 checks. FLAGGED; validation (and train where stated); official metric; no uploads.

L9  stateful plugin: read baseline/run_baseline.py predict_all -> passes over the evaluation set.
L10 transductive pseudo-labelling inside predict_batch: score with the baseline, top 6.5% of the batch as
    pseudo-positives and bottom 50% as pseudo-negatives, fit logistic regression on per-triple statistics,
    re-score. Batches: whole set, random halves, random quarters (seed 0).
L11 train<->validation tile geometry: fingerprints (cell count, hash of coordinates rounded to 1 um relative
    to the tile origin, histogram of 2,000 sampled pairwise distances) compared across splits.
L12 identity recovery: classifier on train predicting the (A,B) pair from rows 2-3 summaries; accuracy on
    validation; then add the predicted pair's train injection rate to the baseline (logistic fit on train).
"""
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import OUT, W, FastAuprc, base, data, parse, save, verify  # noqa: E402

verify()
va, tr = data("validation"), data("train")
s_va, s_tr = base("validation"), base("train")
fa = FastAuprc(va)
res = {}

# L9
src = (W / "package/directed_lr_coupling/baseline/run_baseline.py").read_text()
one_pass = "for i, p in enumerate(positions)" in src and src.count("predict(") <= 3
res["L9"] = {"package_evaluator_passes": 1 if one_pass else "unclear",
             "note": "predict_all calls predict(x) once per triple in a single dict comprehension; the organizers' "
                     "scoring harness is not in the package, so its pass count cannot be read."}
print(f"L9: package evaluator passes over the evaluation set: {res['L9']['package_evaluator_passes']}")

# L10
z = np.load(OUT / "family_validation.npz", allow_pickle=False)
names, S = z["names"].tolist(), z["S"]
cols = [names.index(c) for c in ("pearson|clip4", "slope_t_cov|clip2", "two_part_t_cov|clip4", "partial_cov|clip2")]
ncont = (va["X"][:, 4] > 0).sum(axis=1)
F = np.column_stack([s_va, S[:, cols], np.log(ncont)])


def transductive(ix):
    s = s_va[ix]
    k = max(1, int(round(0.065 * len(ix))))
    order = np.argsort(-s, kind="mergesort")
    pos, neg = order[:k], order[len(ix) // 2:]
    Xf = StandardScaler().fit_transform(F[ix])
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(np.r_[Xf[pos], Xf[neg]], lab)
    return clf.decision_function(Xf)


rng = np.random.default_rng(0)
perm = rng.permutation(len(s_va))
res["L10"] = {"baseline": fa(s_va)}
for shards in (1, 2, 4):
    v = np.empty(len(s_va))
    for part in np.array_split(perm, shards):
        v[part] = transductive(part)
    res["L10"][f"batch_split_{shards}"] = fa(v)
print("L10 transductive pseudo-labelling lr_auprc:", {k: round(x, 4) for k, x in res["L10"].items()})

# L11
fps = {}
for split in ("train", "validation"):
    with np.load(W / f"data/{split}.npz", allow_pickle=False) as zz:
        tiles = sorted({str(zz[f"{d}/tile"]) for d in zz["datasets"]})
        for t in tiles:
            xy = zz[f"{t}/xy"].astype(np.float64)
            rel = np.round(xy - xy.min(axis=0)).astype(np.int64)
            h = hashlib.sha256(np.ascontiguousarray(rel[np.lexsort(rel.T[::-1])]).tobytes()).hexdigest()[:16]
            pick = np.random.default_rng(0).choice(len(xy), size=min(2000, len(xy)), replace=False)
            p = xy[pick]
            dd = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(p), 1)]
            hist = np.histogram(dd, bins=50, range=(0, 1500))[0] / len(dd)
            fps[(split, t)] = {"n_cells": len(xy), "hash": h, "hist": hist,
                               "origin": xy.min(axis=0).round(1).tolist()}
trs = [k for k in fps if k[0] == "train"]
vas = [k for k in fps if k[0] == "validation"]
hash_matches = [(a[1], b[1]) for a in trs for b in vas if fps[a]["hash"] == fps[b]["hash"]]
nearest = []
for b in vas:
    d = min(np.abs(fps[a]["hist"] - fps[b]["hist"]).sum() for a in trs)
    within = min(np.abs(fps[a]["hist"] - fps[c]["hist"]).sum() for a in trs for c in trs if a != c)
    nearest.append(d)
res["L11"] = {"train_tiles": [t for _, t in trs], "validation_tiles": [t for _, t in vas],
              "exact_coordinate_hash_matches": hash_matches,
              "nearest_hist_L1_val_to_train": [round(x, 4) for x in nearest],
              "nearest_hist_L1_within_train_min": round(float(within), 4)}
print(f"L11: exact coordinate matches train<->validation: {hash_matches or 'none'}; nearest histogram L1 "
      f"distance val->train {min(nearest):.4f}-{max(nearest):.4f} (closest pair within train {within:.4f})")

# L12
def feats(d):
    out = []
    for x in d["X"]:
        m = x[4] > 0
        r2, r3 = x[2, m], x[3, m]
        out.append(np.r_[np.percentile(r2, [10, 25, 50, 75, 90]), np.percentile(r3, [10, 25, 50, 75, 90]),
                         r2.std(), r3.mean(), np.log(m.sum())])
    return np.array(out)


pair = lambda d: np.array([f"{parse(s)[3][2]}>{parse(s)[3][3]}" for s in d["sample_id"]])  # noqa: E731
p_tr, p_va = pair(tr), pair(va)
sc = StandardScaler().fit(feats(tr))
clf = LogisticRegression(max_iter=3000, C=1.0, random_state=0).fit(sc.transform(feats(tr)), p_tr)
pred_va = clf.predict(sc.transform(feats(va)))
acc = float(np.mean(pred_va == p_va))
maj = float(np.mean(p_va == max(set(p_tr), key=list(p_tr).count)))
rate = {c: (tr["y"][p_tr == c].sum() + tr["y"].mean()) / ((p_tr == c).sum() + 1) for c in set(p_tr)}
P = clf.predict_proba(sc.transform(feats(va)))
prior_va = P @ np.array([rate[c] for c in clf.classes_])
P_tr = clf.predict_proba(sc.transform(feats(tr)))
prior_tr = P_tr @ np.array([rate[c] for c in clf.classes_])
comb = LogisticRegression(max_iter=2000, random_state=0).fit(np.column_stack([s_tr, np.log(prior_tr)]), tr["y"])
v = comb.decision_function(np.column_stack([s_va, np.log(prior_va)]))
res["L12"] = {"pair_accuracy_validation": acc, "majority_pair_rate": maj, "n_pairs": len(set(p_tr)),
              "baseline": fa(s_va), "baseline_plus_identity_prior": fa(v), "delta": fa(v) - fa(s_va),
              "combiner_coefficients": comb.coef_.round(4).tolist()}
print(f"L12: (A,B) pair recovered from rows 2-3 with accuracy {acc:.3f} (majority {maj:.3f}, {len(set(p_tr))} pairs); "
      f"baseline {fa(s_va):.4f} -> + identity prior {fa(v):.4f} (Δ {fa(v) - fa(s_va):+.4f})")
save("tier3.json", res)
