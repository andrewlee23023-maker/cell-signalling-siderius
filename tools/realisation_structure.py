"""L1 · cross-realisation agreement and pooling. FLAGGED; validation; official metric; no uploads.

Steps: (1) tile -> realisation grouping; (1b) label agreement of the same triple across realisations;
(2) score correlation across realisation pairs by label state; (3) pooling probe (mean over realisations
assigned to every realisation); (4) agreement probe over a gamma grid; (5) per-tile breakdown.
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import agg, base, boot_ci, data, parse, per_ds, save, verify  # noqa: E402

verify()
va = data("validation")
s = base("validation")
y = va["y"]
meta = [parse(x) for x in va["sample_id"].tolist()]
res = {}

reals = defaultdict(set)
for d, tile, r, _ in meta:
    reals[tile].add(r)
res["realisations_per_tile"] = {t: sorted(v) for t, v in sorted(reals.items())}
print("realisations per tile:", {t: len(v) for t, v in sorted(reals.items())})

groups = defaultdict(list)
for i, (_, tile, _, trip) in enumerate(meta):
    groups[(tile, trip)].append(i)
sizes = [len(g) for g in groups.values()]
inj_any = [g for g in groups.values() if y[g].any()]
same_label = np.mean([len(set(y[g])) == 1 for g in groups.values() if len(g) > 1])
p_other = np.mean([y[j] for g in inj_any for i in g if y[i] for j in g if j != i])
res["label_agreement"] = {"groups": len(groups), "group_sizes": sorted(set(sizes)),
                          "fraction_groups_same_label_all_realisations": float(same_label),
                          "P(injected in another realisation | injected here)": float(p_other),
                          "overall_prevalence": float(y.mean())}
print(f"label agreement: P(same triple injected in another realisation | injected here) = {p_other:.3f} "
      f"(prevalence {y.mean():.3f}); groups with identical labels across realisations {same_label:.3f}")

pairs = defaultdict(list)
for g in groups.values():
    for i, j in combinations(g, 2):
        key = {2: "both_injected", 1: "one_injected", 0: "neither"}[int(y[i]) + int(y[j])]
        pairs[key].append((s[i], s[j]))
res["score_corr_by_label_state"] = {}
for k, v in pairs.items():
    a = np.array(v)
    r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if len(a) > 2 else float("nan")
    res["score_corr_by_label_state"][k] = {"r": r, "pairs": len(a)}
    print(f"score correlation across realisations, {k}: r = {r:.3f} over {len(a)} pairs")

pooled = s.copy()
for g in groups.values():
    pooled[g] = s[g].mean()
probes = {"baseline": s, "pooled_mean": pooled}
gammas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
for gma in gammas:
    v = s.copy()
    for g in groups.values():
        m, sd = s[g].mean(), s[g].std()
        agree = float(np.clip(1.0 - sd / (abs(m) + 0.05), -1.0, 1.0))
        v[g] = m * (1.0 + gma * agree)
    probes[f"agreement_gamma_{gma}"] = v
res["probes"] = {}
print(f"{'probe':24} {'lr_auprc':>9} {'95% CI':>18} {'dir_acc':>8}")
for name, v in probes.items():
    a = agg(va, v)
    m, lo, hi = boot_ci(np.array(list(per_ds(va, v).values())))
    res["probes"][name] = {**a, "ci95": [lo, hi]}
    print(f"{name:24} {a['lr_auprc']:>9.4f} [{lo:.3f}, {hi:.3f}] {a['direction_accuracy']:>8.3f}")

tile_ap = defaultdict(lambda: defaultdict(list))
for name in ("baseline", "pooled_mean"):
    for ds, ap in per_ds(va, probes[name]).items():
        tile_ap[ds.rsplit("_r", 1)[0]][name].append(ap)
res["per_tile"] = {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in sorted(tile_ap.items())}
print("per tile (baseline -> pooled):", {t: f"{d['baseline']:.3f}->{d['pooled_mean']:.3f}" for t, d in res["per_tile"].items()})
save("realisation.json", res)
