"""FLAGGED LOOPHOLE PROBES, round 2 — not a scientific solution. Local, validation, official metric, no uploads.

E  cross-realisation pooling: the same tile appears in 3 realisations (same cells, positions, non-LR genes);
   injections are redrawn per realisation. Fact check first: are baseline scores of the SAME triple correlated
   across realisations among non-injected triples (i.e. is the spatial null shared)? Then score
   s_i - mean(s_j: same tile & triple, other realisations).
Q  within-receptor calibration: z-score s_i against triples in the same dataset sharing (R, B).
Both use OTHER samples' inputs (batch information that per-sample predict(x) never sees).
Writes runs/loopholes/probes2.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import W, load, official_baseline_scores, score  # noqa: E402


def parse(sid: str):
    d, t = sid.split(":", 1)
    tile, real = d.rsplit("_r", 1)
    return d, tile, int(real), tuple(t.split("|"))


def main() -> int:
    va = load("validation")
    s = official_baseline_scores(va)
    sids = va["sample_id"].tolist()
    y = va["y"]
    meta = [parse(x) for x in sids]
    out = {}

    # E: fact check — correlation of the same triple's score across realisations (negatives only).
    by_tile_triple = defaultdict(list)
    for i, (_, tile, real, trip) in enumerate(meta):
        by_tile_triple[(tile, trip)].append(i)
    pairs = []
    for idx in by_tile_triple.values():
        neg = [i for i in idx if not y[i]]
        for a in range(len(neg)):
            for b in range(a + 1, len(neg)):
                pairs.append((s[neg[a]], s[neg[b]]))
    pa = np.array(pairs)
    r_null = float(np.corrcoef(pa[:, 0], pa[:, 1])[0, 1]) if len(pa) > 2 else float("nan")
    out["null_shared_across_realisations_r"] = r_null
    out["n_negative_pairs"] = len(pa)

    rows = []

    def add(name, v, note):
        a = score(va, v)["aggregate"]
        rows.append({"probe": name, **{k: round(x, 4) for k, x in a.items()}, "note": note})

    add("baseline (contact Spearman)", s, "reference")
    diff = s.copy()
    for i, (_, tile, real, trip) in enumerate(meta):
        others = [j for j in by_tile_triple[(tile, trip)] if j != i]
        if others:
            diff[i] = s[i] - np.mean(s[others])
    add("E realisation-differenced", diff, "uses other realisations (batch info)")

    by_rb = defaultdict(list)
    for i, (d, _, _, (l, r, a, b)) in enumerate(meta):
        by_rb[(d, r, b)].append(i)
    z = s.copy()
    for idx in by_rb.values():
        if len(idx) >= 3:
            v = s[idx]
            sd = v.std()
            z[idx] = (v - v.mean()) / sd if sd > 0 else 0.0
    add("Q within-(R,B) z-score", z, "uses same-dataset triples (batch info)")

    out["probes"] = rows
    path = W / "runs/loopholes/probes2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"E fact check: corr of the same non-injected triple's baseline score across realisations r = {r_null:.3f} "
          f"over {len(pa)} pairs (0 = null redrawn per realisation; high = shared null)")
    print(f"{'probe':34} {'lr_auprc':>9} {'dir_acc':>8} {'auroc':>7}  note")
    for r in rows:
        print(f"{r['probe']:34} {r['lr_auprc']:>9.4f} {r['direction_accuracy']:>8.4f} {r['lr_auroc']:>7.4f}  {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
