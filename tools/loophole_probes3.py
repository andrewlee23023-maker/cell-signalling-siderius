"""FLAGGED LOOPHOLE PROBES, round 3 — not a scientific solution. Local, validation, official metric, no uploads.

V  sibling competition: triples in the same dataset sharing (R, A, B) with a different ligand.
   Fact checks: how many triples have siblings; are siblings' baseline scores correlated; is the injected
   triple usually the top-scoring sibling? Probe: s_i - max(s_siblings).
W  shared-ligand competition: same (L, A, B), different receptor. Probe: s_i - max(s_siblings).
Both use other triples' inputs from the same dataset (batch information).
Writes runs/loopholes/probes3.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import W, load, official_baseline_scores, score  # noqa: E402


def groups(sids, keyfn):
    g = defaultdict(list)
    for i, s in enumerate(sids):
        d, t = s.split(":", 1)
        g[(d, keyfn(t.split("|")))].append(i)
    return g


def main() -> int:
    va = load("validation")
    s = official_baseline_scores(va)
    y = va["y"]
    sids = va["sample_id"].tolist()
    out, rows = {}, []

    def add(name, v, note):
        a = score(va, v)["aggregate"]
        rows.append({"probe": name, **{k: round(x, 4) for k, x in a.items()}, "note": note})

    add("baseline (contact Spearman)", s, "reference")
    for tag, keyfn, label in (("V", lambda t: (t[1], t[2], t[3]), "same (R,A,B), other ligand"),
                              ("W", lambda t: (t[0], t[2], t[3]), "same (L,A,B), other receptor")):
        g = groups(sids, keyfn)
        multi = [idx for idx in g.values() if len(idx) >= 2]
        n_with = sum(len(idx) for idx in multi)
        pairs = [(s[a], s[b]) for idx in multi for a in idx for b in idx if a < b]
        pa = np.array(pairs) if pairs else np.zeros((0, 2))
        r = float(np.corrcoef(pa[:, 0], pa[:, 1])[0, 1]) if len(pa) > 2 else float("nan")
        top_hits = top_tot = 0
        for idx in multi:
            pos = [i for i in idx if y[i]]
            if len(pos) == 1:
                top_tot += 1
                top_hits += int(max(idx, key=lambda i: s[i]) == pos[0])
        comp = s.copy()
        for idx in multi:
            for i in idx:
                comp[i] = s[i] - max(s[j] for j in idx if j != i)
        out[tag] = {"definition": label, "triples_with_siblings": n_with, "of_total": len(sids),
                    "sibling_score_corr": r, "groups_with_one_positive": top_tot,
                    "positive_is_top_sibling": top_hits}
        add(f"{tag} competition ({label})", comp, "uses same-dataset triples (batch info)")
        print(f"{tag} ({label}): {n_with}/{len(sids)} triples have siblings; sibling score corr r = {r:.3f}; "
              f"in {top_tot} groups with exactly one positive, the positive is the top sibling in {top_hits} "
              f"({top_hits / max(top_tot, 1):.0%}; chance ≈ 1/group size)")
    out["probes"] = rows
    path = W / "runs/loopholes/probes3.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{'probe':48} {'lr_auprc':>9} {'dir_acc':>8} {'auroc':>7}")
    for r in rows:
        print(f"{r['probe']:48} {r['lr_auprc']:>9.4f} {r['direction_accuracy']:>8.4f} {r['lr_auroc']:>7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
