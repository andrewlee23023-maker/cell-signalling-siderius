"""FLAGGED LOOPHOLE PROBES — not a scientific solution. Local, validation split, official metric, no uploads.

Tests the reward-hacking leads not yet closed by decode_report.md:
  #3 mirror-pair suppression: within each {A->B, B->A} pair, push the lower-scoring direction to the bottom.
     Uses the REVERSE sample's input (batch information a per-sample predict(x) never sees).
  #7 direction-blind scorer: score both directions of a pair with the same (symmetric) value.
  #6 identity priors: injection rate of the same (L,R,A,B) / (L,R) / (A,B) in TRAIN (replicates the author's audit).
Writes runs/loopholes/probes.json and prints a table.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import W, load, official_baseline_scores, score  # noqa: E402


def reverse_index(sids: list[str]) -> np.ndarray:
    pos = {s: i for i, s in enumerate(sids)}
    rev = np.full(len(sids), -1)
    for i, s in enumerate(sids):
        d, t = s.split(":", 1)
        l, r, a, b = t.split("|")
        rev[i] = pos.get(f"{d}:{l}|{r}|{b}|{a}", -1)
    return rev


def main() -> int:
    tr, va = load("train"), load("validation")
    sids = va["sample_id"].tolist()
    rev = reverse_index(sids)
    base = official_baseline_scores(va)
    rows = []

    def add(name, s, note):
        a = score(va, s)["aggregate"]
        rows.append({"probe": name, **{k: round(v, 4) for k, v in a.items()}, "note": note})

    add("baseline (contact Spearman)", base, "reference")

    # #3 mirror suppression: the lower direction of each pair drops below every winner.
    span = base.max() - base.min() + 1.0
    mirror = base.copy()
    for i, j in enumerate(rev):
        if j >= 0 and base[i] < base[j]:
            mirror[i] = base[i] - span
    add("#3 mirror suppression", mirror, "uses reverse sample (batch info)")

    # #7 direction-blind: both directions get the pair max (symmetric score).
    sym = np.array([max(base[i], base[j]) if j >= 0 else base[i] for i, j in enumerate(rev)])
    add("#7 direction-blind (pair max)", sym, "ignores direction")

    # #6 identity priors from TRAIN injection frequencies.
    def key_rates(keyfn):
        hits, tot = defaultdict(float), defaultdict(float)
        for s, y in zip(tr["sample_id"].tolist(), tr["y"].tolist()):
            k = keyfn(s.split(":", 1)[1].split("|"))
            hits[k] += y
            tot[k] += 1
        glob = tr["y"].mean()
        return np.array([(hits[k] + glob) / (tot[k] + 1) if (k := keyfn(s.split(":", 1)[1].split("|"))) in tot else glob
                         for s in sids])

    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=1e-9, size=len(sids))  # break exact ties so Health-style ranking is defined
    add("#6 prior (L,R,A,B)", key_rates(lambda t: tuple(t)) + jitter, "train frequency")
    add("#6 prior (L,R)", key_rates(lambda t: (t[0], t[1])) + jitter, "train frequency")
    add("#6 prior (A,B)", key_rates(lambda t: (t[2], t[3])) + jitter, "train frequency")

    out = W / "runs/loopholes"
    out.mkdir(parents=True, exist_ok=True)
    (out / "probes.json").write_text(json.dumps(rows, indent=2))
    print("FLAGGED LOOPHOLE PROBES (validation, official metric; random lr_auprc 0.114 ± 0.014)")
    print(f"{'probe':34} {'lr_auprc':>9} {'dir_acc':>8} {'auroc':>7}  note")
    for r in rows:
        print(f"{r['probe']:34} {r['lr_auprc']:>9.4f} {r['direction_accuracy']:>8.4f} {r['lr_auroc']:>7.4f}  {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
