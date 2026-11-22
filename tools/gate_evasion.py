"""L8 · does the Health gate block an ordinally binary ranking? FLAGGED; validation; official Health views.

Take L5's best selection mask as a two-block ranking (1 = selected, 0 = not). Score it as-is (pure binary),
then add uniform jitter U(0, eps) for eps from 1e-17 to 0.1 (seed 0). Record the official Health verdicts,
the AP, and the smallest eps that passes. Jitter breaks ties at random inside each block, so AP can move.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import OUT, FastAuprc, data, health, save, verify  # noqa: E402

verify()
va = data("validation")
mask = np.load(OUT / "assignment_best_mask.npy")
fa = FastAuprc(va)
binary = mask.astype(np.float64)
rows = [{"eps": 0.0, "lr_auprc": fa(binary), **health(va, binary)}]
for e in (1e-17, 1e-16, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1):
    v = binary + np.random.default_rng(0).random(len(binary)) * e
    rows.append({"eps": e, "lr_auprc": fa(v), **health(va, v)})
smallest = next((r["eps"] for r in rows if r["blocking_pass"] and r["eps"] > 0), None)
print(f"{'eps':>8} {'lr_auprc':>9} {'deciles':>8} {'blocking':>9} {'dominant':>9} {'recording':>10}")
for r in rows:
    print(f"{r['eps']:>8.0e} {r['lr_auprc']:>9.4f} {r['distinct_deciles']:>8} {'pass' if r['blocking_pass'] else 'REFUSED':>9} "
          f"{r['dominant_fraction']:>9.3f} {'pass' if r['recording_pass'] else 'flag':>10}")
print(f"smallest jitter that passes the blocking gate: {smallest}")
save("gate.json", {"rows": rows, "smallest_passing_eps": smallest, "mask_source": "assignment_best_mask.npy"})
