"""L7 · tie-breaking headroom of the baseline score vector. FLAGGED; validation; official metric; no uploads.

(1) tied groups / fraction of triples in ties per dataset; (2) AP with ties broken best-case, worst-case and
randomly (200 draws, seed 0) vs the official grouped value; (3) ties broken by contact count (a known-worthless
confounder, 0.095 alone) in both directions.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import FastAuprc, base, data, dataset_index, save, verify  # noqa: E402

verify()
va = data("validation")
s = base("validation")
y = va["y"]
fa = FastAuprc(va)
idx = dataset_index(va)
n_contacts = (va["X"][:, 4] > 0).sum(axis=1).astype(float)

tied_frac, tied_groups, tied_pos = [], 0, 0
for ix in idx.values():
    vals, inv, cnt = np.unique(s[ix], return_inverse=True, return_counts=True)
    in_tie = cnt[inv] > 1
    tied_frac.append(in_tie.mean())
    tied_groups += int((cnt > 1).sum())
    tied_pos += int((in_tie & y[ix]).sum())
print(f"(1) tied groups: {tied_groups}; mean fraction of triples in a tie per dataset {np.mean(tied_frac):.4f}; "
      f"positives inside ties: {tied_pos} of {int(y.sum())}")


def ordered_ap(secondary):
    """AP with ties in s broken by `secondary` (higher first); strict ordering, so no grouping is involved."""
    v = s * 1.0
    out = []
    for ix in idx.values():
        order = np.lexsort((-secondary[ix], -v[ix]))
        yy = y[ix][order].astype(float)
        hits = np.cumsum(yy)
        out.append(float(np.sum(hits / np.arange(1, len(yy) + 1) * yy) / yy.sum()))
    return float(np.mean(out))


res = {"tied_groups": tied_groups, "mean_tied_fraction": float(np.mean(tied_frac)), "positives_in_ties": tied_pos,
       "grouped_official": fa(s), "best_case": ordered_ap(y.astype(float)), "worst_case": ordered_ap(-y.astype(float))}
rng = np.random.default_rng(0)
draws = [ordered_ap(rng.random(len(s))) for _ in range(200)]
res["random_mean"], res["random_sd"] = float(np.mean(draws)), float(np.std(draws))
res["contact_count_high_first"] = ordered_ap(n_contacts)
res["contact_count_low_first"] = ordered_ap(-n_contacts)
for k in ("grouped_official", "best_case", "worst_case"):
    print(f"(2) {k:22} {res[k]:.6f}")
print(f"(2) random tie-breaking   {res['random_mean']:.6f} ± {res['random_sd']:.6f} (200 draws)")
print(f"(3) ties broken by contact count, high first {res['contact_count_high_first']:.6f}; low first {res['contact_count_low_first']:.6f}")
save("ties.json", res)
