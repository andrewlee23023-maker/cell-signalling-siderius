"""L4 (missing-triple denominator) and L13 (zero-positive datasets). FLAGGED; validation; official metric code."""
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import _metrics, base, data, dataset_index, save, verify  # noqa: E402

verify()
va = data("validation")
s = base("validation")
truth = dict(zip(va["sample_id"].tolist(), va["beta"].tolist()))
src = inspect.getsource(_metrics.per_dataset)
reads = {
    "positives_from_full_truth": "y = np.array([truth[s] > 0 for s in sids])" in src,
    "missing_filled_below_min": "low = (np.nanmin(v)" in src,
    "ap_called_on_full_vector": "average_precision(y, v)" in src,
}
print("code read (_lr_metrics.per_dataset):", reads)

full = {k: float(v) for k, v in zip(va["sample_id"].tolist(), s)}
results = {"code_read": reads}
for frac in (0.10, 0.25, 0.50):
    partial = {}
    for ds, ix in dataset_index(va).items():
        keep = ix[np.argsort(-s[ix], kind="mergesort")][: int(round(len(ix) * (1 - frac)))]
        partial.update({va["sample_id"][i]: float(s[i]) for i in keep})
    t = _metrics.per_dataset(truth, partial)
    a = _metrics.aggregate(t, "lr_auprc")
    results[f"drop_lowest_{int(frac * 100)}pct"] = a
    print(f"L4 drop lowest {int(frac * 100):>2}% of triples per dataset: lr_auprc {a:.6f} (full {0.5603341514:.6f})")

pos = {ds: int(va["y"][ix].sum()) for ds, ix in dataset_index(va).items()}
n = {ds: len(ix) for ds, ix in dataset_index(va).items()}
results["L13_positives_per_dataset"] = pos
results["L13_zero_or_all_positive_datasets"] = [d for d in pos if pos[d] in (0, n[d])]
print(f"L13 positives per validation dataset: min {min(pos.values())}, max {max(pos.values())}; "
      f"datasets with 0 or all positive: {results['L13_zero_or_all_positive_datasets']}")
print("L13 code read: such a dataset gets lr_auprc = NaN (ok = 0 < y.sum() < len(y)) and is skipped by aggregate().")
results["L13_code"] = "NaN-and-skip" if "ok = 0 < y.sum() < len(y)" in src else "unknown"
save("metric_semantics.json", results)
