"""Official metrics + dataset-bootstrap CI vs the baseline for a SIDERIUS run's prediction deliverable(s).
    python tools/run_vs_baseline.py RUN_ENVELOPE_DIR
Prediction files: the task deliverable prefix lr_coupling_scores*.json ({sample_id: score}).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import base, boot_ci, data, per_ds, verify  # noqa: E402

verify()
va = data("validation")
b = base("validation")
sids = va["sample_id"].tolist()
files = sorted(Path(sys.argv[1]).rglob("lr_coupling_scores*.json"))
print(f"{len(files)} prediction file(s)")
pb = per_ds(va, b)
for f in files:
    pred = json.loads(f.read_text())
    if not isinstance(pred, dict) or not set(sids) <= set(pred):
        print(f"skip {f.name}: not a full {{sample_id: score}} map")
        continue
    s = np.array([float(pred[k]) for k in sids])
    ps = per_ds(va, s)
    d = np.array([ps[k] - pb[k] for k in pb])
    m, lo, hi = boot_ci(np.array(list(ps.values())))
    dm, dlo, dhi = boot_ci(d)
    wins = int((d > 0).sum())
    print(f"{f.relative_to(sys.argv[1])}: lr_auprc {m:.4f} [{lo:.3f}, {hi:.3f}]; vs baseline {dm:+.4f} "
          f"[{dlo:+.3f}, {dhi:+.3f}]; better on {wins}/{len(d)} datasets")
