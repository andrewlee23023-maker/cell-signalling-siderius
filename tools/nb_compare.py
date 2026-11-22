"""Score the NB delta-deviance variants like the M1 candidates (official metric; validation = selection only,
train = stability check; dataset-bootstrap CI vs baseline; top-k recall by beta; Health). Writes runs/nb/compare.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import W, load, official_baseline_scores, score  # noqa: E402
from m1_compare import boot_diff, deciles_ok, topk_recall  # noqa: E402
from nb_delta_dev import CANDIDATES  # noqa: E402

va, tr = load("validation"), load("train")
b_va, b_tr = official_baseline_scores(va), official_baseline_scores(tr)
rows = []
for name, fn in [("baseline (contact Spearman)", None), *CANDIDATES.items()]:
    t0 = time.time()
    sv = b_va if fn is None else np.array([fn(x) for x in va["X"]])
    sec = time.time() - t0
    st = b_tr if fn is None else np.array([fn(x) for x in tr["X"]])
    av, at = score(va, sv)["aggregate"], score(tr, st)["aggregate"]
    rows.append({"candidate": name, "val": av, "train": at,
                 "val_minus_baseline_ci95": (0.0, 0.0, 0.0) if fn is None else boot_diff(va, sv, b_va),
                 "train_minus_baseline_ci95": (0.0, 0.0, 0.0) if fn is None else boot_diff(tr, st, b_tr),
                 "recall_topk_by_beta": topk_recall(va, sv), "health": deciles_ok(va, sv) and bool(np.isfinite(sv).all()),
                 "val_seconds": round(sec, 1)})
out = W / "runs/nb"
out.mkdir(parents=True, exist_ok=True)
(out / "compare.json").write_text(json.dumps(rows, indent=2))
print(f"{'candidate':32} {'val':>7} {'Δval [95% CI]':>23} {'dirAcc':>7} {'train':>7} {'Δtrain [95% CI]':>23}  recall@k β.5/1/2  health  sec")
for r in rows:
    m, lo, hi = r["val_minus_baseline_ci95"]
    tm, tlo, thi = r["train_minus_baseline_ci95"]
    rc = r["recall_topk_by_beta"]
    print(f"{r['candidate']:32} {r['val']['lr_auprc']:>7.4f} {m:>+7.3f} [{lo:+.3f},{hi:+.3f}] {r['val']['direction_accuracy']:>7.3f} "
          f"{r['train']['lr_auprc']:>7.4f} {tm:>+7.3f} [{tlo:+.3f},{thi:+.3f}]  {rc['0.5']:.2f}/{rc['1.0']:.2f}/{rc['2.0']:.2f}   "
          f"{'ok' if r['health'] else 'FAIL'}  {r['val_seconds']}")
