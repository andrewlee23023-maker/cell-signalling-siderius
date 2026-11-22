"""cell-signalling data + official scoring, built on the task package's own code.

    from cs_data import load, score
    tr = load("train")          # dict: X [N,5,1024] f32, y [N] bool, beta [N], dataset [N], sample_id [N]
    va = load("validation")
    m = score(va, scores)       # official per-dataset lr_auprc / direction_accuracy / lr_auroc + aggregates

Tensors come from the task plugin's read_npz + contact_matrix (max_contacts
1024, exactly as task.yaml declares) and are cached in data_cache/. Scores use
the plugin's per_dataset + aggregate, so numbers match the official evaluator.
Requires the SIDERIUS interpreter (the plugin imports execute_tools).

CLI:  python tools/cs_data.py   # build caches and print a summary
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

W = Path.home() / "work/cell-signalling"
PKG = W / "package/directed_lr_coupling"
DATA = W / "data"
CACHE = W / "data_cache"
MAX_CONTACTS = 1024
for p in (str(Path.home() / "SIDERIUS/src"), str(PKG / "task/plugins")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _plugin(name: str):
    spec = importlib.util.spec_from_file_location(name, PKG / "task/plugins" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load(split: str) -> dict:
    """split in {'train', 'validation'}. Returns cached arrays (built on first use)."""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"{split}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    task = _plugin("_lr_task")
    table = task.read_npz(str(DATA / f"{split}.npz"))
    n = len(table.sample_ids)
    X = np.zeros((n, task.CHANNELS, MAX_CONTACTS), np.float32)
    for p in range(n):
        X[p] = task.contact_matrix(table, p, MAX_CONTACTS)
    out = {
        "X": X,
        "beta": np.asarray(table.labels, np.float64),
        "y": np.asarray(table.labels) > 0,
        "dataset": np.array(table.datasets),
        "sample_id": np.array(table.sample_ids),
    }
    np.savez(cache, **out)
    return out


def score(split_data: dict, scores) -> dict:
    """Official metrics for scores aligned with split_data['sample_id']."""
    metrics = _plugin("_lr_metrics")
    truth = dict(zip(split_data["sample_id"].tolist(), split_data["beta"].tolist()))
    preds = dict(zip(split_data["sample_id"].tolist(), np.asarray(scores, np.float64).tolist()))
    table = metrics.per_dataset(truth, preds)
    agg = {k: metrics.aggregate(table, k) for k in ("lr_auprc", "direction_accuracy", "lr_auroc")}
    return {"aggregate": agg, "per_dataset": table}


def official_baseline_scores(split_data: dict) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("baseline_final_model", PKG / "baseline/final_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return np.array([mod.predict(x) for x in split_data["X"]])


if __name__ == "__main__":
    for split in ("train", "validation"):
        d = load(split)
        k = d["X"][:, 4].sum(axis=1)
        print(f"{split}: N={len(d['y'])}, datasets={len(set(d['dataset']))}, positives={int(d['y'].sum())} "
              f"({d['y'].mean():.3f}), contacts min/median/max {int(k.min())}/{int(np.median(k))}/{int(k.max())}, "
              f"full-1024 fraction {np.mean(k == MAX_CONTACTS):.3f}")
    va = load("validation")
    m = score(va, official_baseline_scores(va))["aggregate"]
    print("official baseline via cs_data:", {k: round(v, 10) for k, v in m.items()})
