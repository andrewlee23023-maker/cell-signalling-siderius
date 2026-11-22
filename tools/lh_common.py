"""Shared helpers for the part-2 loophole probes (FLAGGED; validation only; official metric code; no uploads).

Every tool calls verify() first: the official baseline must score 0.5603341514 before any other number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import W, _plugin, load, official_baseline_scores, score  # noqa: E402,F401

BASELINE = 0.5603341514
OUT = W / "runs/loopholes"
OUT.mkdir(parents=True, exist_ok=True)
_cache: dict = {}
_metrics = _plugin("_lr_metrics")
_health = _plugin("_lr_health_views")


def data(split: str) -> dict:
    if split not in _cache:
        _cache[split] = load(split)
    return _cache[split]


def base(split: str) -> np.ndarray:
    key = f"base_{split}"
    if key not in _cache:
        f = OUT / f"{key}.npy"
        if f.is_file():
            _cache[key] = np.load(f)
        else:
            _cache[key] = official_baseline_scores(data(split))
            np.save(f, _cache[key])
    return _cache[key]


def agg(d: dict, s) -> dict:
    return score(d, s)["aggregate"]


def per_ds(d: dict, s) -> dict:
    return {k: v["lr_auprc"] for k, v in score(d, s)["per_dataset"].items()}


def verify() -> float:
    a = agg(data("validation"), base("validation"))["lr_auprc"]
    if abs(a - BASELINE) > 1e-9:
        raise SystemExit(f"baseline check FAILED: {a!r} != {BASELINE}")
    print(f"baseline verified: lr_auprc {a:.10f}")
    return a


def parse(sid: str):
    d, t = sid.split(":", 1)
    tile, r = d.rsplit("_r", 1)
    return d, tile, int(r), tuple(t.split("|"))


def reverse_index(d: dict) -> np.ndarray:
    sids = d["sample_id"].tolist()
    pos = {s: i for i, s in enumerate(sids)}
    rev = np.full(len(sids), -1)
    for i, s in enumerate(sids):
        ds, t = s.split(":", 1)
        l, r, a, b = t.split("|")
        rev[i] = pos.get(f"{ds}:{l}|{r}|{b}|{a}", -1)
    return rev


def dataset_index(d: dict) -> dict:
    out = {}
    for i, ds in enumerate(d["dataset"]):
        out.setdefault(ds, []).append(i)
    return {k: np.array(v) for k, v in sorted(out.items())}


class FastAuprc:
    """lr_auprc with the official average_precision, precomputed dataset indices (for many evaluations)."""

    def __init__(self, d: dict):
        self.idx = dataset_index(d)
        self.y = d["y"]

    def per_dataset(self, s) -> np.ndarray:
        s = np.asarray(s, np.float64)
        out = []
        for ix in self.idx.values():
            y = self.y[ix]
            out.append(_metrics.average_precision(y, s[ix]) if 0 < y.sum() < len(y) else np.nan)
        return np.array(out)

    def __call__(self, s) -> float:
        v = self.per_dataset(s)
        return float(np.nanmean(v))


def health(d: dict, s) -> dict:
    """Official Health views: blocking = >= 5 distinct deciles over the whole stream; recording = dominant <= 0.3."""
    sym = _health.score_deciles(dict(zip(d["sample_id"].tolist(), np.asarray(s, np.float64).tolist())))
    vals, cnt = np.unique(sym, return_counts=True)
    dom = float(cnt.max() / cnt.sum())
    return {"distinct_deciles": int(len(vals)), "blocking_pass": bool(len(vals) >= 5),
            "dominant_fraction": round(dom, 4), "recording_pass": bool(dom <= 0.3)}


def boot_ci(per_dataset_values: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    v = per_dataset_values[np.isfinite(per_dataset_values)]
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(reps, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def save(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    return p
