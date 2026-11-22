"""Find and load cell-signalling arrays from an unknown data layout.

The release format is unknown before noon, so this module inspects a data
directory and guesses: feature tensors shaped [N, 5, 1024] (or [N, 1024, 5],
transposed on load), 1-D binary label vectors of length N, and metadata tables
(CSV/TSV/Parquet/JSON lines) with N rows. Split membership comes from names
containing train/val/valid/validation/test.

Heuristics, not truth: always read `python tools/data_io.py DATA_DIR` output
and the task card before trusting a guess. Pass explicit keys if it is wrong.

Uses only NumPy, pandas, h5py and torch, all present in the scoring env.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

C, L = 5, 1024
SPLIT_PATTERNS = {
    "train": re.compile(r"(^|[^a-z])train", re.I),
    "val": re.compile(r"(^|[^a-z])(val|valid|validation|dev)([^a-z]|$)", re.I),
    "test": re.compile(r"(^|[^a-z])test", re.I),
}
LABEL_NAMES = re.compile(r"^(y|label|labels|target|targets|is_true|truth|positive|signal|link)$", re.I)
ARRAY_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth"}
TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".jsonl", ".json"}


@dataclass
class Found:
    path: Path
    key: str  # dataset/array key inside the file ("" for a bare .npy)
    shape: tuple
    dtype: str
    kind: str  # "features" | "features_T" | "labels" | "vector" | "table" | "other"
    split: str | None = None
    columns: list[str] = field(default_factory=list)


def split_of(path: Path, key: str = "") -> str | None:
    text = f"{path.as_posix()}/{key}"
    hits = [name for name, pat in SPLIT_PATTERNS.items() if pat.search(text)]
    return hits[-1] if len(hits) == 1 else (hits[-1] if hits else None)


def _classify(shape: tuple, dtype: str) -> str:
    if len(shape) == 3 and shape[1:] == (C, L):
        return "features"
    if len(shape) == 3 and shape[1:] == (L, C):
        return "features_T"
    if len(shape) == 1:
        return "vector"
    if len(shape) == 2 and shape[1] == 1:
        return "vector"
    return "other"


def _iter_arrays(path: Path):
    """Yield (key, lazy_loader, shape, dtype) for every array in a file."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        yield "", (lambda: np.asarray(np.load(path, allow_pickle=False))), arr.shape, str(arr.dtype)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            for k in z.files:
                a = z[k]
                yield k, (lambda k=k: np.load(path, allow_pickle=False)[k]), a.shape, str(a.dtype)
    elif suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            items = []
            f.visititems(lambda n, o: items.append((n, o.shape, str(o.dtype))) if isinstance(o, h5py.Dataset) else None)
        for k, shape, dt in items:
            def load(k=k):
                with h5py.File(path, "r") as g:
                    return np.asarray(g[k][()])
            yield k, load, tuple(shape), dt
    elif suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        flat = {}
        if isinstance(obj, torch.Tensor):
            flat[""] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    flat[str(k)] = v
        for k, v in flat.items():
            yield k, (lambda v=v: v.numpy()), tuple(v.shape), str(v.dtype).replace("torch.", "")


def _read_table(path: Path):
    import pandas as pd

    s = path.suffix.lower()
    if s == ".csv":
        return pd.read_csv(path)
    if s == ".tsv":
        return pd.read_csv(path, sep="\t")
    if s == ".parquet":
        return pd.read_parquet(path)  # needs pyarrow; not in the scoring env
    if s == ".jsonl":
        return pd.read_json(path, lines=True)
    if s == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return pd.DataFrame(data)
    return None


def inventory(data_dir: str | Path) -> list[Found]:
    data_dir = Path(data_dir)
    found: list[Found] = []
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        try:
            if suffix in ARRAY_SUFFIXES:
                for key, _, shape, dtype in _iter_arrays(path):
                    found.append(Found(path, key, tuple(shape), dtype, _classify(tuple(shape), dtype), split_of(path, key)))
            elif suffix in TABLE_SUFFIXES:
                df = _read_table(path)
                if df is not None:
                    found.append(Found(path, "", df.shape, "table", "table", split_of(path), list(map(str, df.columns))))
        except Exception as exc:  # unreadable files are reported, not fatal
            found.append(Found(path, f"<error: {type(exc).__name__}: {exc}>", (), "?", "other", split_of(path)))
    # Promote length-N binary vectors / named vectors to labels.
    feature_ns = {f.shape[0] for f in found if f.kind in {"features", "features_T"}}
    for f in found:
        if f.kind == "vector" and f.shape[0] in feature_ns and (LABEL_NAMES.match(f.key.split("/")[-1] or f.path.stem) or "int" in f.dtype or "bool" in f.dtype or "float" in f.dtype):
            f.kind = "labels" if LABEL_NAMES.match(f.key.split("/")[-1] or f.path.stem) else "vector"
    return found


def _loader_for(f: Found):
    for key, load, _, _ in _iter_arrays(f.path):
        if key == f.key:
            return load
    raise KeyError(f.key)


@dataclass
class Split:
    X: np.ndarray  # float32 [N, 5, 1024]
    y: np.ndarray | None  # float32 [N] or None
    meta: object | None  # pandas DataFrame with N rows, or None
    vectors: dict  # other length-N vectors by name
    sources: dict  # what was loaded from where


def load_split(data_dir: str | Path, split: str = "train", *, x_key: str | None = None,
               y_key: str | None = None) -> Split:
    """Load one split. Raises with a readable message when the guess is ambiguous."""
    items = inventory(data_dir)
    feats = [f for f in items if f.kind in {"features", "features_T"} and (f.split == split or (split == "train" and f.split is None))]
    if x_key:
        feats = [f for f in items if f.kind in {"features", "features_T"} and f.key == x_key and f.split in {split, None}]
    if len(feats) != 1:
        listing = "\n".join(f"  {f.path} [{f.key}] {f.shape} split={f.split}" for f in items if f.kind.startswith("features"))
        raise ValueError(f"expected exactly one feature tensor for split={split!r}, found {len(feats)}:\n{listing}")
    fx = feats[0]
    X = np.asarray(_loader_for(fx)(), dtype=np.float32)
    if fx.kind == "features_T":
        X = np.ascontiguousarray(X.transpose(0, 2, 1))
    n = X.shape[0]
    same_file = [f for f in items if f.path == fx.path and f.kind in {"labels", "vector"} and f.shape[0] == n]
    same_split = [f for f in items if f.split == fx.split and f.kind in {"labels", "vector"} and f.shape[0] == n]
    pool = same_file or same_split
    labels = [f for f in pool if f.kind == "labels"] if not y_key else [f for f in pool if f.key == y_key or f.path.stem == y_key]
    y = None
    if len(labels) == 1:
        y = np.asarray(_loader_for(labels[0])(), dtype=np.float32).reshape(n)
    vectors = {}
    for f in pool:
        if not labels or f is not labels[0]:
            vectors[f.key or f.path.stem] = np.asarray(_loader_for(f)()).reshape(n)
    meta = None
    tables = [f for f in items if f.kind == "table" and f.shape[0] == n and (f.split == fx.split or f.split is None)]
    if len(tables) == 1:
        meta = _read_table(tables[0].path)
        if y is None:
            cols = [c for c in meta.columns if LABEL_NAMES.match(str(c))]
            if len(cols) == 1:
                y = meta[cols[0]].to_numpy(dtype=np.float32)
                labels = [Found(tables[0].path, cols[0], (n,), "table-col", "labels", fx.split)]
    return Split(X, y, meta, vectors, {
        "X": f"{fx.path}[{fx.key}]" + (" (transposed)" if fx.kind == "features_T" else ""),
        "y": f"{labels[0].path}[{labels[0].key}]" if y is not None else None,
        "meta": str(tables[0].path) if meta is not None else None,
    })


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python tools/data_io.py DATA_DIR [split]")
        return 2
    items = inventory(argv[0])
    print(f"{len(items)} item(s) under {argv[0]}")
    for f in items:
        extra = f" cols={f.columns[:12]}{'...' if len(f.columns) > 12 else ''}" if f.columns else ""
        print(f"  [{f.kind:10}] split={str(f.split):5} {f.path.relative_to(argv[0])} [{f.key}] shape={f.shape} dtype={f.dtype}{extra}")
    split = argv[1] if len(argv) > 1 else "train"
    try:
        s = load_split(argv[0], split)
        print(f"\nload_split({split!r}): X {s.X.shape}, y {None if s.y is None else s.y.shape}, "
              f"meta {None if s.meta is None else s.meta.shape}, vectors {list(s.vectors)}")
        print("sources:", s.sources)
    except Exception as exc:
        print(f"\nload_split({split!r}) needs help: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
