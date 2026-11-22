"""Train-split tensors for cell-signalling, copied from the task package so training is self-contained.

Adapted (MIT) from the task's `task/plugins/_lr_task.py` (read_npz, contact_matrix), itself adapted from the
SIDERIUS CSV template (v1.1.0, MIT). Only the SIDERIUS framework imports are removed; the computation is
identical: per tile 6-NN graph over the real cell positions; per dataset log-normalised LR genes with the
non-LR library; one [5, max_contacts] matrix per eligible directed triple; target 1 if injected else 0.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

KNN = 6
CHANNELS = 5


def _csr(z, prefix):
    return csr_matrix((z[f"{prefix}_data"], z[f"{prefix}_indices"], z[f"{prefix}_indptr"]),
                      shape=tuple(int(v) for v in z[f"{prefix}_shape"]))


def load_split(path: str, max_contacts: int = 1024):
    """Return X [N, 5, max_contacts] float32, y [N] float32 (1 = injected), sample_ids list."""
    X, Y, ids = [], [], []
    with np.load(path, allow_pickle=False) as z:
        genes = [str(g) for g in z["genes"]]
        lr_genes = [str(g) for g in z["lr_genes"]]
        gi = {g: i for i, g in enumerate(genes)}
        lr_cols = np.array([gi[g] for g in lr_genes])
        lr_index = {g: i for i, g in enumerate(lr_genes)}
        tiles = {}
        for d in (str(v) for v in z["datasets"]):
            t = str(z[f"{d}/tile"])
            if t not in tiles:
                base = _csr(z, f"{t}/base")
                lr_base = np.asarray(base[:, lr_cols].sum(axis=1)).ravel()
                total = np.asarray(base.sum(axis=1)).ravel()
                xy = z[f"{t}/xy"].astype(np.float64)
                _, nb = cKDTree(xy).query(xy, k=KNN + 1)
                tiles[t] = (total - lr_base, z[f"{t}/cell_type"].astype(str), nb[:, 1:].astype(np.int64))
            other, ctp, nb = tiles[t]
            lrc = z[f"{d}/lr_counts"].astype(np.float64)
            lib = np.maximum(other, 1.0)
            loglib = np.log(lib)
            lognorm = np.log1p(lrc / lib[:, None] * 1e4).astype(np.float32)
            loglib_c = (loglib - loglib.mean()).astype(np.float32)
            cols = [z[f"{d}/elig_{c}"].astype(str) for c in ("ligand", "receptor", "source", "target")]
            lab = z[f"{d}/label_beta"].astype(np.float64)
            for i, (l, r, a, b) in enumerate(zip(*cols)):
                sid = f"{d}:{l}|{r}|{a}|{b}"
                is_a = ctp == a
                n_a = is_a[nb].sum(axis=1)
                cells = np.flatnonzero((ctp == b) & (n_a > 0))
                if cells.size > max_contacts:
                    seed = int.from_bytes(hashlib.sha256(sid.encode()).digest()[:8], "little")
                    cells = np.sort(np.random.default_rng(seed).choice(cells, max_contacts, replace=False))
                w = is_a[nb[cells]].astype(np.float32)
                x = np.zeros((CHANNELS, max_contacts), dtype=np.float32)
                k = cells.size
                x[0, :k] = (lognorm[nb[cells], lr_index[l]] * w).sum(axis=1) / w.sum(axis=1)
                x[1, :k] = lognorm[cells, lr_index[r]]
                x[2, :k] = loglib_c[cells]
                x[3, :k] = n_a[cells] / KNN
                x[4, :k] = 1.0
                X.append(x)
                Y.append(float(np.nan_to_num(lab[i]) > 0))
                ids.append(sid)
    return np.stack(X), np.array(Y, np.float32), ids
