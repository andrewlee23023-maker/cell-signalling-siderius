"""Training-data augmentation with the task's public generator (build_dataset.py, Zenodo 10.5281/zenodo.24095146,
MIT, copied unchanged next to this file). Uses ONLY train.npz: per train tile the real counts (incl. real LR
columns), cell types and positions, plus the LR pairs. Writes train.npz's 60 datasets plus N new realisations of
each train tile (named <tile>_x<k>, seed = the run seed) to one .npz in the working directory.

Determinism: every random draw in build_dataset.simulate is a function of (seed, tile index, stage, gene index).
Verified locally: with the public seed it regenerates the released t00_00_r0 exactly.
"""

import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build_dataset", HERE / "build_dataset.py")
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)


def _one(args):
    train_npz, t, ti, seed, r = args
    with np.load(train_npz, allow_pickle=False) as z:
        genes = [str(g) for g in z["genes"]]
        lr = pd.DataFrame({"ligand": z["lr_ligand"].astype(str), "receptor": z["lr_receptor"].astype(str)})
        lrg = [str(g) for g in z["lr_genes"]]
        c = sp.csr_matrix((z[f"{t}/base_data"], z[f"{t}/base_indices"], z[f"{t}/base_indptr"]),
                          shape=tuple(int(v) for v in z[f"{t}/base_shape"]))
        cnt, ctp, xy = c.toarray().astype(float), z[f"{t}/cell_type"].astype(str), z[f"{t}/xy"].astype(float)
    sim, _, _, inj, _, E, _ = bd.simulate(cnt, ctp, xy, genes, lr, seed, ti * 100 + r)
    gi = {g: i for i, g in enumerate(genes)}
    key = {(a, b, c_, e): bb for a, b, c_, e, bb in inj[["ligand", "receptor", "source", "target", "beta"]].itertuples(index=False)}
    lab = np.array([key.get(tuple(x), 0.0) for x in E.itertuples(index=False)], dtype=np.float32)
    elig = {col: np.array([str(v) for v in E[col].tolist()], dtype="U32") for col in ("ligand", "receptor", "source", "target")}
    return t, r, sim[:, [gi[g] for g in lrg]].astype(np.int32), elig, lab


def make(train_npz: str, out_npz: str, n: int, seed: int) -> str:
    with np.load(train_npz, allow_pickle=False) as z:
        A = {k: z[k] for k in z.files}
    tiles = sorted({str(A[f"{d}/tile"]) for d in A["datasets"]})
    jobs = [(train_npz, t, ti, seed, r) for ti, t in enumerate(tiles) for r in range(n)]
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, jobs))
    dsets = [str(d) for d in A["datasets"]]
    for t, r, lrc, elig, lab in results:
        d = f"{t}_x{r}"
        dsets.append(d)
        A[f"{d}/tile"] = np.array(t)
        A[f"{d}/lr_counts"] = lrc
        for col, v in elig.items():
            A[f"{d}/elig_{col}"] = v
        A[f"{d}/label_beta"] = lab
    A["datasets"] = np.array(dsets, dtype="U32")
    np.savez_compressed(out_npz, **A)
    return out_npz
