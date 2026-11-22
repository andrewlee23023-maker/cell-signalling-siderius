"""Option 2 (team request): extra TRAIN datasets from the public generator (build_dataset.py, Zenodo 24095146,
MIT, md5-verified). Uses only train.npz (per-tile real counts incl. real LR columns, cell types, positions;
lr_ligand/lr_receptor) — never validation or test tiles.

    python tools/gen_extra.py check            # regenerate t00_00_r0 (public seed) and compare to train.npz
    python tools/gen_extra.py make N SEED      # N new realisations per train tile with SEED -> data_aug/train.npz
                                               # (original 60 train datasets + 10*N new ones named <tile>_x<k>)
"""
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

W = Path.home() / "work/cell-signalling"
spec = importlib.util.spec_from_file_location("build_dataset", W / "downloads/generator/build_dataset.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)
TRAIN = W / "data/train.npz"


def load():
    z = np.load(TRAIN, allow_pickle=False)
    genes = [str(g) for g in z["genes"]]
    lr = pd.DataFrame({"ligand": z["lr_ligand"].astype(str), "receptor": z["lr_receptor"].astype(str)})
    tiles = sorted({str(z[f"{d}/tile"]) for d in z["datasets"]})
    return z, genes, lr, tiles


def tile_arrays(z, t):
    c = sp.csr_matrix((z[f"{t}/base_data"], z[f"{t}/base_indices"], z[f"{t}/base_indptr"]),
                      shape=tuple(int(v) for v in z[f"{t}/base_shape"]))
    return c.toarray().astype(float), z[f"{t}/cell_type"].astype(str), z[f"{t}/xy"].astype(float)


def one(args):
    t, ti, seed, r = args
    z, genes, lr, _ = load()
    cnt, ctp, xy = tile_arrays(z, t)
    sim, ctp2, xy2, inj, dec, E, meta = bd.simulate(cnt, ctp, xy, genes, lr, seed, ti * 100 + r)
    lrg = [str(g) for g in z["lr_genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    key = {(a, b, c, e): bb for a, b, c, e, bb in inj[["ligand", "receptor", "source", "target", "beta"]].itertuples(index=False)}
    lab = np.array([key.get(tuple(x), 0.0) for x in E.itertuples(index=False)], dtype=np.float32)
    return t, r, sim[:, [gi[g] for g in lrg]].astype(np.int32), E, lab, meta


def check():
    z, genes, lr, tiles = load()
    t = tiles[0]
    t0 = time.time()
    _, _, lrc, E, lab, meta = one((t, 0, bd.PUBLIC_SEED, 0))
    ref = z[f"{t}_r0/lr_counts"]
    same_counts = lrc.shape == ref.shape and bool(np.array_equal(lrc, ref))
    same_lab = bool(np.array_equal(lab, z[f"{t}_r0/label_beta"]))
    same_elig = bool(np.array_equal(E["ligand"].values.astype(str), z[f"{t}_r0/elig_ligand"].astype(str)))
    print(f"regenerated {t}_r0 in {time.time() - t0:.1f}s: lr_counts identical {same_counts}, "
          f"labels identical {same_lab}, eligible triples identical {same_elig}; meta {meta}")


def make(n, seed):
    z, genes, lr, tiles = load()
    jobs = [(t, ti, seed, r) for ti, t in enumerate(tiles) for r in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(20, len(jobs))) as ex:
        results = list(ex.map(one, jobs))
    A = {k: z[k] for k in z.files}
    dsets = [str(d) for d in z["datasets"]]
    pos = 0
    for t, r, lrc, E, lab, meta in results:
        d = f"{t}_x{r}"
        dsets.append(d)
        A[f"{d}/tile"] = np.array(t)
        A[f"{d}/lr_counts"] = lrc
        for col in ("ligand", "receptor", "source", "target"):
            A[f"{d}/elig_{col}"] = np.array([str(v) for v in E[col].tolist()], dtype="U32")
        A[f"{d}/label_beta"] = lab
        pos += int((lab > 0).sum())
    A["datasets"] = np.array(dsets, dtype="U32")
    out = W / "data_aug"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "train.npz", **A)
    print(f"generated {len(results)} new train datasets ({pos} positives) in {time.time() - t0:.0f}s; "
          f"wrote {out / 'train.npz'} with {len(dsets)} datasets")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        make(int(sys.argv[2]), int(sys.argv[3]))
