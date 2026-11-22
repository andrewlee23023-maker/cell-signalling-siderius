"""Convert object-dtype arrays in OUR generated data_aug/train.npz to fixed-width unicode, so the file loads with
allow_pickle=False like the released files. (Pickle is only enabled here, on a file this pipeline wrote.)"""
from pathlib import Path

import numpy as np

p = Path.home() / "work/cell-signalling/data_aug/train.npz"
with np.load(p, allow_pickle=True) as z:
    A = {}
    fixed = 0
    for k in z.files:
        a = z[k]
        if a.dtype == object:
            a = np.array([str(v) for v in a.ravel()], dtype="U32").reshape(a.shape)
            fixed += 1
        A[k] = a
np.savez_compressed(p, **A)
with np.load(p, allow_pickle=False) as z:
    bad = [k for k in z.files if z[k].dtype == object]
print(f"converted {fixed} object arrays; object arrays left: {len(bad)}; datasets: {len(A['datasets'])}")
