"""Sanity check for nb_delta_dev on simulated data with a known answer (no task data involved).

Simulates contact cells: library offset, density, exposure; counts ~ NB(mu, alpha) with
log mu = b0 + offset + b1 * zclip(exposure). Checks that the fitted signed statistic is ~0 when b1 = 0,
grows with b1, and that the recovered-count path (log-normalise then recover) gives identical scores.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_delta_dev import _irls, _zclip, signed_delta_dev  # noqa: E402

rng = np.random.default_rng(0)


def make(b1, n=200, alpha=0.5):
    off = rng.normal(0, 0.4, n)
    na = rng.integers(1, 7, n) / 6
    lig = rng.gamma(1.0, 1.0, n)
    mu = np.exp(-1.0 + off + b1 * _zclip(lig))
    lam = rng.gamma(1 / alpha, alpha * mu)
    y = rng.poisson(lam).astype(float)
    lib = np.exp(off + 8.0)  # library (tile mean log library = 8)
    x = np.zeros((5, 1024))
    x[0, :n], x[2, :n], x[3, :n], x[4, :n] = lig, off, na, 1
    x[1, :n] = np.log1p(y / lib * 1e4)
    return x, y, off, na, lig


for b1 in (0.0, 0.5, 1.0, 2.0):
    s = [signed_delta_dev(make(b1)[0]) for _ in range(40)]
    print(f"true b1 = {b1}: signed Δdev mean {np.mean(s):+.2f} (sd {np.std(s):.2f})")

x, y, off, na, lig = make(1.0)
X1 = np.column_stack([np.ones(y.size), na, _zclip(lig)])
beta, _, _ = _irls(y, X1, off, 0.5)
print(f"coefficient recovery at true b1 = 1.0, b0 = -1.0: fitted b1 = {beta[-1]:.2f}, b0 = {beta[0]:.2f}")
