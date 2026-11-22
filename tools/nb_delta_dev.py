"""MISTy/NCEM-style signed delta-deviance with a negative-binomial GLM, per triple (numpy only; no fitted params).

For one triple, input x [5, 1024]:
  counts  y_i = round(expm1(row1_i) * exp(row2_i) * C), C from the smallest non-zero value (count 1);
          exact on this data (tools/check_count_recovery.py: 100% integer, C shared within each tile).
  offset  o_i = row2_i (log library minus the tile mean; the constant is absorbed by the intercept).
  null    log mu = b0 + o + b2 * row3               (density control: fraction of type-A neighbours)
  full    log mu = b0 + o + b2 * row3 + b1 * e      (exposure e = row0, z-scored and clipped at +-4, or raw)
  NB2     Var = mu + alpha * mu^2; alpha by moments from the null Poisson fit, shared by both fits.
  score   sign(b1) * sqrt(max(D_null - D_full, 0))  (signed likelihood-ratio statistic)
0.0 when there are fewer than 5 contact cells, no non-zero count, or a constant exposure.
"""

from __future__ import annotations

import numpy as np

MIN_CELLS = 5
MAX_ITER = 50
ETA_MAX = 30.0


def recover_counts(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    y = np.zeros(r1.size)
    nz = r1 > 0
    if not nz.any():
        return y
    u = np.expm1(r1[nz]) * np.exp(r2[nz])
    y[nz] = np.round(u / u.min())
    return y


def _nb_deviance(y, mu, alpha):
    mu = np.maximum(mu, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(y > 0, y * np.log(y / mu), 0.0)
    if alpha <= 0:
        return 2.0 * float(np.sum(term1 - (y - mu)))
    inv = 1.0 / alpha
    term2 = (y + inv) * np.log((1.0 + alpha * y) / (1.0 + alpha * mu))
    return 2.0 * float(np.sum(term1 - term2))


def _irls(y, X, offset, alpha):
    """NB2 (or Poisson if alpha == 0) GLM with log link by IRLS with step halving. Returns (beta, mu, dev)."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-8)) - offset.mean()
    eta = np.clip(X @ beta + offset, -ETA_MAX, ETA_MAX)
    mu = np.exp(eta)
    dev = _nb_deviance(y, mu, alpha)
    for _ in range(MAX_ITER):
        w = mu / (1.0 + alpha * mu)
        z = (eta - offset) + (y - mu) / mu
        WX = X * w[:, None]
        try:
            new = np.linalg.solve(X.T @ WX + 1e-8 * np.eye(X.shape[1]), WX.T @ z)
        except np.linalg.LinAlgError:
            break
        step = new - beta
        for _h in range(20):
            cand = beta + step
            eta_c = np.clip(X @ cand + offset, -ETA_MAX, ETA_MAX)
            mu_c = np.exp(eta_c)
            dev_c = _nb_deviance(y, mu_c, alpha)
            if np.isfinite(dev_c) and dev_c <= dev + 1e-10:
                break
            step = step / 2.0
        else:
            break
        converged = abs(dev - dev_c) < 1e-8 * (abs(dev) + 1.0)
        beta, eta, mu, dev = cand, eta_c, mu_c, dev_c
        if converged:
            break
    return beta, mu, dev


def _zclip(v, c=4.0):
    sd = v.std()
    return np.clip((v - v.mean()) / sd, -c, c) if sd > 0 else np.zeros_like(v)


def signed_delta_dev(x, clip: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x[4] > 0
    if m.sum() < MIN_CELLS:
        return 0.0
    lig, r1, off, na = x[0, m], x[1, m], x[2, m], x[3, m]
    y = recover_counts(r1, off)
    if y.sum() == 0 or lig.std() == 0:
        return 0.0
    e = _zclip(lig) if clip else lig
    cols0 = [np.ones(y.size)] + ([na] if na.std() > 0 else [])
    X0 = np.column_stack(cols0)
    X1 = np.column_stack(cols0 + [e])
    _, mu0, _ = _irls(y, X0, off, 0.0)  # Poisson null for the dispersion moment estimate
    alpha = max(0.0, float(np.sum((y - mu0) ** 2 - y) / max(np.sum(mu0 ** 2), 1e-12)))
    _, _, d0 = _irls(y, X0, off, alpha)
    b1, _, d1 = _irls(y, X1, off, alpha)
    s = np.sign(b1[-1]) * np.sqrt(max(d0 - d1, 0.0))
    return float(s) if np.isfinite(s) else 0.0


CANDIDATES = {
    "NB Δdev, exposure z-clipped ±4": lambda x: signed_delta_dev(x, True),
    "NB Δdev, exposure raw": lambda x: signed_delta_dev(x, False),
}
