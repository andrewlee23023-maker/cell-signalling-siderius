"""M1 candidate statistics (team specification, 2026-11-21). Numpy only, per-sample, no fitted parameters.

Team choices: (1) Pearson-type association, (2) partial out rows 2 and 3, (3) treat receptor detection
separately from its level, combined as S = p * t_intensity + (1 - p) * t_detection with p = fraction of
contact cells where R is detected, (4) slope t-statistic (grows with the number of contacts), (5) use the
generator's form: z-score the ligand row within the sample and clip at +-4.

Each function takes x [5, 1024] and returns a finite float. Higher = more likely injected.
"""

from __future__ import annotations

import numpy as np

MIN_CELLS = 5


def _cells(x):
    x = np.asarray(x, dtype=np.float64)
    m = x[4] > 0
    return x[0, m], x[1, m], x[2, m], x[3, m]


def _zclip(v, clip=4.0):
    sd = v.std()
    return np.clip((v - v.mean()) / sd, -clip, clip) if sd > 0 else np.zeros_like(v)


def _slope_t(yv, xv, covs) -> float:
    """t-statistic of xv's coefficient in OLS yv ~ 1 + xv + covs (constant covariates dropped)."""
    n = yv.size
    cols = [np.ones(n), xv] + [c for c in covs if c.std() > 0]
    X = np.column_stack(cols)
    p = X.shape[1]
    if n <= p + 1 or xv.std() == 0 or yv.std() == 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = n - p
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(max(s2 * xtx_inv[1, 1], 0.0))
    t = beta[1] / se if se > 0 else 0.0
    return float(t) if np.isfinite(t) else 0.0


def pearson(x, clip=True) -> float:
    l, r, _, _ = _cells(x)
    if l.size < 3 or l.std() == 0 or r.std() == 0:
        return 0.0
    lv = _zclip(l) if clip else l
    return float(np.corrcoef(lv, r)[0, 1])


def partial_pearson(x, clip=True) -> float:
    l, r, lib, na = _cells(x)
    if l.size < MIN_CELLS:
        return 0.0
    lv = _zclip(l) if clip else l
    C = np.column_stack([np.ones(l.size)] + [c for c in (lib, na) if c.std() > 0])
    rl = lv - C @ np.linalg.lstsq(C, lv, rcond=None)[0]
    rr = r - C @ np.linalg.lstsq(C, r, rcond=None)[0]
    if rl.std() == 0 or rr.std() == 0:
        return 0.0
    return float(np.corrcoef(rl, rr)[0, 1])


def single_part_t(x, clip=True) -> float:
    l, r, lib, na = _cells(x)
    if l.size < MIN_CELLS:
        return 0.0
    return _slope_t(r, _zclip(l) if clip else l, (lib, na))


def two_part(x, clip=True) -> float:
    """Team spec: S = p * t_intensity + (1 - p) * t_detection."""
    l, r, lib, na = _cells(x)
    if l.size < MIN_CELLS:
        return 0.0
    lv = _zclip(l) if clip else l
    det = (r > 0).astype(np.float64)
    p = float(det.mean())
    t_det = _slope_t(det, lv, (lib, na))
    on = r > 0
    t_int = _slope_t(r[on], lv[on], (lib[on], na[on])) if on.sum() >= MIN_CELLS else 0.0
    return p * t_int + (1.0 - p) * t_det


CANDIDATES = {
    "C1 Pearson (ligand z-clipped)": lambda x: pearson(x, True),
    "C2 partial Pearson | rows 2,3": lambda x: partial_pearson(x, True),
    "C3 single-part slope t | rows 2,3": lambda x: single_part_t(x, True),
    "C4 two-part t (TEAM SPEC)": lambda x: two_part(x, True),
    "C4b two-part t, no clipping": lambda x: two_part(x, False),
}
