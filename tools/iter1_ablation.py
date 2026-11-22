"""Team's Q1 ablation and Q3 beta diagnostic for the iter1 model (validation, official metric).

(a) partial-correlation z-score alone: z_p exactly as iter1's forward computes it (ranks within contact
    cells, ridge residualisation on [1, row2, row3] with lambda 0.001, Pearson of residuals, Fisher z with
    sqrt(n-3)); no learned weights.
(b) iter1 retrained with use_skip_zscore=false (3 seeds, from tools/model_eval.py runs).
(c) iter1 as is (3 seeds, from tools/model_eval.py runs).
Beta diagnostic: per dataset, AP of the positives with one beta against all negatives (other positives
removed), averaged over datasets — the task card's convention — for each predictor.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import OUT, W, _metrics, agg, base, boot_ci, data, dataset_index, per_ds, save, verify  # noqa: E402

verify()
va = data("validation")
b = base("validation")
pb = per_ds(va, b)


def partial_z(X: np.ndarray, ridge: float = 0.001) -> np.ndarray:
    x = torch.from_numpy(X).double()
    eps = 1e-8
    L, R, c1, c2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    m = (x[:, 4] > 0.5).double()
    n = m.sum(1).clamp(min=1.0)

    def mmean(v):
        return (v * m).sum(1) / n

    def mstd(v):
        mu = mmean(v)
        return torch.sqrt((((v - mu[:, None]) ** 2) * m).sum(1) / n + eps)

    def rank_unit(v):
        vm = v.masked_fill(m == 0, float("inf")).contiguous()
        s, _ = torch.sort(vm, dim=1)
        lo = torch.searchsorted(s, vm, right=False)
        hi = torch.searchsorted(s, vm, right=True)
        return ((0.5 * (lo + hi).double() - 0.5) / n[:, None] - 0.5) * m

    uL, uR = rank_unit(L), rank_unit(R)
    c1s = (c1 - mmean(c1)[:, None]) / mstd(c1)[:, None] * m
    c2s = (c2 - mmean(c2)[:, None]) / mstd(c2)[:, None] * m
    Zc = torch.stack([m, c1s, c2s], dim=-1) * m[..., None]
    Zt = Zc.transpose(1, 2)
    A = Zt @ Zc + (ridge + 1e-6) * torch.eye(3, dtype=x.dtype)[None]
    eL = (uL - (Zc @ torch.linalg.solve(A, Zt @ (uL * m)[..., None])).squeeze(-1)) * m
    eR = (uR - (Zc @ torch.linalg.solve(A, Zt @ (uR * m)[..., None])).squeeze(-1)) * m
    ac = (eL - mmean(eL)[:, None]) * m
    bc = (eR - mmean(eR)[:, None]) * m
    rho = (ac * bc).sum(1) / torch.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1) + 1e-12)
    z = torch.atanh(torch.clamp(rho, -0.999, 0.999)) * torch.sqrt(torch.clamp(n - 3.0, min=1.0))
    return torch.nan_to_num(z).numpy()


def model_scores(work_dir: str) -> np.ndarray:
    wd = Path(work_dir)
    spec = importlib.util.spec_from_file_location("m_" + wd.name, wd / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.CellSignalModel(**json.loads((wd / "cfg.json").read_text()))
    model.load_state_dict(torch.load(wd / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(va["X"][i:i + 256])).numpy().ravel() for i in range(0, len(va["X"]), 256)])


def beta_ap(s: np.ndarray) -> dict:
    out = {}
    for beta in (0.5, 1.0, 2.0):
        vals = []
        for ix in dataset_index(va).values():
            keep = ix[(va["beta"][ix] == 0) | np.isclose(va["beta"][ix], beta)]
            y = np.isclose(va["beta"][keep], beta)
            if 0 < y.sum() < len(y):
                vals.append(_metrics.average_precision(y, s[keep]))
        out[str(beta)] = float(np.mean(vals))
    return out


rows = {"baseline (contact Spearman)": [b], "(a) partial-corr z-score alone": [partial_z(va["X"])]}
for label, name in (("(b) iter1 without skip", "iter1-noskip"), ("(c) iter1 as is", "iter1-mlp")):
    runs = json.loads((OUT / f"model_eval_{name}.json").read_text())["rows"]
    rows[label] = [model_scores(r["work_dir"]) for r in runs]

res = {}
print(f"{'predictor':32} {'lr_auprc (mean ± sd over seeds)':>32} {'Δ vs baseline [95% CI], first seed':>36} {'dirAcc':>7}  AP by β 0.5 / 1 / 2")
for label, ss in rows.items():
    aucs = [agg(va, s)["lr_auprc"] for s in ss]
    dacc = float(np.mean([agg(va, s)["direction_accuracy"] for s in ss]))
    ps = per_ds(va, ss[0])
    dm, dlo, dhi = boot_ci(np.array([ps[k] - pb[k] for k in pb]))
    bt = {k: float(np.mean([beta_ap(s)[k] for s in ss])) for k in ("0.5", "1.0", "2.0")}
    res[label] = {"lr_auprc_mean": float(np.mean(aucs)), "lr_auprc_sd": float(np.std(aucs)), "n_seeds": len(ss),
                  "delta_first_seed": [dm, dlo, dhi], "direction_accuracy": dacc, "ap_by_beta": bt}
    print(f"{label:32} {np.mean(aucs):>22.4f} ± {np.std(aucs):.4f} ({len(ss)}) {dm:>+12.4f} [{dlo:+.3f}, {dhi:+.3f}] "
          f"{dacc:>7.3f}  {bt['0.5']:.3f} / {bt['1.0']:.3f} / {bt['2.0']:.3f}")
save("iter1_ablation.json", res)
