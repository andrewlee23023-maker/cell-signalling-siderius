"""Report evidence for the ensemble submissions (validation, official metric), from the STAGED weights
(byte-identical to a retrain with seed 20261121):
  - lr_auprc with 7-tile bootstrap 95% CI and paired delta vs baseline (same method as tile_bootstrap.py)
    for: v2 ensemble, aug-both ensemble, and each one's m1 / m2 component alone;
  - AP by injected beta (same as final_checks.py);
  - coupling controls on both ensembles: shuffle receptor / ligand (5 seeds), rows 2-3 zeroed, padding mask off.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import W, _metrics, agg, base, data, dataset_index, per_ds, save, verify  # noqa: E402

verify()
va = data("validation")
X0 = va["X"]


def load(sub):
    d = W / "submissions" / sub
    sp = importlib.util.spec_from_file_location("m_" + sub.replace("-", "_"), d / "model.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    m = mod.CellSignalModel(**json.loads((d / "cfg.json").read_text()))
    m.load_state_dict(torch.load(d / "weights.pt", map_location="cpu", weights_only=True))
    return m.eval()


def run(f, X):
    with torch.no_grad():
        return np.concatenate([f(torch.from_numpy(X[i:i + 256])).numpy().ravel() for i in range(0, len(X), 256)])


ens = {"v2 ensemble (submitted)": load("ensemble-noskip-explore2-v2"), "aug-both ensemble": load("ensemble-aug-both")}
preds = {"baseline": base("validation")}
for k, m in ens.items():
    preds[k] = run(m, X0)
    preds[k + " | m1 alone"] = run(m.m1, X0)
    preds[k + " | m2 alone"] = run(m.m2, X0)

# tile bootstrap (identical to tile_bootstrap.py)
pb = per_ds(va, preds["baseline"])
dsets = sorted(pb)
tile_of = np.array([d.rsplit("_r", 1)[0] for d in dsets])
tiles = sorted(set(tile_of))
draws = np.random.default_rng(0).integers(0, len(tiles), size=(5000, len(tiles)))
res = {"tile_bootstrap": {}}
print(f"{'predictor':42} {'lr_auprc [95% tile CI]':>24} {'delta [95% tile CI]':>26}")
for label, s in preds.items():
    ps = per_ds(va, s)
    v = np.array([ps[d] for d in dsets])
    dv = v - np.array([pb[d] for d in dsets])
    tm = np.array([v[tile_of == t].mean() for t in tiles])
    td = np.array([dv[tile_of == t].mean() for t in tiles])
    lo, hi = np.percentile(tm[draws].mean(axis=1), [2.5, 97.5])
    dlo, dhi = np.percentile(td[draws].mean(axis=1), [2.5, 97.5])
    a = agg(va, s)
    res["tile_bootstrap"][label] = {"lr_auprc": float(v.mean()), "ci95_tile": [float(lo), float(hi)],
                                    "delta": float(dv.mean()), "delta_ci95_tile": [float(dlo), float(dhi)],
                                    "datasets_above_baseline": int((dv > 0).sum()),
                                    "direction_accuracy": a["direction_accuracy"]}
    print(f"{label:42} {v.mean():.4f} [{lo:.3f}, {hi:.3f}] {dv.mean():+.4f} [{dlo:+.3f}, {dhi:+.3f}] "
          f">base {(dv > 0).sum()}/21 dir {a['direction_accuracy']:.3f}")


def beta_ap(s):
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


res["beta"] = {k: beta_ap(preds[k]) for k in ["baseline", *ens]}
print("AP by beta (0.5 / 1 / 2):")
for k, v in res["beta"].items():
    print(f"  {k:30} {v['0.5']:.3f} / {v['1.0']:.3f} / {v['2.0']:.3f}")


def control(kind, seed):
    rng = np.random.default_rng(seed)
    X = X0.copy()
    if kind == "mask_off":
        X[:, 4, :] = 1.0
        return X
    for i in range(len(X)):
        idx = np.flatnonzero(X[i, 4] > 0)
        if kind == "shuffle_receptor":
            X[i, 1, idx] = X[i, 1, rng.permutation(idx)]
        elif kind == "shuffle_ligand":
            X[i, 0, idx] = X[i, 0, rng.permutation(idx)]
        elif kind == "zero_covariates":
            X[i, 2:4, idx] = 0.0
    return X


res["controls"] = {}
for kind, seeds in (("shuffle_receptor", range(5)), ("shuffle_ligand", range(5)), ("zero_covariates", [0]), ("mask_off", [0])):
    Xs = [control(kind, s) for s in seeds]
    res["controls"][kind] = {}
    for k, m in ens.items():
        a = [agg(va, run(m, X)) for X in Xs]
        r = {"lr_auprc": float(np.mean([x["lr_auprc"] for x in a])), "lr_auprc_sd": float(np.std([x["lr_auprc"] for x in a])),
             "direction_accuracy": float(np.mean([x["direction_accuracy"] for x in a]))}
        res["controls"][kind][k] = r
        print(f"control {kind:17} {k:26} {r['lr_auprc']:.3f}±{r['lr_auprc_sd']:.3f} dir {r['direction_accuracy']:.3f}")
save("ensemble_report_stats.json", res)
