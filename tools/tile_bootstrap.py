"""Team point 8: uncertainty with a bootstrap that resamples the 7 validation TILES (all 3 realisations of a
tile move together), not the 21 datasets. For each predictor: lr_auprc with a tile-bootstrap 95% CI, and the
paired difference vs the baseline with its tile-bootstrap CI. 5,000 reps, seed 0.
Trained models use the submission seed's weights from tools/model_eval.py runs.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import OUT, base, data, per_ds, save, verify  # noqa: E402

verify()
va = data("validation")
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("abl", Path(__file__).with_name("iter1_ablation.py"))


def partial_z_from_ablation():
    src = Path(__file__).with_name("iter1_ablation.py").read_text()
    start = src.index("def partial_z")
    end = src.index("def model_scores")
    ns: dict = {"np": np, "torch": torch}
    exec(src[start:end], ns)  # noqa: S102 — our own file; reuse the exact z-score code
    return ns["partial_z"](va["X"])


def model_scores(work_dir):
    wd = Path(work_dir)
    sp = importlib.util.spec_from_file_location("m_" + wd.name, wd / "model.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    model = mod.CellSignalModel(**json.loads((wd / "cfg.json").read_text()))
    model.load_state_dict(torch.load(wd / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(va["X"][i:i + 256])).numpy().ravel() for i in range(0, len(va["X"]), 256)])


preds = {"baseline (contact Spearman)": base("validation"), "(a) partial-corr z-score alone": partial_z_from_ablation()}
for label, name in (("(c) iter1 as is", "iter1-mlp"), ("(b) iter1 without skip", "iter1-noskip"),
                    ("iter1 without skip + BCE pos_weight", "iter1-noskip-bce"),
                    ("iter1 without skip + contact subsampling", "iter1-noskip-aug"),
                    ("iter1 without skip + generator data (uploaded)", "iter1-noskip-gen")):
    rows = json.loads((OUT / f"model_eval_{name}.json").read_text())["rows"]
    preds[label] = model_scores(rows[0]["work_dir"])

pb = per_ds(va, preds["baseline (contact Spearman)"])
dsets = sorted(pb)
tile_of = np.array([d.rsplit("_r", 1)[0] for d in dsets])
tiles = sorted(set(tile_of))
rng = np.random.default_rng(0)
draws = rng.integers(0, len(tiles), size=(5000, len(tiles)))
res = {}
print(f"tile bootstrap: {len(tiles)} tiles x 3 realisations, 5,000 reps")
print(f"{'predictor':38} {'lr_auprc [95% tile CI]':>26} {'Δ vs baseline [95% tile CI]':>30}")
for label, s in preds.items():
    ps = per_ds(va, s)
    v = np.array([ps[d] for d in dsets])
    dv = v - np.array([pb[d] for d in dsets])
    tile_mean = np.array([v[tile_of == t].mean() for t in tiles])
    tile_diff = np.array([dv[tile_of == t].mean() for t in tiles])
    bm, bd = tile_mean[draws].mean(axis=1), tile_diff[draws].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    dlo, dhi = np.percentile(bd, [2.5, 97.5])
    res[label] = {"lr_auprc": float(v.mean()), "ci95_tile": [float(lo), float(hi)],
                  "delta": float(dv.mean()), "delta_ci95_tile": [float(dlo), float(dhi)]}
    print(f"{label:38} {v.mean():>8.4f} [{lo:.3f}, {hi:.3f}] {dv.mean():>+12.4f} [{dlo:+.3f}, {dhi:+.3f}]")
save("tile_bootstrap.json", res)
