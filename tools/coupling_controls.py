"""Does the score come from the ligand->receptor coupling, or from something else? (validation, official metric)

Controls applied to the INPUT at evaluation time, per triple, over its contact cells only (mask unchanged):
  shuffle_receptor : permute row 1 across contact cells (keeps every receptor marginal; destroys L-R alignment)
  shuffle_ligand   : permute row 0 across contact cells
  zero_covariates  : set rows 2-3 to 0 on contact cells
Each shuffle is repeated over 5 seeds (mean ± SD). A predictor that uses only the coupling should fall to the
random level (0.114 ± 0.014) under either shuffle; any remainder comes from non-coupling information.
Predictors: official baseline, iter1 without skip (b), uploaded model (b) + generator data (submission seed).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import OUT, agg, data, save, verify  # noqa: E402
from cs_data import PKG  # noqa: E402

verify()
va = data("validation")
X0 = va["X"]
spec = importlib.util.spec_from_file_location("base_fm", PKG / "baseline/final_model.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def load_model(name):
    wd = Path(json.loads((OUT / f"model_eval_{name}.json").read_text())["rows"][0]["work_dir"])
    sp = importlib.util.spec_from_file_location("m_" + name.replace("-", "_"), wd / "model.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    m = mod.CellSignalModel(**json.loads((wd / "cfg.json").read_text()))
    m.load_state_dict(torch.load(wd / "weights.pt", map_location="cpu", weights_only=True))
    return m.eval()


models = {"(b) iter1 without skip": load_model("iter1-noskip"),
          "(b) + generator data (uploaded)": load_model("iter1-noskip-gen")}


def score_all(X):
    out = {"baseline (contact Spearman)": np.array([bm.predict(x) for x in X])}
    with torch.no_grad():
        for k, m in models.items():
            out[k] = np.concatenate([m(torch.from_numpy(X[i:i + 256])).numpy().ravel() for i in range(0, len(X), 256)])
    return out


def control(kind, seed):
    rng = np.random.default_rng(seed)
    X = X0.copy()
    for i in range(len(X)):
        m = X[i, 4] > 0
        idx = np.flatnonzero(m)
        if kind == "shuffle_receptor":
            X[i, 1, idx] = X[i, 1, rng.permutation(idx)]
        elif kind == "shuffle_ligand":
            X[i, 0, idx] = X[i, 0, rng.permutation(idx)]
        elif kind == "zero_covariates":
            X[i, 2:4, idx] = 0.0
    return X


res = {"intact": {k: agg(va, s) for k, s in score_all(X0).items()}}
for kind, seeds in (("shuffle_receptor", range(5)), ("shuffle_ligand", range(5)), ("zero_covariates", [0])):
    runs = [score_all(control(kind, s)) for s in seeds]
    res[kind] = {k: {"lr_auprc": float(np.mean([agg(va, r[k])["lr_auprc"] for r in runs])),
                     "lr_auprc_sd": float(np.std([agg(va, r[k])["lr_auprc"] for r in runs])),
                     "direction_accuracy": float(np.mean([agg(va, r[k])["direction_accuracy"] for r in runs]))}
                 for k in runs[0]}
names = list(res["intact"])
print("lr_auprc (direction_accuracy); random level 0.114 ± 0.014 / 0.50 ± 0.04")
print(f"{'condition':18} " + " ".join(f"{n:>34}" for n in names))
for cond in ("intact", "shuffle_receptor", "shuffle_ligand", "zero_covariates"):
    cells = []
    for n in names:
        r = res[cond][n]
        sd = f"±{r['lr_auprc_sd']:.3f}" if r.get("lr_auprc_sd") else "      "
        cells.append(f"{r['lr_auprc']:.3f}{sd} ({r['direction_accuracy']:.3f})")
    print(f"{cond:18} " + " ".join(f"{c:>34}" for c in cells))
save("coupling_controls.json", res)
