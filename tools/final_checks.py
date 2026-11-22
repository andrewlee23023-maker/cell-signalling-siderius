"""Team falsification tests 2 and 4 for the uploaded model (validation, official metric):
  test 2: AP by injected effect size beta (positives of one beta vs all negatives, per dataset, averaged)
          for baseline, (b) iter1 without skip, uploaded (b)+generator data;
  test 4: padding mask disabled at evaluation (row 4 set to 1 on padded columns, so padding is read as cells).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import OUT, _metrics, agg, base, data, dataset_index, save, verify  # noqa: E402

verify()
va = data("validation")


def load_model(name):
    wd = Path(json.loads((OUT / f"model_eval_{name}.json").read_text())["rows"][0]["work_dir"])
    sp = importlib.util.spec_from_file_location("m_" + name.replace("-", "_"), wd / "model.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    m = mod.CellSignalModel(**json.loads((wd / "cfg.json").read_text()))
    m.load_state_dict(torch.load(wd / "weights.pt", map_location="cpu", weights_only=True))
    return m.eval()


def run(m, X):
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(X[i:i + 256])).numpy().ravel() for i in range(0, len(X), 256)])


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


final = load_model("iter1-noskip-gen")
preds = {"baseline": base("validation"), "(b) iter1 without skip": run(load_model("iter1-noskip"), va["X"]),
         "uploaded (b)+generator": run(final, va["X"])}
res = {"beta": {k: beta_ap(s) for k, s in preds.items()}}
print("test 2 — AP by beta (0.5 / 1 / 2):")
for k, v in res["beta"].items():
    print(f"  {k:26} {v['0.5']:.3f} / {v['1.0']:.3f} / {v['2.0']:.3f}")
Xn = va["X"].copy()
Xn[:, 4, :] = 1.0  # padding now looks like real (all-zero) cells
s_nomask = run(final, Xn)
res["mask_disabled"] = agg(va, s_nomask)
res["intact"] = agg(va, preds["uploaded (b)+generator"])
print(f"test 4 — uploaded model, padding mask disabled: lr_auprc {res['mask_disabled']['lr_auprc']:.4f} "
      f"(intact {res['intact']['lr_auprc']:.4f}); direction {res['mask_disabled']['direction_accuracy']:.3f}")
save("final_checks.json", res)
