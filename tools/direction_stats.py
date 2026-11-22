"""Direction evidence (validation, official direction_accuracy = share of injected (L,R,A->B) with
score(A->B) > score(B->A)): baseline vs final ensemble (v2) and its components; paired difference with the
7-tile bootstrap (same method as tile_bootstrap.py); direction accuracy by injected beta."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import W, base, data, save, verify  # noqa: E402
from cs_data import score  # noqa: E402

verify()
va = data("validation")
print("keys:", sorted(k for k in va if not k.startswith("_")))
def load(sub):
    d = W / "submissions" / sub
    sp = importlib.util.spec_from_file_location("m_" + sub.replace("-", "_"), d / "model.py")
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    m = mod.CellSignalModel(**json.loads((d / "cfg.json").read_text()))
    m.load_state_dict(torch.load(d / "weights.pt", map_location="cpu", weights_only=True)); return m.eval()
def run(m):
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(va["X"][i:i + 256])).numpy().ravel() for i in range(0, len(va["X"]), 256)])
v2 = load("ensemble-noskip-explore2-v2")
preds = {"baseline": base("validation"), "final R7": run(v2), "m1": run(v2.m1), "m2": run(v2.m2)}
tabs = {k: score(va, s)["per_dataset"] for k, s in preds.items()}
dsets = sorted(tabs["baseline"])
tile_of = np.array([d.rsplit("_r", 1)[0] for d in dsets]); tiles = sorted(set(tile_of))
draws = np.random.default_rng(0).integers(0, len(tiles), size=(5000, len(tiles)))
res = {"direction": {}}
b = np.array([tabs["baseline"][d]["direction_accuracy"] for d in dsets])
for k in preds:
    v = np.array([tabs[k][d]["direction_accuracy"] for d in dsets]); dv = v - b
    tm = np.array([v[tile_of == t].mean() for t in tiles]); td = np.array([dv[tile_of == t].mean() for t in tiles])
    lo, hi = np.percentile(tm[draws].mean(1), [2.5, 97.5]); dlo, dhi = np.percentile(td[draws].mean(1), [2.5, 97.5])
    res["direction"][k] = {"mean": float(v.mean()), "ci95_tile": [float(lo), float(hi)], "delta": float(dv.mean()),
                           "delta_ci95_tile": [float(dlo), float(dhi)], "datasets_above_baseline": int((dv > 0).sum()),
                           "datasets_below_baseline": int((dv < 0).sum())}
    print(f"{k:10} dir {v.mean():.3f} [{lo:.3f},{hi:.3f}]  delta {dv.mean():+.3f} [{dlo:+.3f},{dhi:+.3f}]  >{(dv>0).sum()} <{(dv<0).sum()}")
# by beta, pooled over datasets: reverse pair lookup via sample ids
ids = list(va["sample_id"])
if ids is not None:
    idx = {s: i for i, s in enumerate(ids)}
    res["direction_by_beta"] = {}
    for k, s in preds.items():
        out = {}
        for beta in (0.5, 1.0, 2.0):
            acc = []
            for i, sid in enumerate(ids):
                if not np.isclose(va["beta"][i], beta):
                    continue
                d, rest = sid.split(":", 1); l, r, a, bb = rest.split("|")
                j = idx.get(f"{d}:{l}|{r}|{bb}|{a}")
                rv = s[j] if j is not None else -np.inf
                acc.append(1.0 if s[i] > rv else 0.5 if s[i] == rv else 0.0)
            out[str(beta)] = float(np.mean(acc)); out["n_" + str(beta)] = len(acc)
        res["direction_by_beta"][k] = out
        print(f"{k:10} dir by beta 0.5/1/2: {out['0.5']:.3f} / {out['1.0']:.3f} / {out['2.0']:.3f}  (n {out['n_0.5']}/{out['n_1.0']}/{out['n_2.0']})")
save("direction_stats.json", res)
