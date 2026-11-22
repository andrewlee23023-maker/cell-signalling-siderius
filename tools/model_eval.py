"""Train a staged kind=model submission on train only for given seeds, then score each with the official
metric on validation, with a dataset-bootstrap CI vs the baseline.

    python tools/model_eval.py submissions/iter1-mlp 20261121 1 2
Each seed trains in a fresh temporary copy (train data only), using the submission's own train/train.py.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lh_common import W, base, boot_ci, data, per_ds, save, verify  # noqa: E402

verify()
sub = (W / sys.argv[1]).resolve()
seeds = [int(s) for s in sys.argv[2:]]
va = data("validation")
pb = per_ds(va, base("validation"))
train_only = Path(tempfile.mkdtemp(prefix="trainonly-"))
import os  # noqa: E402

TRAIN_NPZ = Path(os.environ.get("TRAIN_NPZ", W / "data/train.npz"))  # e.g. data_aug/train.npz (train tiles only)
shutil.copy2(TRAIN_NPZ, train_only / "train.npz")
print(f"training file: {TRAIN_NPZ}")
rows = []
for seed in seeds:
    work = Path(tempfile.mkdtemp(prefix=f"seed{seed}-"))
    shutil.copytree(sub, work, dirs_exist_ok=True)
    for f in ("weights.pt", "initial.pt"):
        (work / f).unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "train/train.py", "--data_dir", str(train_only), "--seed", str(seed),
                        "--initial-out", "initial.pt", "--out", "weights.pt"], cwd=work, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"training failed for seed {seed}")
    print(r.stdout.strip().splitlines()[-1])
    spec = importlib.util.spec_from_file_location(f"m{seed}", work / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = json.loads((work / "cfg.json").read_text())
    model = mod.CellSignalModel(**cfg)
    model.load_state_dict(torch.load(work / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        s = np.concatenate([model(torch.from_numpy(va["X"][i:i + 256])).numpy().ravel() for i in range(0, len(va["X"]), 256)])
    ps = per_ds(va, s)
    d = np.array([ps[k] - pb[k] for k in pb])
    m, lo, hi = boot_ci(np.array(list(ps.values())))
    dm, dlo, dhi = boot_ci(d)
    rows.append({"seed": seed, "lr_auprc": m, "ci95": [lo, hi], "delta_vs_baseline": dm, "delta_ci95": [dlo, dhi],
                 "datasets_better": int((d > 0).sum()), "work_dir": str(work)})
    print(f"seed {seed}: lr_auprc {m:.4f} [{lo:.3f}, {hi:.3f}]; vs baseline {dm:+.4f} [{dlo:+.3f}, {dhi:+.3f}]; "
          f"better on {int((d > 0).sum())}/{len(d)} datasets")
    if seed == seeds[0]:
        for f in ("weights.pt", "initial.pt"):
            shutil.copy2(work / f, sub / f)
        print(f"copied seed {seed} weights.pt / initial.pt into {sub}")
vals = [r["lr_auprc"] for r in rows]
print(f"across seeds: mean {np.mean(vals):.4f}, sd {np.std(vals):.4f}, range {min(vals):.4f}-{max(vals):.4f}")
save(f"model_eval_{sub.name}.json", {"submission": str(sub), "rows": rows})
