"""Train the fixed-weight score ensemble from the train split only.

    python train/train.py --data_dir $DATA_DIR --seed $SEED --initial-out initial.pt --out weights.pt

The ensemble (../model.py, CellSignalModel) outputs w1 * m1(x) + w2 * m2(x) with w1, w2 fixed in ../cfg.json.
The two components are trained separately, each with its own recipe, then saved together as one state_dict
(keys m1.* and m2.*). initial.pt is the whole ensemble right after construction, before any optimizer step.

m1 — iter1 SIDERIUS model without the skip connection (lr_coupling_mlp). Data: train.npz plus 6 generator
     realisations per TRAIN tile made by train/augment.py with the task's public generator (train/build_dataset.py,
     Zenodo 10.5281/zenodo.24095146, MIT, unchanged; seed = $SEED), written to ./train_augmented.npz.
     Recipe: smooth-L1 on the binary target; AdamW lr 0.002, weight decay 0.001; batch 32; cosine over 24 epochs
     down to 1e-5; last epoch kept.
m2 — explore2 SIDERIUS model (lr_contact_contrast_gbm_head, default config). Data: train.npz only.
     Recipe: smooth-L1; AdamW lr 0.001, weight decay 1e-4; batch 128; 60 epochs, constant lr; last epoch kept.
No validation or test data are read.
"""

import argparse
import json
import os
import random
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from augment import make as augment_train  # noqa: E402
from lr_data import load_split  # noqa: E402

N_EXTRA_REALISATIONS = 6  # per train tile, for m1
M1 = dict(epochs=24, lr=0.002, wd=0.001, batch=32, cosine_min_lr=1e-5)
M2 = dict(epochs=60, lr=0.001, wd=0.0001, batch=128, cosine_min_lr=None)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def build_model():
    spec = spec_from_file_location("submission_model", ROOT / "model.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = json.loads((ROOT / "cfg.json").read_text())
    return mod.CellSignalModel(**cfg)


def train_component(name, module, X, y, recipe, seed, device, t0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).view(-1, 1)
    module.to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=recipe["lr"], weight_decay=recipe["wd"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe["epochs"], eta_min=recipe["cosine_min_lr"])
             if recipe["cosine_min_lr"] is not None else None)
    loss_fn = torch.nn.SmoothL1Loss()
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(recipe["epochs"]):
        module.train()
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(order), recipe["batch"]):
            idx = order[i:i + recipe["batch"]]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(module(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if sched is not None:
            sched.step()
        print(f"[{name}] epoch {epoch + 1}/{recipe['epochs']} loss {total / len(Xt):.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    module.cpu().eval()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--initial-out", default="initial.pt")
    p.add_argument("--out", default="weights.pt")
    a = p.parse_args()
    t0 = time.time()
    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_npz = str(Path(a.data_dir) / "train.npz")
    aug = augment_train(train_npz, str(Path.cwd() / "train_augmented.npz"), N_EXTRA_REALISATIONS, a.seed)
    print(f"augmented train data written to {aug} ({time.time() - t0:.0f}s)", flush=True)

    model = build_model()
    torch.save(model.state_dict(), a.initial_out)  # whole ensemble, before any optimizer step

    X1, y1, _ = load_split(aug)
    print(f"m1 train tensors: X {X1.shape}, positives {int(y1.sum())}", flush=True)
    train_component("m1", model.m1, X1, y1, M1, a.seed, device, t0)
    del X1, y1

    X2, y2, _ = load_split(train_npz)
    print(f"m2 train tensors: X {X2.shape}, positives {int(y2.sum())}", flush=True)
    train_component("m2", model.m2, X2, y2, M2, a.seed + 1, device, t0)

    model.cpu().eval()
    torch.save(model.state_dict(), a.out)
    print(f"saved {a.initial_out} and {a.out} in {time.time() - t0:.0f}s (device {device}, seed {a.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
