"""Reproduce the iter1 SIDERIUS model (skip connection removed) from the train split only, trained on the
train datasets plus generator realisations of the TRAIN tiles.

    python train/train.py --data_dir $DATA_DIR --seed $SEED --initial-out initial.pt --out weights.pt

Step 1 (augmentation): train/augment.py runs the task's public generator (train/build_dataset.py, Zenodo
10.5281/zenodo.24095146, MIT, unchanged) on the 10 train tiles in train.npz to add 6 new realisations per tile
(60 datasets, seed = $SEED), written to ./train_augmented.npz. No validation or test data are read.
Step 2 (training): iter1 recipe — model config in ../cfg.json (use_skip_zscore false); smooth-L1 loss on the
binary target; AdamW lr 0.002, weight decay 0.001; batch 32; cosine schedule over 24 epochs down to 1e-5;
fixed 24 epochs (last epoch kept). initial.pt is saved right after construction, before any optimizer step.
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

EPOCHS, LR, WD, BATCH, MIN_LR = 24, 0.002, 0.001, 32, 1e-5
N_EXTRA_REALISATIONS = 6  # per train tile


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
    aug = augment_train(str(Path(a.data_dir) / "train.npz"), str(Path.cwd() / "train_augmented.npz"),
                        N_EXTRA_REALISATIONS, a.seed)
    print(f"augmented train data written to {aug} ({time.time() - t0:.0f}s)", flush=True)
    X, y, _ = load_split(aug)
    print(f"train tensors: X {X.shape}, positives {int(y.sum())} ({time.time() - t0:.0f}s)", flush=True)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).view(-1, 1)

    model = build_model()
    torch.save(model.state_dict(), a.initial_out)  # before any optimizer step
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=MIN_LR)
    loss_fn = torch.nn.SmoothL1Loss()
    gen = torch.Generator().manual_seed(a.seed)
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        print(f"epoch {epoch + 1}/{EPOCHS} loss {total / len(Xt):.5f}", flush=True)
    model.cpu().eval()
    torch.save(model.state_dict(), a.out)
    print(f"saved {a.initial_out} and {a.out} in {time.time() - t0:.0f}s (device {device}, seed {a.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
