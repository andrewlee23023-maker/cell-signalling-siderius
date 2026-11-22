"""Option 1 (team request): iter1-noskip + contact-cell subsampling augmentation, ONE change.
Each training batch: every sample keeps a random fraction U(0.7, 1.0) of its contact cells; dropped cells are
zeroed in all 5 rows (mask row 4 = 0), which the model's masked statistics and pooling ignore. Randomness from a
torch.Generator seeded with the run seed. Inference and everything else unchanged.
Creates submissions/iter1-noskip-aug from submissions/iter1-noskip.
"""
import shutil
from pathlib import Path

W = Path.home() / "work/cell-signalling"
src, dst = W / "submissions/iter1-noskip", W / "submissions/iter1-noskip-aug"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
for f in ("weights.pt", "initial.pt"):
    (dst / f).unlink(missing_ok=True)
t = (dst / "train/train.py").read_text()

old_batch = "            xb, yb = Xt[idx].to(device), yt[idx].to(device)\n"
new_batch = ("            xb, yb = Xt[idx].clone(), yt[idx].to(device)\n"
             "            keep_frac = KEEP_LO + (1.0 - KEEP_LO) * torch.rand(len(idx), 1, generator=aug_gen)\n"
             "            drop = (torch.rand(len(idx), xb.shape[2], generator=aug_gen) > keep_frac) & (xb[:, 4] > 0)\n"
             "            xb = xb.masked_fill(drop[:, None, :], 0.0).to(device)  # dropped cells: all rows zero, mask 0\n")
assert t.count(old_batch) == 1
t = t.replace(old_batch, new_batch)
old_gen = "    gen = torch.Generator().manual_seed(a.seed)\n"
new_gen = old_gen + "    aug_gen = torch.Generator().manual_seed(a.seed + 1)  # augmentation stream, seeded from the run seed\n"
assert t.count(old_gen) == 1
t = t.replace(old_gen, new_gen)
t = t.replace("EPOCHS, LR, WD, BATCH, MIN_LR = 24, 0.002, 0.001, 32, 1e-5",
              "EPOCHS, LR, WD, BATCH, MIN_LR = 24, 0.002, 0.001, 32, 1e-5\nKEEP_LO = 0.7  # augmentation: keep U(0.7, 1.0) of contact cells per sample")
t = t.replace('"""Reproduce the iter1 SIDERIUS model from the train split only.',
              '"""Reproduce the iter1 SIDERIUS model (skip connection removed) from the train split only, with\n'
              'contact-cell subsampling augmentation (team option 1: keep U(0.7, 1.0) of contact cells per sample per batch).', 1)
(dst / "train/train.py").write_text(t)
print("wrote", dst)
