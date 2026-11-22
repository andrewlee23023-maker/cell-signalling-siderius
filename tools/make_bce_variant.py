"""Team's round-2 hypothesis, tested locally as ONE change: iter1-noskip with smooth-L1 replaced by
BCE-with-logits, pos_weight = negatives / positives computed from the TRAIN labels inside train.py.
Creates submissions/iter1-noskip-bce from submissions/iter1-noskip (weights removed; retrained by model_eval).
"""
import re
import shutil
from pathlib import Path

W = Path.home() / "work/cell-signalling"
src, dst = W / "submissions/iter1-noskip", W / "submissions/iter1-noskip-bce"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
for f in ("weights.pt", "initial.pt"):
    (dst / f).unlink(missing_ok=True)
t = (dst / "train/train.py").read_text()
old_loss = "    loss_fn = torch.nn.SmoothL1Loss()\n"
new_loss = ("    pos_weight = float((yt == 0).sum() / (yt == 1).sum())  # negatives / positives, train labels only\n"
            "    print(f\"pos_weight {pos_weight:.3f}\", flush=True)\n"
            "    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))\n")
assert t.count(old_loss) == 1
t = t.replace(old_loss, new_loss)
t = re.sub(r"smooth-L1 loss on the binary target \(1 = injected\)",
           "BCE-with-logits on the binary target (1 = injected) with pos_weight = negatives/positives on train "
           "(team round-2 change; all else as iter1 without the skip connection)", t, count=1)
(dst / "train/train.py").write_text(t)
print("wrote", dst, "| loss line:", [l for l in t.splitlines() if "loss_fn =" in l][0].strip())
