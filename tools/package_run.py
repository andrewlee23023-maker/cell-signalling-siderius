"""Turn a finished SIDERIUS run into a kind=model submission folder (train-only, fixed epochs, self-contained).

    python tools/package_run.py RUN_ENVELOPE_DIR NAME [--seed 20261121]

Takes the best successful Formal record's model_config / train_config / loss_config and the run's generated
plugin source. Writes submissions/NAME/: model.py (plugin + CellSignalModel wrapper), cfg.json, train/lr_data.py
(task tensor builder, copied), train/train.py (recipe from the record; fixed epochs, last epoch kept — the
validation-loss checkpoint rule is not reproduced because validation must not be read), submission.json.
Supported loss: smooth_l1 (the only loss SIDERIUS allows for this task's regressor output).
"""
import argparse
import json
import shutil
from pathlib import Path

W = Path.home() / "work/cell-signalling"

WRAPPER = '''

# ---- submission entry point (added by tools/package_run.py) ----
_SIDERIUS_MODEL_CLASS = PLUGIN_MODEL_CLASS
_SIDERIUS_CONFIG_CLASS = PLUGIN_CONFIG_CLASS


class CellSignalModel(nn.Module):
    """Wraps the SIDERIUS-generated model. Build as CellSignalModel(**cfg_json) or from its config object.
    Input [B, 5, 1024] float32 -> output [B, 1] float32 (higher = more likely injected)."""

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            if "segmentation_size" in _SIDERIUS_CONFIG_CLASS.model_fields:
                kwargs.setdefault("segmentation_size", 1024)
            config = _SIDERIUS_CONFIG_CLASS(**kwargs)
        elif isinstance(config, dict):
            config = _SIDERIUS_CONFIG_CLASS(**config)
        self.net = _SIDERIUS_MODEL_CLASS(config)

    def forward(self, x):
        return self.net(x)


PLUGIN_MODEL_CLASS = CellSignalModel
'''

TRAIN = '''"""Reproduce the SIDERIUS run {run} model (exp {exp}) from the train split only.

    python train/train.py --data_dir $DATA_DIR --seed $SEED --initial-out initial.pt --out weights.pt

Recipe from the run record: loss {loss}; optimizer {opt} lr {lr} weight decay {wd}; batch {batch}; {sched_doc};
{epochs} epochs, last epoch kept (fixed horizon; validation is never read). initial.pt is saved before any
optimizer step.
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

from lr_data import load_split  # noqa: E402

EPOCHS, LR, WD, BATCH = {epochs}, {lr}, {wd}, {batch}
OPTIMIZER = "{opt}"
SCHEDULE = {sched}


def seed_everything(seed):
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
    return mod.CellSignalModel(**json.loads((ROOT / "cfg.json").read_text()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--initial-out", default="initial.pt")
    p.add_argument("--out", default="weights.pt")
    a = p.parse_args()
    t0 = time.time()
    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, _ = load_split(str(Path(a.data_dir) / "train.npz"))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).view(-1, 1)
    model = build_model()
    torch.save(model.state_dict(), a.initial_out)  # before any optimizer step
    model.to(device)
    opt_cls = {{"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}}[OPTIMIZER]
    opt = opt_cls(model.parameters(), lr=LR, weight_decay=WD)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SCHEDULE["horizon_epochs"], eta_min=SCHEDULE["min_lr"])
             if SCHEDULE and SCHEDULE.get("kind") == "cosine" else None)
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
        if sched is not None:
            sched.step()
        print(f"epoch {{epoch + 1}}/{{EPOCHS}} loss {{total / len(Xt):.5f}}", flush=True)
    model.cpu().eval()
    torch.save(model.state_dict(), a.out)
    print(f"saved {{a.initial_out}} and {{a.out}} in {{time.time() - t0:.0f}}s (device {{device}}, seed {{a.seed}})")


if __name__ == "__main__":
    sys.exit(main())
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("envelope", type=Path)
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=20261121)
    a = ap.parse_args()
    ws = a.envelope / "workspace"
    recs = []
    for f in ws.rglob("records/*/*.json"):
        if f.name.startswith("experiment_results"):
            continue
        d = json.loads(f.read_text())
        if d.get("status") == "success" and d.get("attempt_role") == "formal" and d.get("denoising_score") is not None:
            recs.append((d["denoising_score"], f, d))
    if not recs:
        raise SystemExit("no successful Formal record in this run")
    score, rec_path, d = max(recs, key=lambda r: r[0])
    p = d["params"]
    loss = (p.get("loss_config") or {}).get("loss_type", "smooth_l1")
    if loss != "smooth_l1":
        raise SystemExit(f"loss {loss!r} not supported by this packager")
    tc, mc, mt = p["train_config"], p["model_config"], p["model_type"]
    plugin = next(ws.rglob(f"graph_instances/train/**/plugins/*/{mt}.py"))
    out = W / "submissions" / a.name
    if out.exists():
        shutil.rmtree(out)
    (out / "train").mkdir(parents=True)
    (out / "model.py").write_text(plugin.read_text() + WRAPPER)
    (out / "cfg.json").write_text(json.dumps(mc) + "\n")
    shutil.copy2(W / "submissions/iter1-mlp/train/lr_data.py", out / "train/lr_data.py")
    sched = tc.get("lr_schedule")
    sched_doc = f"cosine over {sched['horizon_epochs']} epochs to {sched['min_lr']}" if sched and sched.get("kind") == "cosine" else "constant lr"
    (out / "train/train.py").write_text(TRAIN.format(
        run=p.get("run_name"), exp=p.get("exp_id"), loss=loss, opt=tc.get("optimizer_type", "adamw"), lr=tc["lr"],
        wd=tc.get("weight_decay", 0.0), batch=tc.get("batch_size", 32), epochs=tc["epochs"],
        sched=json.dumps(sched) if sched else "None", sched_doc=sched_doc))
    (out / "submission.json").write_text(json.dumps({
        "kind": "model", "seed": a.seed, "initial_weights": "initial.pt", "plugin": "model.py",
        "symbol": "CellSignalModel", "weights": "weights.pt", "plugin_config": "cfg.json",
        "train_command": "python train/train.py --data_dir $DATA_DIR --seed $SEED --initial-out initial.pt --out weights.pt",
        "train_dir": "train/"}, indent=2) + "\n")
    print(f"packaged {mt} from {rec_path.name} (SIDERIUS Formal {score:.4f}) -> {out}")
    print(f"  recipe: {loss}, {tc.get('optimizer_type')} lr {tc['lr']} wd {tc.get('weight_decay')} batch {tc.get('batch_size')} "
          f"epochs {tc['epochs']} schedule {sched}; checkpoint rule in SIDERIUS: {tc.get('checkpoint_selection')}")


if __name__ == "__main__":
    main()
