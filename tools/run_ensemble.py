import sys
from pathlib import Path
import numpy as np
import torch
import importlib.util

sys.path.insert(0, '/home/andre/work/cell-signalling/tools')
from lh_common import W, base, boot_ci, data, per_ds, verify

verify()
va = data("validation")
pb = per_ds(va, base("validation"))
X_val = torch.from_numpy(va["X"]).float()

def predict_model(model_cls, weights_path, cfg=None):
    if cfg is not None:
        m = model_cls(**cfg)
    else:
        m = model_cls()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_val), 256):
            batch = X_val[i:i+256]
            out = m(batch)
            if isinstance(out, tuple): out = out[0]
            preds.append(out.numpy().ravel())
        return np.concatenate(preds)

def score_predictions(s, name="Model"):
    ps = per_ds(va, s)
    d = np.array([ps[k] - pb[k] for k in pb])
    m, lo, hi = boot_ci(np.array(list(ps.values())))
    dm, dlo, dhi = boot_ci(d)
    print(f"{name:35s}: lr_auprc {m:.4f} [{lo:.3f}, {hi:.3f}] | vs base: {dm:+.4f} [{dlo:+.3f}, {dhi:+.3f}] | >base: {int((d>0).sum())}/{len(d)}")
    return s, m

predictions = {}

# 1. Submitted Model: iter1-noskip-gen
sub_gen = Path("/home/andre/work/cell-signalling/submissions/iter1-noskip-gen")
spec = importlib.util.spec_from_file_location("model_gen", sub_gen / "model.py")
mod_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_gen)
import json
cfg_gen = json.loads((sub_gen / "cfg.json").read_text())
s_sub, _ = score_predictions(predict_model(mod_gen.CellSignalModel, sub_gen / "weights.pt", cfg_gen), "1. iter1-noskip-gen (submitted)")
predictions["sub_gen"] = s_sub

# 2. iter1-noskip (trained on train only)
sub_noskip = Path("/home/andre/work/cell-signalling/submissions/iter1-noskip")
s_noskip, _ = score_predictions(predict_model(mod_gen.CellSignalModel, sub_noskip / "weights.pt", cfg_gen), "2. iter1-noskip (train only)")
predictions["noskip"] = s_noskip

# 3. iter1-mlp (with skip)
sub_mlp = Path("/home/andre/work/cell-signalling/submissions/iter1-mlp")
spec_mlp = importlib.util.spec_from_file_location("model_mlp", sub_mlp / "model.py")
mod_mlp = importlib.util.module_from_spec(spec_mlp)
spec_mlp.loader.exec_module(mod_mlp)
cfg_mlp = json.loads((sub_mlp / "cfg.json").read_text())
s_mlp, _ = score_predictions(predict_model(mod_mlp.CellSignalModel, sub_mlp / "weights.pt", cfg_mlp), "3. iter1-mlp")
predictions["mlp"] = s_mlp

# 4. explore2 model
exp2_dir = Path("/home/andre/work/cell-signalling/project/results/runs/0759a6f3-4564-4865-a9d8-0d39b4c00bdf/workspace")
exp2_py = exp2_dir / "plugin_source_sentinel/lr_contact_contrast_gbm_head.py"
spec_exp2 = importlib.util.spec_from_file_location("model_exp2", exp2_py)
mod_exp2 = importlib.util.module_from_spec(spec_exp2)
spec_exp2.loader.exec_module(mod_exp2)
exp2_w = exp2_dir / "explore2/graph_instances/train/iteration_001/attempt_001/cached_models/model_lr_contact_contrast_gbm_head_lr_contact_contrast_gbm_head_explore2_001_agent.pth"
cfg_exp2 = mod_exp2.LrContactContrastGbmHeadConfig(segmentation_size=1)
s_exp2, _ = score_predictions(predict_model(lambda: mod_exp2.LrContactContrastGbmHead(cfg_exp2), exp2_w), "4. explore2 (agent 001)")
predictions["explore2"] = s_exp2

# Ensembles
print("-" * 75)
print("ENSEMBLE EXPERIMENTS:")
print("-" * 75)

# Standardize / rank-normalize scores before blending to avoid scale mismatch
def rank_norm(arr):
    from scipy.stats import rankdata
    return rankdata(arr) / len(arr)

from lh_common import dataset_index
ds_idx = dataset_index(va)

def per_dataset_rank_norm(arr):
    from scipy.stats import rankdata
    res = np.zeros_like(arr)
    for ds_name, idxs in ds_idx.items():
        res[idxs] = rankdata(arr[idxs]) / len(idxs)
    return res

# Mean of raw scores
score_predictions((s_sub + s_exp2) / 2.0, "Ensemble: sub_gen + explore2 (raw avg)")
score_predictions((s_sub + s_noskip + s_exp2) / 3.0, "Ensemble: 3 models (raw avg)")

# Per-dataset rank normalized average
rn_sub = per_dataset_rank_norm(s_sub)
rn_exp2 = per_dataset_rank_norm(s_exp2)
rn_noskip = per_dataset_rank_norm(s_noskip)
rn_mlp = per_dataset_rank_norm(s_mlp)

score_predictions((rn_sub + rn_exp2) / 2.0, "Ensemble: sub_gen + exp2 (rank avg)")
score_predictions(0.7 * rn_sub + 0.3 * rn_exp2, "Ensemble: 0.7 sub_gen + 0.3 exp2 (rank)")
score_predictions(0.8 * rn_sub + 0.2 * rn_exp2, "Ensemble: 0.8 sub_gen + 0.2 exp2 (rank)")
score_predictions((rn_sub + rn_noskip + rn_exp2) / 3.0, "Ensemble: 3 models (rank avg)")
score_predictions((rn_sub + rn_exp2 + rn_mlp) / 3.0, "Ensemble: sub_gen + exp2 + mlp (rank)")
score_predictions(0.6 * rn_sub + 0.25 * rn_exp2 + 0.15 * rn_noskip, "Ensemble: weighted (0.6/0.25/0.15)")
