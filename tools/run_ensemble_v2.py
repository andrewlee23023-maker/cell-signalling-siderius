import sys
from pathlib import Path
import numpy as np
import torch
import importlib.util
import json
from scipy.stats import rankdata

sys.path.insert(0, '/home/andre/work/cell-signalling/tools')
from lh_common import W, base, boot_ci, data, per_ds, verify, dataset_index

verify()
va = data("validation")
pb = per_ds(va, base("validation"))
X_val = torch.from_numpy(va["X"]).float()
ds_idx = dataset_index(va)

def per_dataset_rank_norm(arr):
    res = np.zeros_like(arr)
    for ds_name, idxs in ds_idx.items():
        res[idxs] = rankdata(arr[idxs]) / len(idxs)
    return res

def predict_model(model_cls, weights_path, cfg=None):
    if cfg is not None:
        m = model_cls(**cfg) if isinstance(cfg, dict) else model_cls(cfg)
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
    print(f"{name:45s}: lr_auprc {m:.4f} [{lo:.3f}, {hi:.3f}] | vs base: {dm:+.4f} [{dlo:+.3f}, {dhi:+.3f}] | >base: {int((d>0).sum())}/{len(d)}")
    return s, m

predictions = {}

# 1. Submitted: iter1-noskip-gen
sub_gen = Path("/home/andre/work/cell-signalling/submissions/iter1-noskip-gen")
spec = importlib.util.spec_from_file_location("model_gen", sub_gen / "model.py")
mod_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_gen)
cfg_gen = json.loads((sub_gen / "cfg.json").read_text())
s_sub, _ = score_predictions(predict_model(mod_gen.CellSignalModel, sub_gen / "weights.pt", cfg_gen), "1. iter1-noskip-gen (submitted)")
predictions["sub_gen"] = s_sub

# 2. iter1-noskip (train only)
sub_noskip = Path("/home/andre/work/cell-signalling/submissions/iter1-noskip")
s_noskip, _ = score_predictions(predict_model(mod_gen.CellSignalModel, sub_noskip / "weights.pt", cfg_gen), "2. iter1-noskip (train only)")
predictions["noskip"] = s_noskip

# 3. explore2 (smooth-L1)
exp2_dir = Path("/home/andre/work/cell-signalling/project/results/runs/0759a6f3-4564-4865-a9d8-0d39b4c00bdf/workspace")
exp2_py = exp2_dir / "plugin_source_sentinel/lr_contact_contrast_gbm_head.py"
spec_exp2 = importlib.util.spec_from_file_location("model_exp2", exp2_py)
mod_exp2 = importlib.util.module_from_spec(spec_exp2)
spec_exp2.loader.exec_module(mod_exp2)
exp2_w = exp2_dir / "explore2/graph_instances/train/iteration_001/attempt_001/cached_models/model_lr_contact_contrast_gbm_head_lr_contact_contrast_gbm_head_explore2_001_agent.pth"
cfg_exp2 = mod_exp2.LrContactContrastGbmHeadConfig(segmentation_size=1)
s_exp2, _ = score_predictions(predict_model(lambda: mod_exp2.LrContactContrastGbmHead(cfg_exp2), exp2_w), "3. explore2 (smooth-L1)")
predictions["explore2"] = s_exp2

# 4. explore2_focal
focal_dir = Path("/home/andre/work/cell-signalling/project/results/runs/03a66c1e-19cf-4660-824e-702e532e69ef/workspace")
focal_py = focal_dir / "plugin_source_sentinel/lr_contact_contrast_gbm_head_v3.py"
spec_focal = importlib.util.spec_from_file_location("model_focal", focal_py)
mod_focal = importlib.util.module_from_spec(spec_focal)
spec_focal.loader.exec_module(mod_focal)
focal_w = focal_dir / "explore2_focal/graph_instances/train/iteration_001/attempt_001/cached_models/model_lr_contact_contrast_gbm_head_v3_lr_contact_contrast_gbm_head_v3_explore2_focal_001_agent.pth"
cfg_focal = mod_focal.LrContactContrastGbmHeadV3Config(segmentation_size=1)
s_focal, _ = score_predictions(predict_model(lambda: mod_focal.LrContactContrastGbmHeadV3(cfg_focal), focal_w), "4. explore2_focal (focal_cw)")
predictions["focal"] = s_focal

print("\n" + "=" * 80)
print("ENSEMBLE TESTING (Rank Normalized per Dataset)")
print("=" * 80)

rn_sub = per_dataset_rank_norm(s_sub)
rn_exp2 = per_dataset_rank_norm(s_exp2)
rn_focal = per_dataset_rank_norm(s_focal)
rn_noskip = per_dataset_rank_norm(s_noskip)

# Two-way combinations
score_predictions(0.70 * rn_sub + 0.30 * rn_exp2, "0.70 sub_gen + 0.30 explore2")
score_predictions(0.70 * rn_sub + 0.30 * rn_focal, "0.70 sub_gen + 0.30 explore2_focal")
score_predictions(0.80 * rn_sub + 0.20 * rn_focal, "0.80 sub_gen + 0.20 explore2_focal")

# Three-way combinations
score_predictions(0.60 * rn_sub + 0.25 * rn_exp2 + 0.15 * rn_focal, "0.60 sub_gen + 0.25 exp2 + 0.15 focal")
score_predictions(0.70 * rn_sub + 0.15 * rn_exp2 + 0.15 * rn_focal, "0.70 sub_gen + 0.15 exp2 + 0.15 focal")
score_predictions(0.50 * rn_sub + 0.25 * rn_exp2 + 0.25 * rn_focal, "0.50 sub_gen + 0.25 exp2 + 0.25 focal")
score_predictions(0.60 * rn_sub + 0.20 * rn_exp2 + 0.20 * rn_noskip, "0.60 sub_gen + 0.20 exp2 + 0.20 noskip")
