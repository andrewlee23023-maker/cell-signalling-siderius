import torch
from pathlib import Path

out_dir = Path("/home/andre/work/cell-signalling/submissions/ensemble-noskip-explore2")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. iter1-noskip-gen weights
sd_gen = torch.load('/home/andre/work/cell-signalling/submissions/iter1-noskip-gen/weights.pt', map_location='cpu', weights_only=True)
print('gen keys:', len(sd_gen))

# 2. explore2 weights
sd_exp = torch.load('/home/andre/work/cell-signalling/project/results/runs/0759a6f3-4564-4865-a9d8-0d39b4c00bdf/workspace/explore2/graph_instances/train/iteration_001/attempt_001/cached_models/model_lr_contact_contrast_gbm_head_lr_contact_contrast_gbm_head_explore2_001_agent.pth', map_location='cpu', weights_only=True)
if 'state_dict' in sd_exp:
    sd_exp = sd_exp['state_dict']
print('exp keys:', len(sd_exp))

# Combined state dict
combined_weights = {}
for k, v in sd_gen.items():
    combined_weights['m1.' + k] = v
for k, v in sd_exp.items():
    combined_weights['m2.' + k] = v

print('combined keys:', len(combined_weights), 'total params:', sum(v.numel() for v in combined_weights.values()))
torch.save(combined_weights, out_dir / 'weights.pt')

# Initial weights
init_gen = torch.load('/home/andre/work/cell-signalling/submissions/iter1-noskip-gen/initial.pt', map_location='cpu', weights_only=True)
init_exp = torch.load('/home/andre/work/cell-signalling/project/results/runs/0759a6f3-4564-4865-a9d8-0d39b4c00bdf/workspace/explore2/graph_instances/train/iteration_001/attempt_001/cached_models/model_lr_contact_contrast_gbm_head_lr_contact_contrast_gbm_head_explore2_001_agent.pth', map_location='cpu', weights_only=True)
if 'state_dict' in init_exp: init_exp = init_exp['state_dict']
combined_init = {}
for k, v in init_gen.items():
    combined_init['m1.' + k] = v
for k, v in init_exp.items():
    combined_init['m2.' + k] = v
torch.save(combined_init, out_dir / 'initial.pt')
print('saved weights.pt and initial.pt successfully')
