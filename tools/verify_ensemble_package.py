import sys, importlib.util, json, numpy as np, torch
sys.path.insert(0, '/home/andre/work/cell-signalling/tools')
from lh_common import W, base, boot_ci, data, per_ds, verify

verify()
va = data('validation')
pb = per_ds(va, base('validation'))
X_val = torch.from_numpy(va['X']).float()

sub_dir = W / (sys.argv[1] if len(sys.argv) > 1 else "submissions/ensemble-noskip-explore2")
spec = importlib.util.spec_from_file_location('ens_model', sub_dir / 'model.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = json.loads((sub_dir / 'cfg.json').read_text())
model = mod.CellSignalModel(**cfg)
sd = torch.load(sub_dir / 'weights.pt', map_location='cpu', weights_only=True)
model.load_state_dict(sd)
model.eval()

with torch.no_grad():
    preds = []
    for i in range(0, len(X_val), 256):
        preds.append(model(X_val[i:i+256]).numpy().ravel())
    s = np.concatenate(preds)

ps = per_ds(va, s)
d = np.array([ps[k] - pb[k] for k in pb])
m, lo, hi = boot_ci(np.array(list(ps.values())))
dm, dlo, dhi = boot_ci(d)
print(f'Ensemble Package Verification: lr_auprc {m:.4f} [{lo:.3f}, {hi:.3f}] | vs base: {dm:+.4f} [{dlo:+.3f}, {dhi:+.3f}] | >base: {int((d>0).sum())}/{len(d)}')
