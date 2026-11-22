from pathlib import Path

m1_src = Path('/home/andre/work/cell-signalling/submissions/iter1-noskip-gen/model.py').read_text()
m2_src = Path('/home/andre/work/cell-signalling/project/results/runs/0759a6f3-4564-4865-a9d8-0d39b4c00bdf/workspace/plugin_source_sentinel/lr_contact_contrast_gbm_head.py').read_text()

m1_clean = m1_src.replace('class CellSignalModel', 'class _OriginalCellSignalModel').replace('PLUGIN_MODEL_CLASS = CellSignalModel', '')
m2_clean = m2_src.replace('PLUGIN_MODEL_CLASS = LrContactContrastGbmHead', '').replace('PLUGIN_CONFIG_CLASS = LrContactContrastGbmHeadConfig', '').replace('PLUGIN_OUTPUT_TYPE = "regressor"', '')

ensemble_wrapper = '''
class CellSignalModel(nn.Module):
    def __init__(self, w1: float = 0.70, w2: float = 0.30, **kwargs):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        import json
        from pathlib import Path
        cfg1_path = Path(__file__).parent / 'cfg1.json'
        if cfg1_path.is_file():
            cfg1 = json.loads(cfg1_path.read_text())
            self.m1 = _OriginalCellSignalModel(config=cfg1)
        else:
            self.m1 = _OriginalCellSignalModel()
        cfg2 = LrContactContrastGbmHeadConfig(segmentation_size=1024)
        self.m2 = LrContactContrastGbmHead(cfg2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.m1(x)
        s2 = self.m2(x)
        return self.w1 * s1 + self.w2 * s2

PLUGIN_MODEL_CLASS = CellSignalModel
PLUGIN_OUTPUT_TYPE = "regressor"
'''

full_code = m1_clean + '\n\n' + m2_clean + '\n\n' + ensemble_wrapper
out_path = Path('/home/andre/work/cell-signalling/submissions/ensemble-noskip-explore2/model.py')
out_path.write_text(full_code)

cfg1_src = Path('/home/andre/work/cell-signalling/submissions/iter1-noskip-gen/cfg.json').read_text()
(out_path.parent / 'cfg1.json').write_text(cfg1_src)
print('model.py and cfg1.json written')
