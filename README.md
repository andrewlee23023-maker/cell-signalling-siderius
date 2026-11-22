# cell-signalling — Galileo Hackathon 2026 Track 2 (team BioHacking, G2-16)

## Final submission: R7 ensemble (v2)

- `submissions/ensemble-noskip-explore2-v2/` — fixed-weight score ensemble 0.7 m1 + 0.3 m2. Validation
  lr_auprc 0.607 (0.607 ± 0.001 over 3 seeds). Zip `submission_zip/cell-signalling-ensemble-noskip-explore2-v2.zip`,
  SHA-256 `6e0cbef3a543b5fbb0eb0f6aa88c0e84809e1d1ffef662eab898f717c2568f27`.
- `submissions/ensemble-aug-both/` — R8, with m2 also trained on generator data: 0.610 (0.609 ± 0.001 over 3 seeds).
  Packaged and checked, not submitted. SHA-256 `28d94279ebff5a7a88a32b2e76a954ecfe1cf03b938494c69e207fd3b568e73c`.
- `submission/` — earlier upload, R5 (iter1 without skip + generator data): 0.603. SHA-256
  `bc55e302257001ac71bc3c63513f1379d13b141f78f2725fbb8d557dcb08d5f6`.
- `submission_zip/cell-signalling-ensemble-noskip-explore2.zip` — first ensemble upload, which failed: its training
  script did not rebuild its staged weights. Kept for the record.

Every package trains with
`python train/train.py --data_dir $DATA_DIR --seed 20261121 --initial-out initial.pt --out weights.pt`
on the train split only; a clean rerun reproduces the weights byte-identically.

## Other contents

- `report/` — report source; `docs/` — data and methods, decode and reward-hacking records.
- `siderius/` — SIDERIUS configs, workflows, run records and logs; `advice/` — team advice files.
- `tools/`, `results/` — analysis scripts and outputs. `results/loopholes/ensemble_report_stats.json` holds the
  ensemble tile intervals, effect-size breakdown and coupling controls.

Data are not included: Zenodo 10.5281/zenodo.24095146 (CC-BY-4.0). `train/build_dataset.py` in each package is the
task's MIT-licensed generator.
