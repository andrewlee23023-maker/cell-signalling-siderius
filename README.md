# cell-signalling — Galileo Hackathon 2026 Track 2 (team BioHacking, G2-16)

- `submission/` — uploaded model (SHA-256 of `submission_zip/cell-signalling-iter1-noskip-gen.zip`: `bc55e302257001ac71bc3c63513f1379d13b141f78f2725fbb8d557dcb08d5f6`). Train: `python train/train.py --data_dir $DATA_DIR --seed 20261121 --initial-out initial.pt --out weights.pt`.
- `report/` — report source; `docs/data_and_methods.md` — data, models, controls; `docs/` — decode and reward-hacking records.
- `siderius/` — SIDERIUS configs, workflows, run records and logs; `advice/` — team advice files.
- `tools/`, `results/` — analysis scripts and outputs.

Data are not included: Zenodo 10.5281/zenodo.24095146 (CC-BY-4.0). `submission/train/build_dataset.py` is the task's MIT-licensed generator.
