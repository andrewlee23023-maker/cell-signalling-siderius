# Data and methods — cell-signalling (Track 2, team BioHacking, G2-16)

Factual record of the data, models, training, evaluation and controls behind the report. Interpretation is in the
report itself (written by the team).

## Data
- **Source:** task package `cell-signalling` (release `track2-2026-11-21`, SHA-256 verified) and Zenodo record
  10.5281/zenodo.24095146, fetched with the package's `fetch_data.py` (sizes and md5 verified).
- **Content:** real Xenium breast-cancer tile geometry and cell types (Janesick et al. 2023) with 31 re-simulated
  ligand/receptor genes; directed couplings (L, R, A→B) injected with β ∈ {0.5, 1, 2}.
- **Splits used:** train (`train.npz`, 10 tiles × 6 realisations = 60 datasets, 6,828 triples, 463 positives) for
  all fitting; validation (`validation.npz`, 7 tiles × 3 realisations = 21 datasets, 2,214 triples, 151
  positives) for comparison and model choice only. No test data used.
- **Model input:** `[5, 1024]` per triple, built with the task's own `read_npz` + `contact_matrix`: row 0 mean
  log-normalised ligand over type-A neighbours, row 1 receptor of the B cell, row 2 log library minus tile mean,
  row 3 fraction of type-A neighbours, row 4 mask.
- **Generator augmentation (final model):** the task's public generator `build_dataset.py` (md5 3994e987…, MIT),
  run on the 10 **train tiles only** with seed 20261121, 6 new realisations per tile (60 datasets, 474 positives).
  Check: with the public seed it regenerates released dataset `t00_00_r0` exactly (counts, labels, eligibility).

## Evaluation
- **Metric:** official `lr_auprc` (per-dataset average precision, averaged over the 21 validation datasets), plus
  `direction_accuracy` and `lr_auroc`, all computed with the task's `_lr_metrics.py`. Random ranking 0.114 ± 0.014.
- **Baseline reproduction:** official contact Spearman 0.5603341514 (exact).
- **Uncertainty:** 7-tile bootstrap (all realisations of a tile resampled together, 5,000 reps, seed 0); 3 training
  seeds (20261121, 1, 2) for trained models.
- **Health:** official `scores_rank_triples` (blocking) passed for every model reported.

## Models
| Model | Description | Validation lr_auprc |
| --- | --- | --- |
| Baseline | Spearman(row 0, row 1) over contact cells | 0.560 |
| M1 two-part slope t | team statistic; algorithm | 0.515 |
| NB Δdeviance | per-triple negative-binomial GLM on recovered counts | 0.402 |
| (a) partial-corr z-score | iter1's rank partial correlation (rows 2–3 removed), Fisher z; no weights | 0.563 |
| (c) iter1 | SIDERIUS-generated `lr_coupling_mlp` (15.9k params), Claude Sonnet 5.5 + team advice | 0.583 ± 0.004 |
| (b) iter1 without skip | (c) with the partial-correlation skip removed | 0.593 ± 0.003 |
| (b) + contact subsampling | keep U(0.7, 1.0) of contacts per sample per batch | 0.591 ± 0.003 |
| **(b) + generator data (final, uploaded)** | (b) trained on 60 train + 60 generated train-tile datasets | **0.601 ± 0.003** (submission seed 0.6033) |

## Training (final model)
Smooth-L1 loss on the binary target; AdamW lr 0.002, weight decay 0.001; batch 32; cosine schedule over 24 epochs
to 1e-5; fixed 24 epochs (last epoch kept; validation never read). `initial.pt` saved before any optimizer step.
Command: `python train/train.py --data_dir $DATA_DIR --seed 20261121 --initial-out initial.pt --out weights.pt`.
A clean, network-free rerun from `train.npz` alone regenerates the augmentation and reproduces `weights.pt`
byte-identically (684 s, RTX 5060 laptop GPU). Artifact `cell-signalling-iter1-noskip-gen.zip`, SHA-256
`bc55e302257001ac71bc3c63513f1379d13b141f78f2725fbb8d557dcb08d5f6`.

## Controls (final model, validation)
| Control | lr_auprc | direction_accuracy |
| --- | --- | --- |
| Intact | 0.603 | 0.815 |
| Receptor row permuted within each triple's contacts (5 seeds) | 0.120 ± 0.011 | 0.523 |
| Ligand row permuted (5 seeds) | 0.113 ± 0.013 | 0.490 |
| Padding mask disabled | 0.143 | 0.605 |
| Rows 2–3 zeroed | 0.559 | 0.801 |

AP by injected β (positives of one β vs all negatives, per dataset, averaged): baseline 0.207 / 0.510 / 0.642;
final 0.313 / 0.631 / 0.655.

## SIDERIUS runs (workflow mode, single iterations)
Trained models: iter1 0.5875 · explore2 0.592 · idea 3 0.577 · augdata 0.575 · loop2b iter 1 0.564 · loop2c iter 1
0.528 · smoke 0.189. Failures: Claude implement replies failing SIDERIUS's output-format check (3 runs); custom loss
refused (2); multi-iteration hand-off crash (2); dataset-profile binding error (1); classifier output refused on the
first smoke run. Local WSL GPU runs used a GPU-accounting patch (branch `wsl-gpu-accounting`, commit 1f9e28b6);
the submitted model does not depend on SIDERIUS.

## Evaluation-process probes (flagged; validation only; nothing uploaded)
27 reward-hacking leads tested (`reward_hacking_explorations.md`, `reward_hacking_part2_results.md`). Only the
Health-gate evasion was demonstrated (a binary score is refused; with 1e-17 jitter it passes). Selection on the
open validation split inflates the best-of-43 score by +0.044 on held-out tiles.

## Cost
Claude Sonnet 5.5 ≈ $3.35 (plus $1.01 carried over from 2026-11-20 tests); OpenAI ≈ $0.61; Gemini Flash-Lite
tracked in tokens only. Prices: Anthropic model table (2026-11-03); developers.openai.com/api/docs/pricing.
