# Reward-hacking explorations — cell-signalling (Track 2)

**Status: FLAGGED exploration record. None of these probes is, or was submitted as, a scientific solution.**
All probes ran locally on the **validation** split with the **official metric code**
(`task/plugins/_lr_metrics.py`: per-dataset average precision, averaged over the 21 validation datasets).
The evaluator was read, never modified. No test data were used. Nothing was uploaded.

This file records methods and numbers only. Interpretation, claims and the loophole explanation are for the
team to write (README §7: the team writes loophole explanations).

Reference points (task card / official baseline, reproduced exactly):

| Reference | lr_auprc | direction_accuracy |
| --- | --- | --- |
| Random ranking (200 repeats, task card) | 0.114 ± 0.014 | 0.50 ± 0.04 |
| Official baseline, contact Spearman (reproduced: 0.5603341514) | 0.560 | 0.827 |

---

## 1. Leads closed by the decode diagnostics (`decode_report.md`)

| Lead | Probe | Result (lr_auprc) |
| --- | --- | --- |
| #2 Padding / contact count | number of contact cells as the score | 0.095 / 0.096 (as is / negated) |
| #5 Confounders without signalling | 13 alignment-free per-triple features (means, SDs, non-zero fractions of rows 0–3; contact count) | 0.073–0.120 |
| #8 File-order leak | position of the triple within its dataset | 0.121 / 0.102 |
| #9 Simulator artifacts | gradient boosting (depth 2, 200 trees) on the 13 features, fitted on **train** only | 0.090 (direction 0.425) |
| #4 Tie handling | read from the metric code: tied scores are grouped (scikit-learn convention) | not exploitable by design |
| #10 Odd values | read from the metric code: non-finite scores raise; a missing triple ranks below every prediction of its dataset | not exploitable by design |
| Health check | blocking: ≥ 5 occupied within-dataset deciles | refuses constant / near-binary scores |

Within-dataset normalisation of any kind cannot change the metric: AP is computed per dataset, so any
monotone transform applied within a dataset leaves the score unchanged.

## 2. Batch / cross-sample probes (`tools/loophole_probes.py`, `tools/loophole_probes2.py`, `tools/loophole_probes3.py`)

All of these use **other samples' inputs** (information a per-sample `predict(x)` never sees).

| Lead | Probe | lr_auprc | direction_accuracy |
| --- | --- | --- | --- |
| #1 Pair-relative score | baseline s(A→B) − s(B→A) | 0.366 | 0.827 |
| #3 Mirror suppression | lower-scoring direction of each pair pushed below every winner | 0.543 | 0.827 |
| #7 Direction-blind scoring | both directions get the pair maximum | 0.347 | 0.500 |
| #6 Identity priors | train injection frequency of (L,R,A,B) / (L,R) / (A,B) | 0.115 / 0.127 / 0.136 | 0.52 / 0.56 / 0.57 |
| E Cross-realisation differencing | s_i − mean(s of the same triple in the tile's other realisations) | 0.438 | 0.783 |
| Q Within-receptor calibration | z-score within the dataset's triples sharing (R, B) | 0.238 | 0.691 |
| V Sibling competition (same R, A, B; other ligand) | s_i − max(sibling scores) | 0.538 | 0.822 |
| W Sibling competition (same L, A, B; other receptor) | s_i − max(sibling scores) | 0.442 | 0.787 |

Supporting facts measured during these probes:

- Reverse triple present for 100% of triples; pairs with both directions injected: 6.
- Same non-injected triple across realisations: baseline score correlation r = 0.047 over 1,927 pairs
  (the spatial null is redrawn per realisation).
- Siblings: 390 / 2,214 triples share (R, A, B) with another ligand (score correlation r = 0.098); in 21 such
  groups with exactly one positive, the positive scores highest in 17 (81%). 1,056 / 2,214 share (L, A, B)
  with another receptor (r = 0.156); in 68 groups with one positive, it scores highest in 63 (93%).

## 3. Not tested (by design)

- **F · Reading the validation truth at scoring time.** The validation labels are in the data folder. The task
  card already states that direct lookup "is possible" and that only rules and submission audits prevent it,
  so this is disclosed, not a discovery. Attempting it would be cheating, not a demonstration.
- **Regenerating validation truth from the public generator and seed (20261109).** Redundant: the validation
  truth is already open. The test seed is secret.
- **Uploaded weights not reproduced by `train_command`.** A review-process gap, not a metric loophole;
  submitting it would be a misleading entry.

## 4. Evaluation-process probe: selection on the open validation split (`tools/selection_optimism.py`)

Method:
- 43 distinct fixed statistics (no fitting): 12 association types (Spearman; Pearson; partial Pearson,
  slope t, detection t, intensity t and two-part t, each with and without rows 2–3 as covariates) × 4 ligand
  transforms (raw, z-clip ±2, z-clip ±4, rank); identical score vectors removed.
- 70 splits of the 7 validation tiles into 3 or 4 selection tiles and the remaining holdout tiles.
- For each split, the best variant (mean per-dataset AP on the selection tiles) among N randomly drawn
  variants is scored on the holdout tiles. Optimism = selection score − holdout score.

| N variants searched | Score on selection tiles | Score on holdout tiles | Optimism (mean ± SD) | Baseline, no selection |
| --- | --- | --- | --- | --- |
| 1 | 0.475 | 0.475 | +0.000 ± 0.096 | 0.000 |
| 5 | 0.532 | 0.509 | +0.023 ± 0.100 | 0.000 |
| 10 | 0.541 | 0.505 | +0.036 ± 0.096 | 0.000 |
| 25 | 0.547 | 0.504 | +0.043 ± 0.093 | 0.000 |
| 43 | 0.551 | 0.507 | +0.044 ± 0.090 | 0.000 |

Best variant on all validation vs train (independent tiles; nothing fitted):

| Statistic | Validation lr_auprc | Train lr_auprc |
| --- | --- | --- |
| Best on validation: slope t with rows 2–3 as covariates, ligand z-clipped ±2 | 0.539 | 0.548 |
| Official baseline (contact Spearman) | 0.560 | 0.529 |

Caveat: the 43-variant family does not contain the baseline exactly (its rank variants break ties
ordinally; the baseline averages tied ranks, and row 1 is 48% zeros). The optimism measurement does not
depend on this.

Related measurement (M1 comparison, `runs/m1/compare.json`): the team's two-part slope-t statistic scores
0.515 on validation (Δ vs baseline −0.045, 95% dataset-bootstrap CI [−0.102, +0.011]) and 0.556 on train
(baseline 0.529).

## 5. Reproduce

From `~/work/cell-signalling` in WSL, with `PY=~/SIDERIUS/.venv/bin/python`:

```bash
$PY tools/cs_data.py              # builds tensors with the task's own code; checks baseline = 0.5603341514
$PY tools/cs_decode.py            # section 1  -> decode_report.md / .json
$PY tools/loophole_probes.py      # #3, #7, #6 -> runs/loopholes/probes.json
$PY tools/loophole_probes2.py     # E, Q       -> runs/loopholes/probes2.json
$PY tools/loophole_probes3.py     # V, W       -> runs/loopholes/probes3.json
$PY tools/selection_optimism.py   # section 4  -> runs/loopholes/selection_optimism.json
$PY tools/m1_compare.py           # M1 table   -> runs/m1/compare.json
```

Everything is deterministic (fixed seeds where randomness is used: bootstrap seed 0, split sampling seed 0).
