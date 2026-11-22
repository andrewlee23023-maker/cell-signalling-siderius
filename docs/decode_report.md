# cell-signalling decode report (validation, official metric)

Random ranking: lr_auprc 0.114 ± 0.014, direction_accuracy 0.50 ± 0.04 (task card).

## 1 · Channels over contact cells (validation)

| row | meaning (task card) | zero frac | p50 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| 0 | mean log-norm L over type-A neighbours | 0.368 | 2.75 | 6.99 | 9.25 |
| 1 | log-norm R of the B cell | 0.482 | 3.71 | 10.1 | 19 |
| 2 | log library − tile mean | 0.000 | -0.0767 | 1.41 | 2 |
| 3 | # type-A neighbours / 6 | 0.000 | 0.167 | 0.833 | 1 |

## 2 · Official baseline (contact Spearman)

lr_auprc 0.5603, direction_accuracy 0.8266, lr_auroc 0.8871

Positives found in their dataset's top-k (k = number of positives), by effect size and contact count:

| β | positives | found in top-k | median contacts (found / missed) |
| --- | --- | --- | --- |
| 0.5 | 55 | 11 (20%) | 130 / 187 |
| 1.0 | 43 | 21 (49%) | 120 / 197 |
| 2.0 | 53 | 39 (74%) | 147 / 144 |

## 3 · Shortcuts that ignore L–R alignment (loophole leads #2 padding/contacts, #5 confounders)

Each scored as is and negated; anything clearly above random would be a confounder the metric rewards.

| feature | lr_auprc as is | negated |
| --- | --- | --- |
| rec_frac_nonzero | 0.097 | 0.120 |
| rec_mean | 0.099 | 0.113 |
| lig_std | 0.109 | 0.100 |
| rec_std | 0.099 | 0.107 |
| lig_frac_nonzero | 0.096 | 0.106 |
| nA_mean | 0.101 | 0.096 |
| loglib_mean | 0.089 | 0.101 |
| nA_std | 0.100 | 0.099 |
| lig_mean | 0.096 | 0.099 |
| n_contacts | 0.095 | 0.096 |
| loglib_std | 0.092 | 0.092 |
| loglib_frac_nonzero | 0.073 | 0.073 |
| nA_frac_nonzero | 0.073 | 0.073 |

## 4 · File-order leak (lead #8)

Position of the triple within its dataset as the score: lr_auprc 0.121 (as is) / 0.102 (negated).

## 5 · Simulator-artifact probe (lead #9)

Gradient boosting (depth 2, 200 trees) on the 13 alignment-free features above, fitted on train only: lr_auprc 0.090, direction_accuracy 0.425. Near random means no exploitable artifact in the marginals.

## 6 · Reverse-pair structure (leads #1 batch-relative, #3 mirror pairs, #7 direction)

- Reverse triple present for 100.0% of triples; pairs where both directions are injected: 6.
- Pair-relative score s(A→B) − s(B→A) from the baseline: lr_auprc 0.366, direction_accuracy 0.827 (baseline 0.560 / 0.827).
- This needs the reverse sample's tensor, which a per-sample `predict(x)` never sees; only `predict_batch` over a
  batch containing both could use it. Whether that is legitimate cross-sample pooling or a batch-composition
  loophole is a team judgement (brief §8.1); the evaluator's batching is not documented in the package.

## 7 · Evaluator behaviour (read from task/plugins/_lr_metrics.py; leads #4, #10)

- Ties: average precision groups tied scores (scikit-learn convention), so ties are not credited optimistically.
- Non-finite scores raise; a missing triple ranks below every prediction of its dataset.
- Health (blocking): at least 5 occupied within-dataset score deciles, so constant/near-binary scores are refused.
- Metrics are per dataset then averaged with equal weight: small datasets count as much as large ones.
