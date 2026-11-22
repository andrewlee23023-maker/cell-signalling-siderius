# Reward-hacking explorations, part 2 — results

**Status: FLAGGED exploration record. Nothing here is, or was submitted as, a scientific solution.**
Results for the leads planned in `cell-signalling-reward_hacking_part2.md` (sections 6–11 there). Same ground
rules as part 1: local, **validation** split, **official metric code** (`task/plugins/_lr_metrics.py`) and
**official Health views** (`task/plugins/_lr_health_views.py`), evaluator read and never modified, no test
data, nothing uploaded. L2 and L3 are **simulations on validation**; the live leaderboard was not probed and
no uploads were consumed. Every tool re-verified the baseline at **0.5603341514** before reporting.

This file records methods, numbers and where the plan's assumptions did not hold. Interpretation, claims and
loophole explanations are for the team to write (README §7).

## Summary

| Lead | Outcome | Key number(s) |
| --- | --- | --- |
| L1 Cross-realisation pooling | Closed; premise does not hold | labels redrawn per realisation; pooling 0.281 |
| L2 Selection on hidden test via quota (simulated) | Weaker than estimated | σ_diff 0.018; optimism +0.036 at k = 10 (< 0.045) |
| L3 Leaderboard as label oracle (simulated) | Structural, not practical at 10 uploads | 0.122 at 10 queries; 0.348 at 1,000; Ladder holds ~0.11 |
| L4 Missing-triple denominator | Closed (benign) | dropping lowest 10% → 0.560334 (unchanged) |
| L5 Group-exclusivity assignment | Closed | best 0.5614 (+0.001, noise) |
| L6 Direction transform sweep | Closed (monotone) | 0.560 → 0.484 over λ ∈ [0, 1] |
| L7 Tie-breaking headroom | Closed; premise does not hold | baseline has 0 tied scores |
| **L8 Health gate vs binary ranking** | **Demonstrated** | binary refused; + jitter 1e-17 passes |
| L9 Stateful plugin, multi-pass | Package evaluator: 1 pass | organizer harness not readable |
| L10 Transductive pseudo-labelling | Closed | 0.5585 (whole batch) < 0.5603 |
| L11 Tile geometry reuse | Closed | no coordinate matches |
| L12 Identity recovery as a feature | Closed | identity 20% recoverable; adding it Δ −0.005 |
| L13 Zero-positive datasets | Not present | 3–14 positives per dataset; metric would NaN-and-skip |

---

## L1 · Cross-realisation agreement and pooling (`tools/realisation_structure.py`)

- Grouping: 7 tiles × 3 realisations (every tile has realisations r0–r2).
- **Labels are redrawn per realisation** (task card: "the injections are redrawn in every realisation"):
  P(same triple injected in another realisation | injected here) = **0.099** vs prevalence 0.068; 81.6% of
  triple groups have identical labels across realisations, almost all of them all-negative.
- Score correlation of the same triple across realisations: neither injected r = 0.047 (1,927 pairs); one
  injected r = −0.398 (272 pairs); both injected r = −0.188 (15 pairs).

| Probe | lr_auprc | 95% CI | direction_accuracy |
| --- | --- | --- | --- |
| Baseline | 0.5603 | [0.490, 0.626] | 0.827 |
| Pooled mean over realisations | 0.2811 | [0.226, 0.341] | 0.707 |
| Agreement, γ = 0.25 / 0.5 / 1 / 2 / 4 | 0.259 / 0.253 / 0.240 / 0.214 / 0.195 | — | 0.714 – 0.640 |

Per tile, baseline → pooled: all 7 tiles drop (0.470–0.715 → 0.238–0.359).

## L2 · Selection on a hidden test through the upload quota — simulation (`tools/selection_hidden.py`)

- σ_diff (SD of paired differences between the 12 best variants of the part-1 §4 family, validation range
  0.519–0.539, 2,000-rep dataset bootstrap): **0.0183** (spread over 66 pairs 0.0032–0.0266). The plan's
  estimate of 0.029 from the M1 interval was too high.
- Best-of-k with the §4 tile-split protocol (selection tiles stand in for the leaderboard):

| k | Simulated optimism [5%, 95%] | Analytic E[max]·σ_diff | Correlated subset (r > 0.9) |
| --- | --- | --- | --- |
| 1 | +0.000 | +0.000 | +0.000 |
| 2 | +0.019 [−0.003, +0.053] | +0.010 | +0.018 |
| 5 | +0.034 [−0.001, +0.071] | +0.021 | +0.032 |
| 10 | +0.036 [+0.002, +0.072] | +0.028 | +0.032 |

- Crossing with |M1 − baseline| = 0.045: **not reached for k ≤ 10** (simulated or analytic).

## L3 · Leaderboard as a label oracle — simulation (`tools/leaderboard_oracle.py`)

Oracle returns only lr_auprc on the 21 validation datasets. Score of the attacker's final ranking
(information extracted), mean ± SD over 10 seeds:

| Queries | Boosting (random vectors) | Boosting + Ladder (η 0.01, rounded) | Hill-climb from baseline | Hill-climb + Ladder |
| --- | --- | --- | --- | --- |
| 1 | 0.113 ± 0.015 | 0.114 ± 0.015 | 0.5603 | 0.5603 |
| 10 | 0.122 ± 0.011 | 0.120 ± 0.013 | 0.5603 | 0.5603 |
| 100 | 0.166 ± 0.021 | 0.122 ± 0.014 | 0.5604 | 0.5603 |
| 200 | 0.196 ± 0.020 | 0.111 ± 0.011 | 0.5606 | 0.5603 |
| 1,000 | 0.348 ± 0.029 | 0.116 ± 0.012 | 0.567 ± 0.009 | 0.567 ± 0.009 |

Random reference 0.114. At 10 queries the boosting attack is within noise of random.

## L4 · Missing-triple denominator (`tools/metric_semantics.py`)

- Code read: positives come from the **full truth**; missing triples are filled below the dataset minimum;
  AP is computed on the full vector.
- Dropping the lowest 10% / 25% / 50% of triples per dataset: 0.560334 / 0.560053 / 0.557512 (full 0.560334).

## L5 · Global assignment under group exclusivity (`tools/group_assignment.py`)

- Injected count per dataset is **not constant**: 3–14 (no second construction leak).

| Variant (≤ 1 per group; k̂ = round(0.065 n)) | lr_auprc | Selected | Precision | Group acc. (L,A,B) | Group acc. (R,A,B) | Health |
| --- | --- | --- | --- | --- | --- | --- |
| (L,A,B) | 0.5529 | 1,127 | 0.121 | 63/68 | 6/21 | pass |
| (L,A,B) + k̂ | 0.5566 | 144 | 0.472 | 37/68 | 5/21 | pass |
| (R,A,B) | 0.5614 | 1,260 | 0.113 | 25/68 | 16/21 | pass |
| (R,A,B) + k̂ | 0.5603 | 144 | 0.472 | 33/68 | 5/21 | pass |
| Joint (bipartite matching) | 0.5502 | 1,055 | 0.126 | 62/68 | 13/21 | pass |
| Joint + k̂ (greedy truncation) | 0.5559 | 144 | 0.465 | 36/68 | 4/21 | pass |

Best +0.001 over the baseline, inside the baseline's bootstrap interval [0.490, 0.626].

## L6 · Calibrated direction transform (`tools/direction_sweep.py`)

- Family A, s' = s + λ(s − s_rev): monotone decline 0.5603 (λ = 0) → 0.5211 (0.5) → 0.4844 (1.0).
  Note: λ = 1 is 2s − s_rev; the pure difference (0.366, part 1 #1) is the λ → ∞ limit.
- Family B, (1 − w)·rank(s) + w·rank(σ((s − s_rev)/τ)): 0.560 (w = 0) → 0.366 (w = 1), identical for every τ
  (ranking a sigmoid of the same difference does not depend on τ).
- Tile-split holdout gain of the λ selected on selection tiles vs λ = 0: −0.0039 ± 0.0062.

## L7 · Tie-breaking headroom (`tools/tie_headroom.py`)

- The baseline score vector has **0 tied groups** (the 48% zeros produce ties inside each correlation, not
  between triples' scores).
- Grouped (official), best-case, worst-case and random tie-breaking (200 draws) all give 0.560334; breaking
  ties by contact count changes nothing.

## L8 · Health gate vs an ordinally binary ranking (`tools/gate_evasion.py`) — demonstrated

Two-block ranking from L5's best selection mask (1 = selected, 0 = not), plus uniform jitter U(0, ε):

| ε | lr_auprc | Distinct deciles | Blocking gate | Dominant fraction | Recording check |
| --- | --- | --- | --- | --- | --- |
| 0 (pure binary) | 0.117 | 4 | **refused** | 0.431 | flag |
| 1e-17 | 0.118 | 6 | **pass** | 0.413 | flag |
| 1e-15 | 0.131 | 10 | pass | 0.141 | pass |
| 1e-12 … 1e-1 | 0.149 | 10 | pass | 0.103 | pass |

At ε = 1e-17 the jitter cannot change values at 1.0 in float64; it only breaks ties among the zeros, at random.
The gate counts occupied deciles; it does not test whether the score carries more than binary information.
AP stays at random level, so this evades the check without gaining score.

## L9 · Stateful plugin across calls (`tools/tier3_checks.py`)

The package's evaluator (`baseline/run_baseline.py`, `predict_all`) calls `predict(x)` once per triple in a
single pass. The organizers' scoring harness is not in the package, so its pass count cannot be read.

## L10 · Transductive pseudo-labelling inside `predict_batch` (`tools/tier3_checks.py`)

Top 6.5% of the batch by baseline score as pseudo-positives, bottom 50% as pseudo-negatives, logistic
regression on 6 per-triple statistics, re-score:

| Batch | lr_auprc |
| --- | --- |
| Baseline (no pseudo-labelling) | 0.5603 |
| Whole validation set | 0.5585 |
| Random halves | 0.5547 |
| Random quarters | 0.5506 |

## L11 · Train ↔ validation tile geometry (`tools/tier3_checks.py`)

No exact coordinate-layout match between train and validation tiles (hash of coordinates rounded to 1 µm,
relative to each tile's origin). Pairwise-distance histogram L1 distance, validation → nearest train tile:
0.0143–0.2445; closest pair within train: 0.0116.

## L12 · Identity recovery as an additive feature (`tools/tier3_checks.py`)

- (A, B) pair predicted from rows 2–3 summaries (logistic regression on train): validation accuracy 0.203
  (majority pair 0.073; 36 pairs).
- Baseline + predicted pair's train injection rate (combiner fitted on train): 0.5555 vs 0.5603 (Δ −0.0048).

## L13 · Zero-positive datasets (`tools/metric_semantics.py`)

Every validation dataset has 3–14 positives; none has 0 or all. Code read: such a dataset would get NaN and
be skipped by `aggregate()`.

## Reproduce

From `~/work/cell-signalling` in WSL:

```bash
bash tools/run.sh metric_semantics.py realisation_structure.py direction_sweep.py \
  tie_headroom.py group_assignment.py gate_evasion.py selection_hidden.py leaderboard_oracle.py tier3_checks.py
```

Outputs: `runs/loopholes/{metric_semantics,realisation,direction_sweep,ties,assignment,gate,selection_hidden,oracle,tier3}.json`.
Fixed seeds: bootstrap 0, split sampling 0, tie-breaking 0, jitter 0, oracle seeds 0–9.
