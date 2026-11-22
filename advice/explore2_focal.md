1. Use the lr_contact_contrast_gbm_head architecture (the best SIDERIUS architecture from explore2, validation 0.592).
   This is a 23-statistic feature extractor + masked DeepSets + 2-layer MLP head. The forward pass is fully
   vectorized — no Python for-loops. All statistics are computed with batched PyTorch tensor ops.
2. Statistics (must include all of these):
   - Spearman and Pearson correlation of R with L over masked contacts.
   - Rank partial correlation and raw partial correlation of R with L, residualizing rows 2 (log library) and 3 (neighbor fraction).
   - OLS slope and t-statistic for both raw and rank versions.
   - Tercile contrast: mean receptor R in top-third vs bottom-third ligand L contacts.
   - Neighbor-fraction split: partial correlation in low-nfrac vs high-nfrac contacts.
   - Summary moments: mean L, std L, mean R, std R, fraction-at-min L/R, mean nfrac, log(n_contacts).
   - Derived: (r_spearman - r_partial_spearman) and (r_partial - r_pearson) as correction indicators.
   - Shrinkage by n / (n + shrink_n0) applied to all statistics before the head.
3. CHANGE from explore2: loss function. Replace smooth-L1 with the built-in class-weighted focal loss (focal_cw).
   Smooth-L1 on 0/1 targets does not align with the ranking metric lr_auprc. focal_cw adjusts for the class
   imbalance (13.7:1 negative:positive ratio) and focuses training on hard examples.
4. Keep the Spearman skip connection (use_skip_spearman=True) and masked DeepSets branch (use_deepset=True).
5. Checkpoint selection: last_completed_epoch. Fixed epochs in [24, 32], cosine schedule.
6. No Python for-loops in the forward pass. Padding: apply mask (row 4 == 1) before every statistic.
7. Covariates: rows 2 and 3 are nuisance; residualize them out, never use as direct evidence.
