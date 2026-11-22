1. Architecture: build on the iter1 lr_coupling_mlp without the skip connection (keep parameter count under 20k).
2. Multi-quantile statistics: in addition to full-contact rank partial correlation, compute:
   - Rank partial correlation restricted to upper-quartile ligand contacts (row 0 >= 75th percentile of non-zero contacts).
   - High-vs-low ligand contrast of receptor expression (mean R in top tercile minus bottom tercile of L over valid contacts).
   - Fraction of non-zero contacts where both ligand and receptor exceed their respective median non-zero values.
3. Covariates: rows 2 (log library offset) and 3 (fraction of sender neighbours) are nuisance covariates. They must strictly be partialled out or conditioned on, never fed as direct evidence of signalling.
4. Padding: compute every quantile, percentile, and statistic strictly over non-padded cells (row 4 == 1). Build an explicit boolean mask and apply it before any pooling or statistic calculation.
5. Loss: use the built-in class-weighted focal loss (focal_cw) or BCEWithLogitsLoss with pos_weight = 13.7. Custom losses and smooth-L1 are refused.
6. Model output: single continuous score (regressor) with at least 5 occupied deciles to satisfy the Health gate.
7. Stopping & evaluation: fixed 24 epochs, cosine schedule; keep the last completed epoch (checkpoint_selection = last_completed_epoch). Validation split is for final comparison only.
