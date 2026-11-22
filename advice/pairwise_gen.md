1. Build on the iter1 lr_coupling_mlp architecture without the skip connection (< 20k parameters), trained on the
   AUGMENTED dataset (120 train datasets = 60 original train tiles + 60 generator realisations of train tiles).
   This is the same augmented data that produced the submitted best score (0.601).
2. Loss: replace smooth-L1 with BCEWithLogitsLoss using pos_weight = negative-to-positive ratio computed from the
   full augmented train split (approximately 13.7, recalculate from the actual augmented set). This is a pointwise
   classification loss that better aligns with the ranking metric lr_auprc.
   Alternatively use the built-in class-weighted focal loss (focal_cw). Do NOT use smooth-L1 or custom losses.
3. Features: rank partial correlation of row 1 (receptor) with row 0 (ligand) over valid contacts, with rows 2-3
   (log library offset and sender fraction) residualized out. Masked DeepSets mean and max pooling over valid contacts.
4. Padding: apply explicit boolean mask from row 4 == 1 before ALL operations — correlations, pooling, statistics.
   Padded cells must have zero contribution.
5. Covariates: rows 2 and 3 are nuisance; never use them as direct evidence of signalling.
6. Checkpoint selection: fixed 24 epochs, cosine schedule to 1e-5, keep the last completed epoch.
7. No Python for-loops in the forward pass. All operations must be batched PyTorch tensor ops.
