1. Build on the best SIDERIUS model design: lr_contact_contrast_gbm_head (from explore2, validation 0.592).
   This is a 23-statistic closed-form extractor (Spearman, Pearson, rank partial correlation, OLS slope & t-stats,
   tercile contrast, neighbor-fraction split, summary moments) + masked DeepSets + 2-layer MLP head.
   Keep all operations vectorized using batched PyTorch ops (no Python for-loops).
2. Data augmentation: train on the 120 datasets in the generator-augmented training set (data_aug).
   To avoid running over the per-invocation timeout, keep model parameters compact (< 20k) and batch size at 64.
3. Loss function: use the built-in class-weighted focal loss (focal_cw) or BCEWithLogitsLoss with pos_weight = 13.7.
   Do not use smooth-L1 on 0/1 targets.
4. Keep the Spearman skip connection (use_skip_spearman=True) and masked DeepSets branch (use_deepset=True).
5. Padding: mask (row 4 == 1) strictly applied before every statistic and pooling operation.
6. Covariates: rows 2 and 3 residualized out; never used as direct evidence.
7. Checkpoint selection: fixed 24 epochs, cosine schedule, keep last completed epoch.
