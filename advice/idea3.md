1. Build on the lr_contact_contrast_gbm_head design from the previous run (validation 0.592): closed-form
   dependence statistics between R and the neighbour-ligand signal over the masked contact cells (Spearman,
   Pearson, partial correlation after regressing out rows 2-3, OLS slope, a high- vs low-ligand tercile
   contrast of R, a neighbour-count split), concatenated with a small masked DeepSets embedding and a
   2-layer MLP head. Change one thing at a time relative to it.
2. Loss: use the built-in class-weighted focal loss (focal_cw); custom losses are refused
   for this task. Do not use SmoothL1Loss on 0/1 labels.
3. Covariates: rows 2 and 3 (x[:, 2:4, :]) are nuisance covariates. They must be
   partialled out or conditioned on, never used as direct evidence of signalling.
4. Padding: compute every statistic and every pooling operation over non-padded
   cells only; build an explicit mask from the padding and apply it before
   mean/max pooling.
5. Statistics: prefer rank-based association (Spearman / rank partial correlation).
   Do not use negative-binomial or t-statistic association scores; both
   underperformed the baseline on validation.
6. Model size: do not increase parameter count. Keep features fixed if width is
   varied.
7. Selection: choose epochs and hyperparameters on a tile-grouped split of the
   TRAIN data. Validation is for final comparison only.
8. Reporting: treat any difference under 0.02 lr_auprc as unresolved. Report
   uncertainty with a bootstrap that resamples the 7 tiles, not the 21 datasets.
