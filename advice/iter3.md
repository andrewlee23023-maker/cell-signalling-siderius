1. Ablation before new architecture. Run three variants of iter1 with identical
   training: (a) the partial-correlation z-score alone, no learned weights;
   (b) iter1 with the direct z-score skip removed; (c) iter1 unchanged.
   Report the 3-seed mean ± SD for each.
2. Loss: use BCEWithLogitsLoss with pos_weight = n_negative / n_positive
   (computed on the train split). Do not use SmoothL1Loss on 0/1 labels.
   If dataset IDs are available to the trainer, a pairwise ranking loss over
   positive/negative pairs from the SAME dataset is preferred.
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
