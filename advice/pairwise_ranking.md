1. Architecture: keep the lightweight iter1 lr_coupling_mlp feature extractor (< 20k parameters), taking [5, 1024] input.
2. Objective: replace pointwise regression with a pairwise ranking objective aligned with lr_auprc:
   - For every positive sample (y = 1) in a batch, form pairs with negative samples (y = 0) from the same dataset/tile.
   - Use MarginRankingLoss (margin = 0.5) on (score_pos - score_neg).
   - If intra-dataset batching is unavailable to the trainer, use BCEWithLogitsLoss with pos_weight = 13.7.
3. Features: extract rank partial correlation (covariates in rows 2-3 residualized out), Spearman rank correlation, and masked DeepSets mean and max embeddings over non-padded cells.
4. Padding: apply explicit boolean mask from row 4 before any statistic calculation or pooling. Padded cells must have zero contribution.
5. Model size: do not increase parameter count. Keep width moderate (e.g., hidden dims 64-128).
6. Selection: train fixed 24 epochs; last completed epoch is kept. No validation reading during training.
