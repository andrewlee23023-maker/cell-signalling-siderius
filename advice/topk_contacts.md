1. Scientific motivation: weak-effect positives (injected beta=0.5) are preferentially MISSED when the contact cell
   count is high (median 187 contacts for missed vs 130 for found). The mean ligand over all contacts is diluted
   by background, non-signaling neighbors. Using only the top-K highest-ligand contacts focuses on the most
   evidence-rich contacts and directly addresses this bottleneck.
2. Architecture: build on the iter1 lr_coupling_mlp without the skip connection (< 20k parameters total).
3. Top-K contact selection (fully vectorized, no Python loops):
   - Compute an effective ligand value per contact cell: row 0, zeroed where row 4 == 0 (padding mask).
   - Select the top K=64 contacts by ligand value using torch.topk along the contact dimension.
   - Also keep a top K=64 by receptor value (row 1) for a complementary view.
   - Compute rank partial correlation and masked statistics on the top-K selected contacts only.
   - Also compute the same statistics on the BOTTOM K=64 contacts (lowest ligand) as a "noise" baseline feature.
   - Concatenate top-K stats with bottom-K stats and their difference as features to the MLP.
4. All selection and aggregation must use PyTorch tensor indexing/gather ops — no Python for-loops.
5. Covariates: residualize row 1 against rows 2-3 before any statistic computation.
6. Loss: built-in focal_cw or BCEWithLogitsLoss with pos_weight = 13.7. Custom losses and smooth-L1 are refused.
7. Checkpoint selection: fixed 24 epochs, cosine schedule to 1e-5, keep last completed epoch.
8. Reporting threshold: treat differences under 0.02 lr_auprc as unresolved (noise floor from tile bootstrap).
