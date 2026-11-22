1. Architecture: soft bivariate density copula + rank association MLP. Keep parameter budget under 18k.
2. Representation (all operations must be fully vectorized using PyTorch tensor ops — no Python for-loops, no chunked iteration over samples or contacts):
   - For each valid contact cell, project (row 0, row 1) onto a 4x4 grid of 2D Gaussian kernels (soft binning). Use broadcasting to compute all 16 Gaussian kernel responses in one batched tensor operation: [B, 1, C] - [1, 16, 1] style expansion, NOT a loop over grid cells.
   - Multiply by the mask (row 4) to zero out padded cells, then sum across the contact dimension and divide by masked contact count. Result: [B, 16] joint density vector.
   - Concatenate the 16 joint density features with rank partial correlation (rows 2-3 residualized out) and high/low tercile contrast of receptor over ligand quartiles.
3. Covariates: residualize row 1 (receptor) against rows 2 and 3 using a closed-form OLS in a single batched operation before computing the density representation.
4. Padding: apply explicit boolean mask from row 4 == 1; masked cells must contribute 0 to all statistics and density bins. Never iterate over contacts in Python — use masked tensor sum/mean.
5. No Python for-loops anywhere in the forward pass. No chunked loops. Implement ALL statistics using batched PyTorch operations (torch.sum, torch.mean, broadcasting, masked_fill, etc.).
6. Loss: built-in focal_cw or BCEWithLogitsLoss with pos_weight = 13.7. Custom losses are refused.
7. Checkpoint selection: fixed 24 epochs, cosine schedule, keep last completed epoch.
