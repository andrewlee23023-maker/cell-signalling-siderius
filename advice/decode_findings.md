1. The baseline misses weak effects: 74% of the 53 β = 2 positives are found in the top-k, compared with 20% of the 55 β = 0.5 positives, and the missed weak positives have more contact cells (median 187 vs 130 for the ones found). A better model could weight the training loss by β, upweighting weak-effect positives so it learns them better.
2. Keep the relationship between rows 0 and 1 (ligand and receptor), and use rows 2 and 3 (library offset and fraction of sender neighbours) as covariates.
3. Handle padding properly, and make sure the padding does not affect predictions.
4. Normalize for the smooth spatial fields and nuisance perturbations. Do not use batch-relative scoring.
5. Keep the model size and training time small.
6. Nothing to avoid at the moment.
