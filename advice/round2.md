1. Build on the iter1 model (lr_coupling_mlp) without the skip connection, and change one thing at a time relative to it.
2. Loss: use a class-weighted loss instead of smooth-L1 — binary cross-entropy with pos_weight set to the negative-to-positive ratio (about 13.7 on train), as a custom loss, or the built-in class-weighted focal loss (focal_cw). Smooth-L1 treats the labels as numbers to fit, which isn't what a ranking metric rewards.
3. Stopping: train a fixed number of epochs and keep the last completed epoch (checkpoint_selection = last_completed_epoch), so validation is used only for the final comparison.
4. Learning rate and epoch count, within reason, are your choice.
5. Avoid model-based count statistics (the negative-binomial and t-statistic variants), alignment-free features, and anything that uses other samples' inputs.
