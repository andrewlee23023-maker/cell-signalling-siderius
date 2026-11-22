1. Build on the iter1 model (lr_coupling_mlp) without the skip connection, and change one thing at a time relative to it.
2. Loss: use the built-in class-weighted focal loss (focal_cw) instead of smooth-L1; custom losses are refused for this task. Smooth-L1 treats the labels as numbers to fit, which isn't what a ranking metric rewards.
3. Stopping: train a fixed number of epochs and keep the last completed epoch (checkpoint_selection = last_completed_epoch), so validation is used only for the final comparison.
4. Learning rate and epoch count, within reason, are your choice.
5. Avoid model-based count statistics (the negative-binomial and t-statistic variants), alignment-free features, and anything that uses other samples' inputs.
