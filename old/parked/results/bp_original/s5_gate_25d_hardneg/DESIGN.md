# 2.5D hard-negative retrain

- Freeze S5; label TRAIN empty_fp = GT empty & S5 nonempty
- Weighted sampler 0.45/0.4/0.15
- Extra loss weight ×2 on empty_fp batches
- Early-stop: AUROC(nonempty vs empty_fp)
- Fuse with dual μ-head (product / weighted)
