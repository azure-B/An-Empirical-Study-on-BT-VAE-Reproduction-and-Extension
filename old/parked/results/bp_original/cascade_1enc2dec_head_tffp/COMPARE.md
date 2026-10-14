# 1enc·2dec + cls Head (TF/FP-aware)

VAL AUROC=0.7396 · raw SEG ne=0.716

| system | ne | empty FP | TF | FT | note |
|---|---:|---:|---:|---:|---|
| messy wprod+mass | 0.601 | 0.305 | — | — | 3 backbones |
| **1enc2dec head+mass (ne≥0.60)** | **0.650** | **0.485** | 33 | 162 | τ=0.01 m≥1280 |
| 1enc2dec head+mass (ne≥0.57) | 0.590 | 0.380 | 51 | 127 | τ=0.09 m≥640 |

weights: w_FT=3.0 w_TF=2.5 · seconds=2258
