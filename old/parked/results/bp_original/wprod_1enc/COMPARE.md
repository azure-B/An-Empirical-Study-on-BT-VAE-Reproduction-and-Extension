# wprod 1-encoder (S5 μ × ctx μ) + min-mass

VAL AUROC: μ=0.7195 ctx=0.6930 wprod=0.7144

| system | ne | empty FP | note |
|---|---:|---:|---|
| messy wprod+mass | 0.601 | 0.305 | S5+dual+ResNet25d |
| **1enc wprod+mass (ne≥0.60 pick)** | **0.563** | **0.350** | τ=0.01 m≥1280 |
| 1enc wprod+mass (ne≥0.57 pick) | 0.563 | 0.350 | τ=0.18 m≥1280 |
| 1enc wprod-only τ0.01 | 0.664 | 0.790 | no mass |

Encoder count: **1** (frozen S5). Heads only for presence.
