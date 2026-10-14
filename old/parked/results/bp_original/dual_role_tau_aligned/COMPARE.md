# Dual-role: τ from classification → S5 inference

τ chosen on VAL **classification** (F1 / Youden / Acc). Inference = keep S5 iff P≥τ.

## TEST

| run | τ rule | τ | ne | empty FP | sens | cls F1 |
|-----|--------|--:|---:|---------:|-----:|-------:|
| none | S5 raw | — | 0.664 | 0.790 | 0.974 | — |
| none | cls-f1 | 0.34 | 0.523 | 0.219 | 0.678 | 0.667 |
| none | cls-youden | 0.56 | 0.395 | 0.126 | 0.500 | 0.593 |
| none | cls-acc | 0.56 | 0.395 | 0.126 | 0.500 | 0.593 |

| emptyfp | S5 raw | — | 0.664 | 0.790 | 0.974 | — |
| emptyfp | cls-f1 | 0.16 | 0.527 | 0.243 | 0.683 | 0.668 |
| emptyfp | cls-youden | 0.16 | 0.527 | 0.243 | 0.683 | 0.668 |
| emptyfp | cls-acc | 0.33 | 0.489 | 0.186 | 0.622 | 0.659 |
