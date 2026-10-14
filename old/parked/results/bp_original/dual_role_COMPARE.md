# Dual-role: none vs emptyfp cls (TEST)

S5 frozen mask + presence head on cls μ. Gate = keep S5 if P≥τ.

| cls init | pick | ne | FP | AUROC |
|----------|------|---:|---:|------:|
| none | plainFloor@0.22 | **0.600** | 0.383 | 0.782 |
| none | @0.30 | 0.546 | 0.266 | 0.782 |
| emptyfp | plainFloor@0.13 | 0.546 | **0.287** | 0.777 |
| emptyfp | bal@0.32 | 0.496 | **0.204** | 0.777 |
| S5 raw | — | 0.664 | 0.790 | — |
| prior none∧P@0.20 | — | 0.533 | 0.266 | — |

none-cls keeps higher ne; emptyfp-cls pushes FP lower but ne drops more.
