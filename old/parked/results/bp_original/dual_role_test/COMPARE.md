# Dual role test: skip=recon, plain=presence

No retrain. S5 mask vs none gate variants.

## TEST

| method | role | ne | FP |
|--------|------|---:|---:|
| S5_raw | skip only | 0.664 | 0.790 |
| plain_raw | plain only | 0.579 | 0.380 |
| soft_merge | skip×σ(max plain) | **0.601** | **0.356** |
| C_plainMass>0 → S5 | plain SEG fire → keep S5 | 0.604 | 0.359 |
| C_maxp@0.86 → S5 | plain max-prob thr | 0.594 | 0.332 |
| A_head@0.20 → S5 | PresenceHead(plain μ) → S5 | 0.591 | 0.455 |
| A_head → plain mask | head + plain SEG | 0.533 | 0.266 |

VAL S5 empty-FP (257): head rejects 104, plainMass 125, both 65; still kept by both 93.
