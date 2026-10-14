# Dual-role design (+ empty-FP FT cls line)

## Lines
| line | net | role | train |
|------|-----|------|-------|
| A | S5 `s5_concat` | recon / SEG mask | **frozen** |
| B | plain no-skip | classification features | warm head → FT encoder+decoder |
| head | `LesionPresenceHead(μ_B)` | lesion yes/no | trained |

Infer: `keep S5 if P(head)≥τ else blank`.

## Classification init variants
1. **none** → `dual_role_train/` (running)
2. **empty-FP FT** (`none_emptyfp/seg_last.pt`, TEST FP≈0.16) → `dual_role_train_emptyfp/` (queued after 1)

## Loss
`BCE(head) + λ_plain·DiceCE(plain) + β·KL` — no grad into S5.
