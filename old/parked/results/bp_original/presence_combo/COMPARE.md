# Presence gate combo (case-by-case)

Frozen SEG `none` + gates only (no retrain). Keep if present; blank if rejected.

| gate | source |
|------|--------|
| C0 | raw SEG (no gate) |
| C1 | top-k50 μ-logit mean ≥ 0.94936 |
| C2 | LesionPresenceHead τ=0.11 |
| P_old | PresenceHead τ=0.20 |

## TEST (decision rules fixed on VAL)

| method | ne Dice | empty FP | sens | vs C0 |
|--------|---------|----------|------|-------|
| C0 | **0.579** | 0.380 | 0.826 | baseline |
| AND(C2_new) | 0.576 | 0.353 | 0.817 | weak FP cut |
| MAJ2of3 | 0.575 | 0.317 | 0.804 | mild FP, ne~kept |
| AND(C1_topk) | 0.561 | 0.275 | 0.743 | top-k alone |
| AND(C1+C2) | 0.559 | 0.263 | 0.739 | best AND without P |
| AND(P_old) | 0.533 | 0.266 | 0.743 | old head still strong |
| AND(C2+P) | 0.533 | 0.251 | 0.743 | small gain over P |
| AND(C1+P) | 0.517 | **0.216** | 0.678 | aggressive |
| AND(C1+C2+P) / MAJ3 | 0.517 | **0.207** | 0.678 | lowest FP |
| OR(*) | ~0.58 | ~0.37 | ~0.82 | ≈ C0 (useless) |
| SOFT_mean@0.40 (VAL-constrained) | 0.571 | 0.344 | — | near C0, mild |

## VAL empty-FP: who rejects what (among FT)

| reject set | count |
|------------|------:|
| P_only | 23 |
| C1_only | 11 |
| C2_only | 4 |
| C1∩P | 10 |
| C2∩P | 7 |
| C1∩C2 | 0 |
| all3 | 0 |
| none (still FP) | 79 |

→ P_old carries most unique FP kills; C1 adds some; C2 barely unique. Triple AND ≈ C1∧P (C2 redundant under AND).

## Takeaways

1. **OR / soft-union ≈ C0** — cannot recover empty FP without hurting little.
2. **AND stacks FP↓ and ne↓** — Pareto: MAJ2 or C1 alone / C1∧C2 for milder; C1∧P or all3 for lowest FP.
3. **Old P still dominates new C2** for empty FP; C2 alone is weak.
4. No combo beats C0 nonempty Dice while also beating P_old on FP.
