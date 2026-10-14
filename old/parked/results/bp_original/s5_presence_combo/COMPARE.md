# S5 × presence combo

Hard reject. Hybrid = none μ gates + S5 mask. Native = S5 μ gates (topk recal on VAL) + S5 mask.
Artifacts: `val_cases_*.csv`, `test_combos_*.csv`, `report.json`.

## TEST refs
- none C0: ne=0.579 fp=0.380
- s5 C0: ne=0.664 fp=0.790

## Verdict (TEST)

| pick | ne | fp | note |
|------|---:|---:|------|
| none C0 | 0.579 | 0.380 | baseline |
| none ∧ P_old | 0.533 | 0.266 | prior best FP |
| **hybrid MAJ2** | **0.636** | 0.488 | +ne vs none, FP≪S5 |
| hybrid P_old | 0.591 | 0.455 | +ne vs none∧P |
| hybrid Soft_mean@0.14 | 0.653 | 0.596 | almost S5 ne |
| hybrid ∧ C1 | 0.565 | 0.269 | ≈ none∧C1 (C1 kills S5 boost) |
| native topk@0.82 | 0.659 | 0.638 | ne kept, FP still high |

C1(none) keep-set ≈ none-gated cases → under AND(C1*) S5 barely helps ne.
Without hard C1 (MAJ2 / P / soft), S5 nonempty Dice carries through.
Native: none-trained P/C2 on S5 μ poorly calibrated (FP stays high).

## hybrid_noneGate_s5Seg (topk=0.9494)
| method | ne Dice | empty FP | sens |
|--------|--------:|---------:|-----:|
| C0 | 0.6637 | 0.790 | 0.974 |
| AND(C1_topk) | 0.5649 | 0.269 | 0.735 |
| AND(C2_new) | 0.6545 | 0.704 | 0.952 |
| AND(P_old) | 0.5909 | 0.455 | 0.857 |
| AND(C1_topk+C2_new) | 0.5624 | 0.257 | 0.730 |
| AND(C1_topk+P_old) | 0.5169 | 0.216 | 0.670 |
| AND(C2_new+P_old) | 0.5909 | 0.428 | 0.852 |
| AND(C1_topk+C2_new+P_old) | 0.5169 | 0.207 | 0.670 |
| MAJ2of3 | 0.6364 | 0.488 | 0.913 |

VAL empty-FP reject (hybrid): C1_only=69, C1∩P=57, P_only=23, all3=17, none_reject=78

## native_s5Gate_s5Seg (topk=0.82 VAL-recal)
| method | ne Dice | empty FP | sens |
|--------|--------:|---------:|-----:|
| C0 | 0.6637 | 0.790 | 0.974 |
| AND(C1_topk) | 0.6591 | 0.638 | 0.917 |
| AND(C2_new) | 0.6460 | 0.680 | 0.948 |
| AND(P_old) | 0.5120 | 0.545 | 0.757 |
| AND(C1_topk+C2_new) | 0.6414 | 0.563 | 0.891 |
| AND(C1_topk+P_old) | 0.5109 | 0.452 | 0.713 |
| AND(C2_new+P_old) | 0.5037 | 0.506 | 0.743 |
| AND(C1_topk+C2_new+P_old) | 0.5025 | 0.422 | 0.700 |
| MAJ2of3 | 0.6509 | 0.677 | 0.948 |

## control_noneGate_noneSeg (topk=0.9494)
| method | ne Dice | empty FP | sens |
|--------|--------:|---------:|-----:|
| C0 | 0.5792 | 0.380 | 0.826 |
| AND(C1_topk) | 0.5614 | 0.275 | 0.743 |
| AND(P_old) | 0.5325 | 0.266 | 0.743 |
| AND(C1_topk+C2_new+P_old) | 0.5166 | 0.207 | 0.678 |
| MAJ2of3 | 0.5747 | 0.317 | 0.804 |
