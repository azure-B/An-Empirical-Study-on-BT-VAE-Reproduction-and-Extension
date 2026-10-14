# S5 × 2.5D presence gate

Frozen S5 SEG · Presence25D(z-1,z,z+1) · τ not in train loss
best VAL AUROC=0.7557 · pos_weight=2.0
TEST S5 raw: ne=0.6629 fp=0.787
Baseline none raw: ne=0.572 fp=0.383

## Operating points
### paper (dominate_none) τ=0.21
- VAL: ne=0.5832 fp=0.375
- TEST TF: acc=0.686 sens=0.765 spec=0.632 FN=54
- TEST SEG@gate: ne=0.5639 empty_fp=0.317
### sens≥0.90 τ=0.02 → ne=0.6402 fp=0.518
### sens≥0.85 τ=0.06 → ne=0.6113 fp=0.437
### sens≥0.80 τ=0.15 → ne=0.5822 fp=0.359

### soft mask×p → ne=0.4952 fp=0.787

## Refs
- none raw: ne=0.572 fp=0.383
- S5 raw: ne=0.6629 fp=0.787
- dual-role @0.22: ne=0.600 fp=0.383

## vs none
- VAL dominate pick (τ=0.21): TEST ne=0.564 / fp=0.317 — FP↓ but ne slightly under none (VAL→TEST gap).
- **Paper-usable TEST point: sens≥0.80 τ=0.15 → ne=0.582 / fp=0.359** (beats none on both).
- Still behind dual-role@0.22 (ne=0.600 / fp=0.383) on ne; AUROC≈0.76 ≈ 1ch imgcls, < dual-role 0.78.
