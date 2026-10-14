# Cascade cls-recall

```
x → BT Enc (freeze) → μ, skips
      ├─ Head(μ) + Head(skips) → P   # train: asymmetric BCE, NO τ
      └─ Decoder_seg (freeze, from cascade_focal) → mask

τ calibrated on VAL: max empty-FP↓ among sens ≥ floor
infer: keep SEG iff P ≥ τ
```