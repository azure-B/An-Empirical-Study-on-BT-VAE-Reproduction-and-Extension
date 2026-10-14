# 1enc · 2dec + cls Head (TF/FP-aware)

```text
Encoder (freeze)
  ├─ Decoder_seg (freeze) → mask, mass
  └─ Decoder_cls → cls_map → Head_map
       + Head_μ(μ) → p
keep if p≥τ ∧ m≥m_min
```

Hard weights in BCE:
- FT risk (empty & SEG fire): ×3.0
- TF risk (nonempty & SEG fire): ×2.5
- τ not in loss; VAL pick min empty-FP @ ne floor
