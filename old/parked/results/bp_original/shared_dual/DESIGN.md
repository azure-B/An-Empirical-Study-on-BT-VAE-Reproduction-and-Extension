# Shared-encoder dual decoder

```
x → Encoder (S5, frozen)
      → μ, skips
           ├─ Decoder_skip(μ, skips)  → SEG   [frozen from s5_concat]
           └─ Decoder_cls(μ) + Head(μ) → P → τ gate on SEG
```

Train: warm head → FT decoder_cls+head. No second encoder.
Infer: keep skip mask iff P≥τ.
