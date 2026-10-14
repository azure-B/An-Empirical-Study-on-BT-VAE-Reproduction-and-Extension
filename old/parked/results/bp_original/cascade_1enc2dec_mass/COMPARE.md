# Clean 1-encoder · 2-decoder (+ min-mass)

## Structure (review-friendly)

```text
x → Encoder (shared BT)
      ├─ μ + skips → Decoder_seg (S5) → mask, mass m
      └─ μ → Decoder_cls (amax → p)     [strict 2-dec]
         or Head(μ)+Head(skips) → p     [same encoder, light heads]
keep if p ≥ τ and m ≥ m_min
```

No second VAE encoder, no external ResNet.

---

## TEST results

| System | ne | empty FP | note |
|---|---:|---:|---|
| none raw | 0.572 | 0.383 | baseline |
| **wprod+mass (messy)** | **0.601** | **0.305** | S5 + dual enc + 25d CNN |
| 1enc2dec focal DecCls + mass | 0.571 | 0.350 | clean; ≈none ne, FP slightly better |
| 1enc2dec focal score-only | 0.572 | 0.377 | learned-style τ |
| **1enc2dec cls_recall + mass** | **0.635** | 0.461 | clean; ne↑ but FP worse |
| cascade raw SEG (shared) | 0.716 | 0.913 | ungated upper bound |

### Detail (this run)

| recipe | raw ne/fp | score-gate | +mass |
|---|---|---|---|
| DecCls (focal) | 0.716/0.913 | 0.572/0.377 @τ0.60 | 0.571/0.350 (τ0.60, m≥640) |
| Heads (cls_recall) | 0.716/0.913 | 0.530/0.437 @τ0.66 | **0.635/0.461** (τ0.63, m≥1280) |

---

## Takeaway

| Goal | Pick |
|---|---|
| **Easy review / clean paper figure** | 1enc–2dec (focal DecCls) |
| **Best empty-FP @ ne≈0.60** | messy wprod+mass (0.601/0.305) |
| **Higher ne, accept FP** | 1enc2dec cls_recall+mass (0.635/0.461) |

Clean architecture **does not yet beat** wprod+mass on the (ne≥0.60, FP↓) Pareto — cls branch is the bottleneck (need better presence under one encoder), not SEG.
