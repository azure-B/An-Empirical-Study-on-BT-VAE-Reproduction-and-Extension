# Skip Ablation Summary (COVID)

| Stage | Mode | TEST Dice | nonempty mean | nonempty median | best val |
|---|---|---:|---:|---:|---:|
| S1 | `none` | 0.6544 | 0.5358 | 0.6098 | 0.7826 @25 |
| S2 | `full` | 0.7629 | 0.7829 | 0.8391 | 0.8471 @346 |
| S3 | `b0b3` | 0.7848 | 0.7796 | 0.8361 | 0.8273 @31 |
| S4 | `b0b3_learn` | 0.7928 | 0.7749 | 0.8351 | 0.8093 @71 |
| S5 | `b0b3_concat` | 0.8110 | 0.7890 | 0.8494 | 0.8636 @83 |

## Mapping

- B0=e0(256), B1=e1(128), B2=e2(64), B3=e3(32), B4=e4(16)
- S3–S5: B0..B3 on, **B4 off only**
- S2: reused from `archive/skiponly_covid_base32` (full additive)
- S4: zero-init 1×1 + learnable scale (no Attention)
- S5: concat(decoder, skip) → 3×3 Conv+BN+LeakyReLU (no Attention, no learnable γ)
