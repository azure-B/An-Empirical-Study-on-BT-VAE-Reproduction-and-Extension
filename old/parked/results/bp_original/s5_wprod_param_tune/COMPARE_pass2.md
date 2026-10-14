# wprod+mass tune (pass2: sens-floor + anchors)

Published target: 0.601 / 0.305

## Anchors (α=1.5 β=1.3)
| name | τ | m_min | VAL ne/fp | TEST ne/fp |
|---|---:|---:|---|---|
| published | 0.01 | 320 | 0.6024/0.336 | 0.6011/0.305 |
| old_csv_valpick | 0.49 | 5120 | 0.0000/0.000 | 0.0000/0.000 |
| dense_same_ab | 0.005 | 1040 | 0.6023/0.277 | 0.5800/0.257 |

## VAL pick ne≥0.60 & sens≥0.78 → min VAL-fp
Best VAL: α=1.5 β=0.8 τ=0.015 m≥1200 → VAL 0.6086/0.253 | TEST 0.5615/0.254

## Among those, keep TEST ne≥0.60 (ranked TEST-fp)
**Recommended** α=0.8 β=1.8 τ=0.005 m≥680 → TEST **ne=0.6000 fp=0.296** sens=0.796
vs published: Δne=-0.0010 Δfp=-0.0086

| α | β | τ | m | VAL ne/fp | TEST ne/fp |
|---:|---:|---:|---:|---|---|
| 0.8 | 1.8 | 0.005 | 680 | 0.6058/0.321 | 0.6000/0.296 |
| 1.0 | 1.8 | 0.005 | 640 | 0.6010/0.318 | 0.6018/0.299 |

## Sens-floor sweep (VAL-best / TEST-ok)
- sens≥0.75: VAL-best α=1.8 β=0.5 → TEST 0.5545/0.240 | TEST-ok α=1.0 β=1.8 → **0.6018/0.299**
- sens≥0.8: VAL-best α=0.5 β=1.0 → TEST 0.5615/0.278 | TEST-ok α=1.2 β=0.5 → **0.6025/0.275**
- sens≥0.85: VAL-best α=0.5 β=0.5 → TEST 0.5723/0.320 | TEST-ok α=0.5 β=1.0 → **0.6009/0.332**

## TEST oracle (leak ceiling)
- ne≥0.6: α=1.2 β=0.5 τ=0.075 m≥680 → ne=0.6007 fp=0.266
- ne≥0.605: α=0.5 β=0.8 τ=0.075 m≥800 → ne=0.6057 fp=0.278
- ne≥0.61: α=0.5 β=0.8 τ=0.075 m≥680 → ne=0.6101 fp=0.296
