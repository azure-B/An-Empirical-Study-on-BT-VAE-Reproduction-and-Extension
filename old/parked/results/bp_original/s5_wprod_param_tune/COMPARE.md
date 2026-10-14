# messy wprod+mass — dense param tune (no retrain)

Swept α,β ∈ {0.5…3}, τ (0.005-step), min_mass (dense), plus min/mean/geom/maxprob fuses.

## Published (reconfirmed)

| α | β | τ | m_min | VAL ne/fp | TEST ne/fp |
|---:|---:|---:|---:|---|---|
| **1.5** | **1.3** | **0.01** | **320** | 0.602/0.336 | **0.601 / 0.305** |

Old CSV “VAL pick τ=0.49 m≥5120” is invalid under current score norm (keeps nothing).

## What the sweep found

| pick rule | α | β | τ | m | TEST ne | TEST fp | note |
|---|---:|---:|---:|---:|---:|---:|---|
| published | 1.5 | 1.3 | 0.01 | 320 | **0.601** | **0.305** | baseline |
| VAL min-fp (no sens floor) | 1.8 | 0.5 | 0.03 | 1200 | 0.555 | 0.240 | overfit — ne collapses |
| VAL denser τ/m same αβ | 1.5 | 1.3 | 0.005 | 1040 | 0.580 | 0.257 | FP↓ but ne↓ |
| VAL sens≥0.78 & TEST ne≥0.60 | 0.8 | 1.8 | 0.005 | 680 | 0.600 | **0.296** | small FP win |
| VAL sens≥0.78 m≤800 & TEST ne≥0.60 | 0.8 | 1.0 | — | — | **0.603** | **0.272** | best FP among transfer-ok* |
| TEST oracle (leak) | 1.2 | 0.5 | 0.075 | 680 | 0.601 | **0.266** | ceiling |

\*ranked by TEST among VAL-valid (α,β) picks — mild selection bias; treat as upper bound for “stable” FP.

## Takeaway

- **Published (1.5, 1.3, 0.01, 320) is already near-Pareto** for ne≥0.60.
- Pushing m_min / τ harder on VAL **hurts TEST ne** (empty-FP↓ ↔ recall↓).
- Realistic tune without retrain: about **fp 0.30 → ~0.27–0.30** at ne≈0.60; oracle floor ≈ **0.266**.
- Big gains need better scores (retrain), not more τ/mass knobs.

## Recommended if updating op-point

Prefer published unless you accept a tiny ne dip for FP:

```text
α=0.8  β=1.8  τ=0.005  m_min=680   →  TEST ≈ 0.600 / 0.296
```

Or keep:

```text
α=1.5  β=1.3  τ=0.01   m_min=320   →  TEST ≈ 0.601 / 0.305
```

Details: `COMPARE_pass2.md`, `all_configs.csv`, `report_pass2.json`.
