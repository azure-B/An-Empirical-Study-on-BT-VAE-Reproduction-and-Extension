# wprod+mass → 1-encoder compression

## Goal

Keep the **gate math** that beat dual@0.22 (weighted product + min-mass), drop the **extra backbones** (plain VAE encoder + ResNet18).

| Messy wprod | This (1enc) |
|---|---|
| S5 SkipVAE encoder | **same** (sole encoder) |
| Dual `cls_net` encoder → Head(μ) | Head_μ(**S5 μ_z**) |
| Presence25D ResNet18(I_{z-1:z+1}) | Head_ctx(**S5 μ** of neighbors) |
| s ∝ p^1.5 · p^1.3 + m≥m_min | **unchanged** |

## Graph

```text
I_{z-1}, I_z, I_{z+1}
        │  (same frozen S5 encoder ×3)
        ▼
   μ_prev, μ_c, μ_next   +   skips_c → Dec_seg → mask, mass m
        │
        ├─ Head_μ(μ_c)              → p_μ
        └─ Head_ctx([μ_p,μ_c,μ_n])  → p_ctx
                    │
         s ∝ p_μ^1.5 · p_ctx^1.3
         keep iff s≥τ ∧ m≥m_min
```

## Why this preserves wprod

1. **Heterogeneous product** — single-slice μ vs multi-slice context (AND-like reject).
2. **One weight set** — no second VAE, no external CNN.
3. **τ ∉ BCE** — train heads with BCE; calibrate (τ, m_min) on VAL.

## Train

- Freeze S5 (encoder+decoder).
- Train Head_μ + Head_ctx only (pos_weight BCE, early-stop VAL AUROC of product score).
- Eval: same sweep as messy `eval_mass_filter` (α=1.5, β=1.3).

Target: approach messy **0.601 / 0.305** under clean 1-encoder review graph.
