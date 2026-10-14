# Final model card — S5 × wprod+mass gate

Protocol: `bp_train_30` · TEST n=564 (230 nonempty / 334 empty) · pixel thr=0.5 · empty–empty Dice=1  
Artifacts: `s5_concat/`, `dual_role_train/`, `s5_gate_25d/`, `s5_wprod_mass_filter/`  
(COVID is separate: `results/covid_pipeline/` — still training; not this card.)

---

## 1. Chosen system

**Name:** S5 skip-SEG + dual×2.5D weighted-product presence gate + min-mass filter  

**What is frozen / trained**

| Module | Role | Checkpoint |
|---|---|---|
| S5 SkipVAE (`s5_concat`) | Pixel SEG mask (strong nonempty Dice) | `s5_concat/seg_best.pt` |
| Dual-role plain line + `LesionPresenceHead(μ)` | Presence score `p_dual` | `dual_role_train/.../dual_role_best_auroc.pt` |
| Presence25D ResNet18 | Presence score `p_25d` on FLAIR `(z−1,z,z+1)` | `s5_gate_25d/.../cls25d_best_auroc.pt` |
| Gate (no learnable τ in loss) | Keep / blank S5 mask | τ=0.01, min_mass=320 (VAL→TEST) |

**Inference (one slice)**

1. S5 → mask logits → binary mask @0.5, mass = #predicted lesion pixels  
2. Dual head → `p_dual = σ(Head(μ_cls))`  
3. 2.5D CNN → `p_25d = σ(ResNet18([I_{z-1}, I_z, I_{z+1}]))`  
4. Score  
   \[
   s = \frac{p_{\mathrm{dual}}^{1.5}\, p_{\mathrm{25d}}^{1.3}}{\max s}
   \]  
5. **Keep** S5 mask iff \( s \ge 0.01 \) **and** \(\mathrm{mass} \ge 320\); else blank.

---

## 2. Principles (paper wording)

Not a single named “wprod paper” — composition of standard ideas:

1. **Strong SEG + weak empty control**  
   Skip-concat S5 raises nonempty Dice vs no-skip VAE, but empty-slice FP explodes. Gate after SEG, do not retrain SEG for FP.

2. **Cascade / reject option**  
   Slice-level presence decides whether the SEG output is trusted. Blanking empty FP does not change the SEG network.

3. **Classifier product rule (AND-like fusion)**  
   Weighted product of two heterogeneous presence scores (latent-μ head × image 2.5D) keeps a slice only when both agree. Related: Kittler-style classifier combination / product rule.

4. **Area prior (small-component suppression)**  
   Tiny predicted blobs on empty slices are classic SEG FP; min pred-mass filter removes them without a second network.

5. **τ not in classification loss**  
   Presence heads train with BCE only; operating point (τ, mass) chosen on VAL for the medical trade-off (ne ≥ dual baseline, empty-FP ↓).

---

## 3. TEST performance

### Primary (chosen operating point)

| Metric | Value |
|---|---:|
| Nonempty Dice (ne) | **0.601** |
| Empty FP rate | **0.305** |
| Presence sens / spec | 0.796 / 0.695 |
| TP / TN / FP / FN | 183 / 232 / 102 / 47 |
| Gate AUROC (wprod score) | 0.797 |
| τ / min_mass | 0.01 / 320 |

### Baselines (same TEST)

| System | ne | empty FP |
|---|---:|---:|
| none raw (no-skip VAE) | 0.572 | 0.383 |
| S5 raw (no gate) | 0.664 | 0.790 |
| dual-role @0.22 (prior gate) | 0.600 | 0.383 |
| wprod only (no mass) | 0.601 | 0.335 |
| **wprod + mass (final)** | **0.601** | **0.305** |

**Claim vs prior:** vs dual@0.22, ne matched (~0.60) and empty-FP **0.383 → 0.305** (−7.8 pp). vs none raw, ne **0.572 → 0.601** and FP **0.383 → 0.305**.

### Component AUROCs (presence)

| Scorer | VAL AUROC (approx.) |
|---|---:|
| Dual μ-head | 0.81 (TEST ~0.78) |
| Presence25D | 0.76 |
| wprod score (TEST) | 0.80 |

---

## 4. Why this design (short)

- S5 alone: best ne, unusable empty FP.  
- Dual alone: FP≈none, ne only ≈0.60.  
- Product fuse: both must fire → FP↓ with ne held.  
- Min-mass: removes residual tiny empty blobs → FP 0.335→0.305.  
- Hardneg-heavy 2.5D retrain *failed* (ne collapsed); final 25d is the original AUROC-selected ckpt.

---

## 5. Repro pointers

```text
SEG:     results/bp_original/s5_concat/seg_best.pt
Dual:    results/bp_original/dual_role_train/checkpoints/dual_role_best_auroc.pt
2.5D:    results/bp_original/s5_gate_25d/checkpoints/cls25d_best_auroc.pt
Eval:    eval_mass_filter.py  →  s5_wprod_mass_filter/report.json
```

COVID counterpart (domain-isolated, do not mix ckpts): `results/covid_pipeline/`.
