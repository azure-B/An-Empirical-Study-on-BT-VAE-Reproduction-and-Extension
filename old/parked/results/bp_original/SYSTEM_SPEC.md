# S5 × wprod+mass — System Specification (standalone)

Single-file brief for external review. Covers architecture, layers, math, training, and TEST metrics.  
Domain: brain FLAIR slice lesion SEG (`bp_train_30`). Input \(I\in\mathbb{R}^{1\times 256\times 256}\).

---

## 0. One-sentence summary

A **frozen S5 skip-VAE** produces a strong lesion mask; a **slice-level gate** (weighted product of a μ-head and a 2.5D CNN, plus min predicted mass) blanks empty false-positive slices without retraining SEG.

---

## 1. End-to-end inference graph

```text
FLAIR slice I_z
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
┌─────────────┐            ┌──────────────────┐
│ S5 SkipVAE  │            │ Dual cls_net     │
│ (frozen)    │            │ (no-skip VAE)    │
│ → mask M    │            │ → μ_cls          │
│ → mass m    │            │ → Head(μ)→p_dual │
└──────┬──────┘            └────────┬─────────┘
       │                            │
       │                   I_{z-1},I_z,I_{z+1}
       │                            ▼
       │                   ┌──────────────────┐
       │                   │ Presence25D      │
       │                   │ ResNet18 (3ch)   │
       │                   │ → p_25d          │
       │                   └────────┬─────────┘
       │                            │
       │         s ∝ p_dual^1.5 · p_25d^1.3
       │                            │
       └──────────► GATE ◄──────────┘
                    keep iff s≥τ ∧ m≥m_min
                    else blank
                         │
                         ▼
                   output mask M̂
```

**Final operating point (VAL→TEST):** \(\tau=0.01\), \(m_{\min}=320\) pixels.

---

## 2. Hyperparameters (shared backbone)

| Symbol | Value | Meaning |
|---|---:|---|
| `base` | 32 | channel multiplier |
| latent | \(1024\times 8\times 8\) | \(=32\cdot\mathrm{base}\) at bottleneck |
| spatial | \(256\to128\to64\to32\to16\to8\) | 5× downsample |
| Dropout2d | 0.1 | encoder/decoder |
| pixel thr | 0.5 | \(\hat M=\mathbb{1}[\sigma(\ell)\ge 0.5]\) |

Channel schedule: \(32\to 64\to 128\to 256\to 520\to 1024\) (i.e. \(\mathrm{base}\times\{1,2,4,8,16,32\}\)).

---

## 3. Module A — S5 SkipVAE (segmentation)

**Role:** produce lesion logits \(\ell\in\mathbb{R}^{1\times H\times W}\).  
**File:** `models_bp_skip.py` · ckpt `s5_concat/seg_best.pt`  
**At final infer:** frozen; deterministic decode uses \(\mu\) (not sampled \(z\)).

### 3.1 Encoder

Stem: `Conv3×3 → BN → LeakyReLU(0.2) → Dropout2d`

Then 5× `ResBlockDown` (residual: main path Conv-stride2 + shortcut), each followed by Dropout2d:

| Stage | out channels | typical size |
|---|---:|---:|
| e0 (stem) | 32 | 256² |
| e1 | 64 | 128² |
| e2 | 128 | 64² |
| e3 | 256 | 32² |
| e4 | 512 | 16² |
| h → μ, logvar | 1024 | 8² |

\[
\mu,\log\sigma^2 = 1\times1\text{-Conv+BN}(h),\quad
z=\mu+\varepsilon\odot\exp(\tfrac12\log\sigma^2)\ \ (\text{train only})
\]

Skips returned: \((e_0,e_1,e_2,e_3,e_4)\).

### 3.2 Decoder (S5 = B0–B3 concat fuse, **B4 off**)

Upsample path: 5× `ResBlockUp` (nearest×2 + residual).  
Skip fusion = **concat + 3×3 Conv + BN + LeakyReLU** (`ConcatConvFuse`), applied at e3,e2,e1,e0 only (**e4 unused**).

| After | fuse with | decoder ch before fuse |
|---|---|---:|
| up1, up2 | — | 1024 |
| fuse3 | e3 (256ch) | 1024 |
| up3 | — | 512 |
| fuse2 | e2 | 512 |
| up4 | — | 256 |
| fuse1 | e1 | 256 |
| up5 | — | 32 |
| fuse0 | e0 | 32 |
| out | `Conv3×3 → 1` | logits \(\ell\) |

### 3.3 SEG losses (training)

Dice+BCE on \(\sigma(\ell)\) vs GT mask \(Y\), plus KL on \((\mu,\log\sigma^2)\):

\[
\mathcal{L}_{\mathrm{SEG}} = 0.99\cdot\mathcal{L}_{\mathrm{DiceCE}} + 0.01\cdot\mathrm{KL}
\]

(Exact weights from `train_original_bp`: `RECON_WEIGHT`, `BETA_LOCAL`.)

### 3.4 Mask & mass at infer

\[
\hat M = \mathbb{1}[\sigma(\ell)\ge 0.5],\qquad
m = \sum_{u,v}\hat M_{uv}
\]

---

## 4. Module B — Dual-role presence (\(p_{\mathrm{dual}}\))

**Role:** slice-level P(lesion present) from latent of a **no-skip** VAE line.  
**Files:** `models_dual_role.py`, `models_lesion_presence.py`  
**Ckpt:** `dual_role_train/checkpoints/dual_role_best_auroc.pt`

### 4.1 Two lines

| Line | Network | Train at dual phase | Infer use |
|---|---|---|---|
| A `skip_net` | S5 SkipVAE | **frozen** | SEG mask (same as §3) |
| B `cls_net` | OrigVAE (no skips) | encoder+decoder+head | \(\mu_{\mathrm{cls}}\) only for head |

Plain decoder: upsample residual blocks **without** skip concat (original BT-VAE).

### 4.2 LesionPresenceHead

Input: \(\mu_{\mathrm{cls}}\in\mathbb{R}^{1024\times 8\times 8}\)

\[
\begin{aligned}
a &= \mathrm{GAP}(\mu),\quad b=\mathrm{GMP}(\mu)\\
h &= \mathrm{ReLU}(W_1[a;b]+c_1)\in\mathbb{R}^{128}\\
\mathrm{logit}_{\mathrm{dual}} &= W_2 h + c_2\in\mathbb{R}\\
p_{\mathrm{dual}} &= \sigma(\mathrm{logit}_{\mathrm{dual}})
\end{aligned}
\]

Dropout 0.5 on hidden. No sigmoid inside the module (applied outside).

### 4.3 Dual training loss (simplified)

\[
\mathcal{L} = \mathrm{BCE}(\mathrm{logit}_{\mathrm{dual}},\, y)
 + \lambda_{\mathrm{plain}}\cdot\mathrm{DiceCE}(\mathrm{plain\ SEG},\, Y)
 + \beta\cdot\mathrm{KL}(\mu_{\mathrm{cls}})
\]

with \(y=\mathbb{1}[\sum Y>0]\), \(\lambda_{\mathrm{plain}}=0.25\), no grad into S5.  
Warm head-only, then FT cls line. **Gate τ not in this loss.**

---

## 5. Module C — Presence25D (\(p_{25d}\))

**Role:** image-space presence with adjacent-slice context.  
**File:** `models_presence_25d.py` · ckpt `s5_gate_25d/.../cls25d_best_auroc.pt`

**Input:** stack as 3-channel image
\[
X = [I_{z-1},\, I_z,\, I_{z+1}]\in\mathbb{R}^{3\times 256\times 256}
\]
(missing neighbor → replicate center).

**Backbone:** torchvision ResNet18, random init (not ImageNet), `conv1` stays 3ch (Kaiming re-init), FC replaced by:

`Dropout(0.3) → Linear(512 → 1)` → logit → \(p_{25d}=\sigma(\cdot)\).

**Loss:** \(\mathrm{BCEWithLogits}\) with `pos_weight=2.0`, early-stop on VAL AUROC. τ ∉ loss.

---

## 6. Gate — weighted product + min-mass

### 6.1 Score (wprod)

\[
\tilde s = p_{\mathrm{dual}}^{\alpha}\, p_{25d}^{\beta},\quad
\alpha=1.5,\ \beta=1.3
\]

\[
s = \frac{\tilde s}{\max_i \tilde s_i}
\quad\text{(normalize over the split / batch used in eval)}
\]

Product is **AND-like**: either score near 0 → \(s\) collapses (rejects empty FP that only one model likes).

### 6.2 Decision

\[
\mathrm{keep} = \mathbb{1}[s \ge \tau]\ \land\ \mathbb{1}[m \ge m_{\min}]
\]

\[
\hat M_{\mathrm{final}} =
\begin{cases}
\hat M & \text{if keep}\\
\mathbf{0} & \text{otherwise}
\end{cases}
\]

Chosen: \(\tau=0.01\), \(m_{\min}=320\).

### 6.3 Design principles (not one paper)

1. Strong SEG + post-hoc empty control (cascade / reject option)  
2. Heterogeneous classifier **product rule** (μ-head × 2.5D CNN)  
3. Small-blob / area prior (min mass)  
4. Calibrate \((\tau,m_{\min})\) on VAL; do not bake τ into BCE  

---

## 7. Metrics definitions

TEST: \(n=564\) (230 nonempty / 334 empty GT). Pixel thr 0.5. Empty–empty Dice \(:=1\).

| Name | Definition |
|---|---|
| **ne** (nonempty Dice) | mean Dice over slices with \(\sum Y>0\) |
| **empty FP rate** | among GT-empty slices, fraction with \(\sum\hat M_{\mathrm{final}}>0\) |
| Presence TF | keep vs \(y=\mathbb{1}[\sum Y>0]\) |

---

## 8. TEST results (final)

| System | ne | empty FP |
|---|---:|---:|
| none raw (no-skip) | 0.572 | 0.383 |
| S5 raw (ungated) | 0.664 | 0.790 |
| dual-only gate @0.22 | 0.600 | 0.383 |
| wprod only | 0.601 | 0.335 |
| **wprod + mass (final)** | **0.601** | **0.305** |

Final presence: sens **0.796**, spec **0.695**, TP/TN/FP/FN = **183/232/102/47**, gate AUROC **0.797**.

**Claim:** vs dual@0.22, ne held (~0.60) while empty FP **0.383→0.305**; vs none raw, both ne↑ and FP↓.

---

## 9. What is *not* in this system

- Cascade dual-decoder / learned τ-in-loss (archived)  
- 1ch image-only ResNet (archived; replaced by 2.5D)  
- Hard-negative 25d finetune (tried; hurt ne)  
- Soft `mask×p` as primary deliverable (side experiments only)  

---

## 10. Artifact map (repo)

```text
models_bp_skip.py          S5
models_dual_role.py        dual wrapper
models_lesion_presence.py  Head(μ)
models_presence_25d.py     2.5D
eval_mass_filter.py        final gate eval
train_original_bp.py       S5/none train
train_dual_role.py         dual train
train_s5_gate_25d.py       25d train
```

Ckpts under `results/bp_original/{s5_concat,dual_role_train,s5_gate_25d}/`.  
Numbers: `s5_wprod_mass_filter/report.json`.
