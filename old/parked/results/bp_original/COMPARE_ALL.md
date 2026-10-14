# BP experiments — full COMPARE

Protocol: `bp_train_30` · same VAL/TEST loaders · pixel thr=0.5 · empty–empty Dice=1  
TEST n=564 (230 nonempty / 334 empty) · VAL n=563 (227 nonempty / 336 empty)

Primary medical view: **presence T/F** (empty FP ↓ without killing lesion recall) + **nonempty Dice**.

---

## 1. Core SEG baselines (TEST)

| Experiment | Dice | Nonempty (ne) | Empty FP | Nonempty P | Nonempty R | Notes |
|---|---:|---:|---:|---:|---:|---|
| **none** | 0.599 | 0.572 | 0.383 | 0.829 | 0.551 | original VAE, full FT |
| **s5 skip** | 0.396 | **0.663** | 0.787 | 0.777 | 0.650 | high ne, high empty FP |
| region seed | 0.536 | 0.650 | 0.542 | 0.806 | 0.639 | stopped early |
| encoder freeze | 0.398 | 0.677 | 0.793 | 0.727 | 0.685 | empty FP worse |
| BT proto | 0.243 | 0.478 | 0.919 | 0.576 | 0.472 | weak |
| empty-FP FT (none) | 0.675 | 0.437 | **0.162** | 0.896 | 0.428 | FP↓ but ne↓ |

Source: `COMPARE.md` / `COMPARE.json` (2026-09-28 re-eval).

### VAL snapshot (baselines)

| Experiment | Dice | ne | Empty FP |
|---|---:|---:|---:|
| none | 0.608 | 0.644 | 0.417 |
| s5 skip | 0.419 | 0.699 | 0.771 |
| presence τ=0.20 | 0.665 | 0.601 | 0.292 |
| empty-FP FT | 0.671 | 0.471 | 0.193 |

---

## 2. Presence gate on frozen none (TEST)

| Experiment | ne | Empty FP | Notes |
|---|---:|---:|---|
| none raw | 0.572–0.579 | 0.380–0.383 | C0 |
| **presence τ=0.20** | 0.533 | **0.269** | Head(μ) · prior FP reference |
| neighbor isolated | 0.513 | 0.269 | no train |
| neighbor small | 0.568 | 0.335 | no train |

---

## 3. Presence combo (none mask + multi-gate) — TEST

Frozen none SEG. Keep if gate says present.

| method | ne | empty FP | sens |
|---|---:|---:|---:|
| C0 raw | **0.579** | 0.380 | 0.826 |
| MAJ2of3 | 0.575 | 0.317 | 0.804 |
| AND(C1_topk) | 0.561 | 0.275 | 0.743 |
| AND(P_old) | 0.533 | 0.266 | 0.743 |
| AND(C1+P) | 0.517 | 0.216 | 0.678 |
| AND(C1+C2+P) | 0.517 | **0.207** | 0.678 |

Takeaway: AND lowers FP and ne together; no combo beats C0 ne while beating P_old FP.  
Detail: `presence_combo/COMPARE.md`

---

## 4. S5 × presence combo — TEST

| pick | ne | fp | note |
|---|---:|---:|---|
| s5 C0 | **0.664** | 0.790 | SEG upper bound on ne |
| none ∧ P_old | 0.533 | 0.266 | prior FP best |
| **hybrid MAJ2** (none gates + S5 mask) | **0.636** | 0.488 | +ne vs none, FP≪S5 |
| hybrid P_old | 0.591 | 0.455 | |
| hybrid Soft@0.14 | 0.653 | 0.596 | almost S5 ne |
| native topk@0.82 | 0.659 | 0.638 | ne kept, FP still high |
| hybrid ∧ C1+P | 0.517 | **0.207** | aggressive |

Takeaway: S5 ne carries if gate is soft/majority; hard C1∧P kills the S5 boost.  
Detail: `s5_presence_combo/COMPARE.md`

---

## 5. Dual-role (S5 frozen + cls line + head) — TEST

S5 mask kept iff P(head μ_cls) ≥ τ.

| cls init | pick | ne | FP | AUROC |
|---|---|---:|---:|---:|
| none | plainFloor@0.22 | **0.600** | 0.383 | 0.782 |
| none | @0.30 | 0.546 | 0.266 | 0.782 |
| emptyfp | plainFloor@0.13 | 0.546 | 0.287 | 0.777 |
| emptyfp | bal@0.32 | 0.496 | **0.204** | 0.777 |
| S5 raw | — | 0.664 | 0.790 | — |
| prior none∧P@0.20 | — | 0.533 | 0.266 | — |

### τ from classification metrics (none-cls)

| τ rule | τ | ne | empty FP | sens | cls F1 |
|---|--:|---:|---:|---:|---:|
| S5 raw | — | 0.664 | 0.790 | 0.974 | — |
| cls-f1 | 0.34 | 0.523 | 0.219 | 0.678 | 0.667 |
| cls-youden / acc | 0.56 | 0.395 | 0.126 | 0.500 | 0.593 |

### τ sweep 0.01–0.15 highlight (none-cls → S5)

| τ | ne | FP | sens | TT | FT | TF |
|--:|---:|---:|-----:|---:|---:|---:|
| 0.01 | 0.664 | 0.790 | 0.974 | 224 | 264 | 6 |
| 0.10 | 0.648 | 0.626 | 0.930 | 214 | 209 | 16 |
| 0.15 | 0.639 | 0.530 | 0.904 | 208 | 177 | 22 |

Detail: `dual_role_COMPARE.md`, `dual_role_train/`, `dual_role_tau_*`

---

## 6. Cascade + follow-up (done)

### 6a. `bt_cascade` (Dice/BCE, learned τ≈0.615) — TEST

| | value |
|---|---|
| TP / TN / FP / FN | 155 / 228 / 106 / 75 |
| acc / sens / spec | 0.679 / 0.674 / 0.683 |
| F1 / AUROC | 0.631 / 0.753 |
| SEG@gate ne | 0.497 |
| empty FP | 0.308 |

Raw SEG VAL ne peak ~0.68; gate cuts ne to ~0.50.

### 6b. `presence_imgcls` ResNet18 (no μ) — TEST

| pick | acc | sens | spec | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| τ=0.50 | 0.688 | 0.726 | 0.662 | — | 0.757 |
| acc τ=0.88 | **0.707** | 0.552 | 0.814 | 0.606 | 0.757 |

Better than cascade cls (0.68) but **≪ 0.90**.

### 6c. `presence_fuse` (imgcls × SEG) — TEST @acc

| recipe | acc | sens | spec | ne | empty FP |
|---|---:|---:|---:|---:|---:|
| cascade_cls | **0.711** | 0.513 | 0.847 | 0.386 | **0.150** |
| img_only | 0.707 | 0.552 | 0.814 | 0.420 | 0.177 |
| mean_img_cas | 0.711 | 0.539 | 0.829 | 0.416 | 0.165 |
| max_img_seg | 0.637 | **0.852** | 0.488 | **0.630** | 0.506 |

Fusion does **not** get acc to 0.90. High-ne recipes still dump empty FP.

### 6d. `bt_cascade_focal` (Dice+Focal, τ≈0.654) — TEST

| | value |
|---|---|
| TP / TN / FP / FN | 157 / 235 / 99 / 73 |
| acc / sens / spec | **0.695** / 0.683 / 0.704 |
| F1 / AUROC | 0.646 / 0.757 |
| SEG@gate ne | **0.521** |
| empty FP | **0.281** |
| raw SEG ne | **0.716** (VAL peak 0.693) |

vs plain cascade: slightly better TF + gated ne + FP. vs presence τ=0.20 (ne 0.533 / fp 0.269): **almost the same Pareto**, raw SEG much stronger.

Next if still chasing acc 0.90: 2.5D / attention (step 4), not more μ-head tweaks.

### 6e. `bt_cascade_cls_recall` (freeze SEG focal · recall-first cls · τ≠loss) — TEST

Design: freeze encoder+SEG from `cascade_focal`; train `Head(μ)+Head(skips)` with `pos_weight=3`; **τ not in BCE**; pick τ on VAL by sens-floor (minimize empty FP among sens≥floor). Early-stop @ ep16, VAL AUROC≈0.64.

| pick | τ | TF sens / FN | gated ne | empty FP |
|---|---:|---:|---:|---:|
| sens≥0.95 | 0.58 | **0.965** / **8** | **0.695** | 0.823 |
| sens≥0.90 | 0.62 | 0.935 / 15 | **0.676** | 0.746 |
| sens≥0.85 | 0.63 | 0.917 / 19 | 0.664 | 0.650 |
| sens≥0.80 | 0.64 | 0.830 / 39 | 0.606 | 0.569 |
| f1 | 0.66 | 0.722 / 64 | 0.530 | 0.437 |
| cascade_focal gated (ref) | 0.65 | 0.683 / 73 | 0.521 | **0.281** |
| raw SEG (frozen) | — | — | **0.716** | 0.913 |

Takeaway: recall-first τ **does** unlock gated ne (0.52 → **0.66–0.69**), FN≪focal — but cls AUROC still ~0.65 so empty FP stays high. Need a stronger discriminator (2.5D / better features), not just softer τ.

Detail: `bt_cascade_cls_recall/COMPARE.md`

### 6f. `s5_gate_25d` (freeze S5 · 2.5D ResNet18 · τ≠loss) — TEST

Design: frozen `s5_concat` SEG; train Presence25D on FLAIR `(z-1,z,z+1)`; τ not in BCE. Early-stop @32, VAL AUROC≈0.756.

| pick | τ | gated ne | empty FP | vs none |
|---|---:|---:|---:|---|
| S5 raw | — | **0.663** | 0.787 | — |
| VAL dominate-none | 0.21 | 0.564 | **0.317** | FP↓, ne slightly under |
| **sens≥0.80** | 0.15 | **0.582** | **0.359** | **beats both** (0.572/0.383) |
| sens≥0.85 | 0.06 | 0.611 | 0.437 | ne↑ fp↑ |
| dual-role@0.22 (ref) | 0.22 | 0.600 | 0.383 | still better ne@similar fp |

Takeaway: 2.5D cls-only **can beat none** (ne 0.582 / fp 0.359) with frozen S5 — paper-minimum met. Does **not** yet beat dual-role ne or hit stretch (ne≥0.65 & fp≤0.383); AUROC≈imgcls, not a big jump over 1ch.

Detail: `s5_gate_25d/COMPARE.md`

### 6g. `s5_fuse_reject` (dual μ-head × 2.5D → S5 gate) — TEST

Empty-FP push without SEG retrain. Fuse dual-role presence with Presence25D; gate frozen S5.

| recipe | τ | ne | empty FP | vs dual@0.22 |
|---|---:|---:|---:|---|
| dual@0.22 (ref) | 0.22 | 0.600 | 0.383 | — |
| dual alone (VAL beat) | 0.28 | 0.554 | 0.284 | FP↓ ne↓ |
| **prod** `p_d·p_25` | 0.03 | **0.606** | **0.356** | **beats both** |
| **wprod** `p_d^1.5·p_25^1.3` | 0.01 | **0.601** | **0.335** | **best FP beat** |
| logistic rejector | 0.41 | 0.639 | 0.737 | failed (AUROC~0.40) |

Takeaway: **AND-like product fuse** cuts empty FP ~5pp vs dual while holding ne≥0.60. Tiny logistic on (p, mass, maxprob) did not help — need better hard-negative training next if FP must go ≪0.33.

Detail: `s5_fuse_reject/COMPARE.md`

### 6h. Hardneg retrain + mass filter (empty-FP push) — TEST

| recipe | ne | empty FP | note |
|---|---:|---:|---|
| wprod (prior best) | 0.601 | 0.335 | dual^1.5·25d^1.3 |
| 25d hardneg finetune fuse | ≲0.57 | — | **failed** (killed ne) |
| 25d scratch + rej v3 | ~0.57 | ~0.27 | FP↓ but ne < 0.60 |
| **wprod + min_mass≥320** | **0.601** | **0.305** | **beats prior** |

Takeaway: aggressive hardneg retrain hurt ne. Best empty-FP cut while holding ne≥0.60 is **wprod gate + drop tiny S5 blobs** (min pred mass). fp 0.383→0.305 vs dual@0.22.

Detail: `s5_wprod_mass_filter/COMPARE.md`

---

## 7. Pareto cheat-sheet (TEST, selected)

| Goal | Best-ish pick | ne | FP |
|---|---|---:|---:|
| Max nonempty Dice | s5 C0 | 0.664 | 0.790 |
| Low FP, accept ne drop | none empty-FP FT / AND(C1+P) | 0.44–0.52 | 0.16–0.21 |
| Prior balanced gate | presence τ=0.20 | 0.533 | 0.269 |
| S5 ne + milder FP | hybrid MAJ2 | 0.636 | 0.488 |
| Dual-role mid | none-cls @0.22 | 0.600 | 0.383 |
| Dual-role lower FP | none-cls @0.30 | 0.546 | 0.266 |
| cascade_focal gated | learned τ=0.65 | 0.521 | 0.281 |
| cascade_focal raw | — | **0.716** | 0.913 |
| **cls_recall sens≥0.85** | VAL τ=0.63 | **0.664** | 0.650 |
| **s5_gate_25d sens≥0.80** | VAL τ=0.15 | **0.582** | **0.359** |
| **fuse wprod (beat dual)** | VAL τ=0.01 | **0.601** | **0.335** |
| **wprod + mass≥320** | τ=0.01 | **0.601** | **0.305** |

Hard lesson across runs: **FP↓ almost always costs ne** unless SEG is strong *and* cls FN stays low. Product fuse + min-mass filter is current best vs dual@0.22.

---

## 8. Artifact index

| Path | Content |
|---|---|
| `COMPARE.md` / `COMPARE.json` | Core SEG re-eval table |
| `NOTE.md` | Early experiment notes |
| `presence_combo/` | none multi-gate |
| `s5_presence_combo/` | hybrid / native S5×gate |
| `dual_role_*` | two-line S5+cls, τ sweeps |
| `none/`, `s5_concat/`, `none_presence/`, `none_emptyfp/`, … | per-run ckpts & logs |
| `bt_shared_dual/` | aborted shared-μ / skips-cls attempt |
| `bt_cascade/` / `bt_cascade_focal/` | cascade phases |
| `bt_cascade_cls_recall/` | freeze SEG + recall-first cls / sens-floor τ |
| `s5_gate_25d/` | freeze S5 + 2.5D presence gate |
| `s5_fuse_reject/` | dual × 2.5D product fuse / rejector |
| `s5_gate_25d_hardneg/` | hardneg finetune (failed) |
| `s5_empty_rejector_v3/` | 25d scratch + MLP rej |
| `s5_wprod_mass_filter/` | wprod + min-mass empty-FP cut |
| `../RESULTS.md` | older skip-only / ablation summary |

Unused root train/eval scripts archived under `../../old/unused_bp_root/`.

---

*Last updated: 2026-10-06 — wprod+mass: ne 0.601 / fp 0.305 (best vs dual).*
