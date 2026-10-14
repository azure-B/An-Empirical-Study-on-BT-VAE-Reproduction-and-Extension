# Active code — final S5 × wprod+mass (+ COVID twin)

Everything else for experiments lives under `old/` (esp. `old/not_final_s5_wprod/`).

## Models
| File | Role |
|---|---|
| `models_bp_skip.py` | S5 SkipVAE (SEG) |
| `models_dual_role.py` | S5 line + plain line + Head(μ) |
| `models_lesion_presence.py` | LesionPresenceHead |
| `models_presence_25d.py` | 2.5D ResNet18 presence (messy wprod) |
| `models_wprod_1enc.py` | **1enc wprod**: S5 + Head_μ × Head_ctx(3μ) |
| `models_bt_cascade.py` | 1enc·2dec cascade (amax presence) |
| `models_bt_cascade_cls.py` | 1enc·2dec + **Head on cls** (TF/FP train) |
| `original/models1.py` | no-skip OrigVAE (via `train_original_bp`) |

## Train / eval
| File | Role |
|---|---|
| `data_load_BT.py` | loaders |
| `train_original_bp.py` | BT + none/S5 SEG (BP) |
| `train_dual_role.py` | dual μ-head on frozen S5 |
| `train_s5_gate_25d.py` | Presence25D |
| `train_lesion_presence.py` | metrics helpers (+ legacy presence) |
| `eval_s5_fuse_reject.py` | dual×25d score collect / fuse |
| `eval_mass_filter.py` | **final** messy wprod + min-mass gate |
| `train_wprod_1enc.py` | **1enc compress**: train heads + wprod+mass VAL→TEST |
| `train_1enc2dec_head_tffp.py` | **1enc2dec+Head**: TF/FP hard-weight + τ/mass gate |

## COVID (same structure, isolated dirs)
| File | Role |
|---|---|
| `covid_config.py` | `results/covid_pipeline/`, `checkpoints_covid/` |
| `train_covid_seg.py` | BT + none + S5 on COVID |
| `train_covid_gate.py` | 25d + μ-head + wprod+mass |
| `run_covid_gate_when_ready.py` | waiter |

## Results
- BP final metrics: `results/bp_original/FINAL_MODEL.md`
- COVID: `results/covid_pipeline/` (do not mix ckpts with BP)
