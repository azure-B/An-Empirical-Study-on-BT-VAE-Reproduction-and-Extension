S5 concat training stopped at epoch 25 (user request), then region-seed experiment started.

- none TEST: `none/eval_now.json`
- S5 logs: `train_s5.log`
- Region seed: `../bp_region_seed/` (stopped ~ep 30, best val 0.4215 @23)
- none + encoder freeze: `none_freeze/` (stopped ~ep 37; empty FP worse than none)
- none + presence gate: `none_presence/` (frozen none SEG + slice presence head)
- neighbor gate (no train): `none_neighbor/`
- BT spatial proto (frozen BT μ 8×8 + 16×16): `bt_proto/`
- none empty-FP fine-tune: `none_emptyfp/`
- S5 + empty-FP fine-tune: `s5_emptyfp/`
