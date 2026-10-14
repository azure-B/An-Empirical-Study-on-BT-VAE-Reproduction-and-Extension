# BT cascade + Dice/Focal

```
x → BT Encoder (frozen) → μ (+ skips)
      ├─ μ → Decoder_cls → presence − τ   # focal BCE
      └─ μ + skips → Decoder_seg         # Dice + Focal + 1/area^p

infer: if decision ≥ 0 then SEG else blank
```

γ=2.0 α_seg=0.75 size_power=0.5
BT ckpt: `D:\Desktop\BT_VAE\checkpoints_bp_original\bt_encoder_best.pt`