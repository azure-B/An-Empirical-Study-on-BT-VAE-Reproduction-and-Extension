# 1 Encoder · 2 Decoder (paper-clean)

## Structure
- **1× Encoder** (BT, shared μ + skips)
- **Decoder_seg**: S5 concat skips → lesion mask
- **Cls branch**: either `Decoder_cls(μ)` (amax→p) or `Head(μ)+Head(skips)`
- **Gate**: τ on p + optional min pred-mass

No second VAE encoder, no external ResNet — unlike wprod+mass.
