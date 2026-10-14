# S5 × 2.5D hardneg + dual fuse

init=cls25d_best_auroc.pt · mix ne/fp/tn=0.45/0.4/0.15
best VAL ne|emptyFP AUROC=0.7049
target: ne≥0.6 and fp < 0.335 (prev wprod)

REF dual@0.22 ne=0.5995 fp=0.383
### dual τ=0.28 → ne=0.5541 fp=0.284 beats_prev=False
### p25d_hn τ=0.05 → ne=0.5617 fp=0.362 beats_prev=False
### mean τ=0.24 → ne=0.5616 fp=0.305 beats_prev=False
### min τ=0.05 → ne=0.5592 fp=0.353 beats_prev=False
### prod τ=0.01 → ne=0.5687 fp=0.380 beats_prev=False

## Champion
- none
- beats prev wprod fp<0.335: False
