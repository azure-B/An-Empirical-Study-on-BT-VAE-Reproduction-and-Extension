# Rejector v3 + 25d scratch + neighbor

25d AUROC=0.7573 · rej VAL hardAUC=0.7774
target ne≥0.6 fp<0.335

REF dual@0.22 ne=0.5969 fp=0.384
### dual τ=0.28 → ne=0.5517 fp=0.285 beats_prev=False
### p25d τ=0.58 → ne=0.5752 fp=0.384 beats_prev=False
### prod τ=0.17 → ne=0.5671 fp=0.273 beats_prev=False
### wprod τ=0.08 → ne=0.5730 fp=0.273 beats_prev=False
### wprod_rej τ=0.06 → ne=0.5709 fp=0.276 beats_prev=False
### prod_rej τ=0.13 → ne=0.5713 fp=0.285 beats_prev=False
### rejector τ=0.49 → ne=0.5737 fp=0.273 beats_prev=False
### wprod_neigh τ=0.03 → ne=0.5587 fp=0.378 beats_prev=False
### wprod_neigh_rej τ=0.02 → ne=0.5683 fp=0.381 beats_prev=False

## Champion
- TEST_wprod_neigh_rej: ne=0.5683 fp=0.381 beats_prev=False
