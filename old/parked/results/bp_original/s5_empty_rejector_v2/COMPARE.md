# Empty rejector v2 (hardneg MLP)

train on nonempty ∪ S5-empty-FP · hardAUC=0.9999
target: ne≥0.6 fp<0.335

REF dual@0.22 ne=0.5995 fp=0.383
### rejector τ=0.47 → ne=0.5963 fp=0.335 beats_prev=False
### dual τ=0.28 → ne=0.5541 fp=0.284 beats_prev=False
### prod_dual_25 τ=0.03 → ne=0.6062 fp=0.356 beats_prev=False
### prod_dual_rej τ=0.13 → ne=0.5662 fp=0.299 beats_prev=False
### prod_25_rej τ=0.05 → ne=0.5969 fp=0.380 beats_prev=False
### prod3 τ=0.01 → ne=0.6119 fp=0.371 beats_prev=False
### min_dual_rej τ=0.28 → ne=0.5541 fp=0.284 beats_prev=False
### wprod_prev τ=0.01 → ne=0.6013 fp=0.335 beats_prev=False
### wprod_rej τ=0.01 → ne=0.5882 fp=0.320 beats_prev=False

## Champion
- TEST_wprod_prev: ne=0.6013 fp=0.335 beats_prev=False
