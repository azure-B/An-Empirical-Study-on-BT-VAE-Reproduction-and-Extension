# S5 fuse + empty rejector

Target: beat dual@0.22 (ne≥0.6, fp<0.383)
dual VAL AUROC=0.8098 | 25d=0.7557 | rejector=0.4014
REF dual@0.22 TEST ne=0.5995 fp=0.383

### dual (beat_dual) τ=0.28
- VAL ne=0.6025 fp=0.342
- TEST ne=0.5541 fp=0.284 sens=0.726 FN=63 beats_dual=False
### p25d (ne_floor_min_fp) τ=0.10
- VAL ne=0.6057 fp=0.435
- TEST ne=0.5969 fp=0.383 sens=0.826 FN=40 beats_dual=False
### mean (beat_dual) τ=0.30
- VAL ne=0.6027 fp=0.360
- TEST ne=0.5717 fp=0.302 sens=0.757 FN=56 beats_dual=False
### min (ne_floor_min_fp) τ=0.09
- VAL ne=0.6060 fp=0.414
- TEST ne=0.5946 fp=0.359 sens=0.813 FN=43 beats_dual=False
### prod (ne_floor_min_fp) τ=0.03
- VAL ne=0.6093 fp=0.405
- TEST ne=0.6062 fp=0.356 sens=0.822 FN=41 beats_dual=True
### max (beat_dual) τ=0.45
- VAL ne=0.6020 fp=0.366
- TEST ne=0.5723 fp=0.311 sens=0.765 FN=54 beats_dual=False
### rejector (ne_floor_min_fp) τ=0.41
- VAL ne=0.6700 fp=0.726
- TEST ne=0.6390 fp=0.737 sens=0.961 FN=9 beats_dual=False

## Beat dual
- TEST_prod

## Weighted product (empty-FP push)
### p_dual^1.5 * p_25d^1.3 (VAL pick=ne_floor_min_fp) τ=0.01
- TEST ne=0.6013 fp=0.335 sens=0.813 FN=43
- vs dual@0.22: ne +0.0018, fp -0.048
