# Cascade cls-recall (separate τ)

SEG frozen from cascade_focal · Head(μ)+Head(skips) · τ NOT in train loss
best VAL AUROC=0.6400 · pos_weight=3.0
TEST raw SEG: ne=0.7156 fp=0.913

## τ picks (VAL calibrate → TEST)
### sens-floor 0.95 → τ=0.58
- VAL: sens=0.952 empty_fp=0.860 ne=0.665
- TEST TF: TP=222 TN=39 FP=295 FN=8 acc=0.463 sens=0.965 spec=0.117 F1=0.594
- TEST SEG@gate: ne=0.6945 empty_fp=0.823

### sens-floor 0.90 → τ=0.62
- VAL: sens=0.903 empty_fp=0.771 ne=0.637
- TEST TF: TP=215 TN=65 FP=269 FN=15 acc=0.496 sens=0.935 spec=0.195 F1=0.602
- TEST SEG@gate: ne=0.6762 empty_fp=0.746

### sens-floor 0.85 → τ=0.63
- VAL: sens=0.885 empty_fp=0.711 ne=0.628
- TEST TF: TP=211 TN=98 FP=236 FN=19 acc=0.548 sens=0.917 spec=0.293 F1=0.623
- TEST SEG@gate: ne=0.6635 empty_fp=0.650

### sens-floor 0.80 → τ=0.64
- VAL: sens=0.824 empty_fp=0.613 ne=0.587
- TEST TF: TP=191 TN=125 FP=209 FN=39 acc=0.560 sens=0.830 spec=0.374 F1=0.606
- TEST SEG@gate: ne=0.6057 empty_fp=0.569

### F1 → τ=0.66
- TEST: acc=0.605 sens=0.722 spec=0.524 ne=0.5296 empty_fp=0.437

## Refs
- cascade_focal gated: ne=0.521 fp=0.281 acc=0.695 sens=0.683
- presence τ=0.20: ne=0.533 fp=0.269
- dual-role @0.22: ne=0.600 fp=0.383