# BT cascade + Dice/Focal (small lesion)

Cls/τ: focal BCE · SEG: Dice + Focal + size↑ for small lesions
learned τ_prob≈0.6536 (logit=0.6351)
CLS AUROC=0.7440 | SEG VAL ne≈0.6928

## Presence T/F (gate)
### VAL_learned_tau
- TP=156 TN=230 FP=106 FN=71
- sens=0.687 spec=0.685 prec=0.595 F1=0.638 AUROC=0.744
- SEG@gate: ne=0.5183 empty_fp=0.312

### TEST_learned_tau
- TP=157 TN=235 FP=99 FN=73
- sens=0.683 spec=0.704 prec=0.613 F1=0.646 AUROC=0.757
- SEG@gate: ne=0.5214 empty_fp=0.281

## SEG / other
- TEST_seg_raw: ne=0.7156 fp=0.913
- VAL_learned_tau: ne=0.5183 fp=0.312 τ=0.654
- TEST_learned_tau: ne=0.5214 fp=0.281 τ=0.654