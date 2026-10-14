# BT cascade + learnable τ

Primary view: presence T/F. SEG dice secondary.
learned τ_prob≈0.6146 (logit=0.4667)

## Presence T/F (gate)
### VAL_learned_tau
- TP=159 TN=216 FP=120 FN=68
- sens=0.700 spec=0.643 prec=0.570 F1=0.628 AUROC=0.730 acc=0.666
- SEG@gate: ne=0.5045 empty_fp=0.330

### TEST_learned_tau
- TP=155 TN=228 FP=106 FN=75
- sens=0.674 spec=0.683 prec=0.594 F1=0.631 AUROC=0.753 acc=0.679
- SEG@gate: ne=0.4972 empty_fp=0.308

## SEG / other
- TEST_seg_raw: ne=0.6950 fp=0.886
- VAL_learned_tau: ne=0.5045 fp=0.330 τ=0.615
- TEST_learned_tau: ne=0.4972 fp=0.308 τ=0.615