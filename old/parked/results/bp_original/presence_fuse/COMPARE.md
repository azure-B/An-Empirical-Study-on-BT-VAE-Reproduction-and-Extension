# Presence fuse (imgcls × SEG / cascade)

Primary: T/F accuracy · Secondary: gated ne / empty FP

- **img_only** @acc τ=0.88: acc=0.707 sens=0.552 spec=0.814 F1=0.606 | ne=0.420 empty_fp=0.177 TP/TN/FP/FN=127/272/62/103
- **seg_max** @acc τ=0.99: acc=0.640 sens=0.839 spec=0.503 F1=0.655 | ne=0.622 empty_fp=0.497 TP/TN/FP/FN=193/168/166/37
- **cascade_cls** @acc τ=0.75: acc=0.711 sens=0.513 spec=0.847 F1=0.591 | ne=0.386 empty_fp=0.150 TP/TN/FP/FN=118/283/51/112
- **mean_img_seg** @acc τ=0.94: acc=0.709 sens=0.543 spec=0.823 F1=0.604 | ne=0.416 empty_fp=0.177 TP/TN/FP/FN=125/275/59/105
- **max_img_seg** @acc τ=0.99: acc=0.637 sens=0.852 spec=0.488 F1=0.657 | ne=0.630 empty_fp=0.506 TP/TN/FP/FN=196/163/171/34
- **prod_img_seg** @acc τ=0.88: acc=0.711 sens=0.548 spec=0.823 F1=0.607 | ne=0.419 empty_fp=0.177 TP/TN/FP/FN=126/275/59/104
- **mean_img_cas** @acc τ=0.75: acc=0.711 sens=0.539 spec=0.829 F1=0.603 | ne=0.416 empty_fp=0.165 TP/TN/FP/FN=124/277/57/106
- **max_img_cas** @acc τ=0.88: acc=0.711 sens=0.587 spec=0.796 F1=0.624 | ne=0.438 empty_fp=0.195 TP/TN/FP/FN=135/266/68/95