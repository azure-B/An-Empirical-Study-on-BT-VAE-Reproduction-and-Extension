# Presence image classifier (step1)

ResNet18 on FLAIR only · no VAE μ
best VAL AUROC=0.7672

## TEST T/F (primary)
- TEST_thr0.50: acc=0.688 sens=0.726 spec=0.662 F1=0.655 TP/TN/FP/FN=167/221/113/63 τ=0.50
- TEST_acc: acc=0.707 sens=0.552 spec=0.814 F1=0.606 TP/TN/FP/FN=127/272/62/103 τ=0.88
- TEST_f1: acc=0.686 sens=0.730 spec=0.656 F1=0.655 TP/TN/FP/FN=168/219/115/62 τ=0.49
- TEST_bal: acc=0.707 sens=0.552 spec=0.814 F1=0.606 TP/TN/FP/FN=127/272/62/103 τ=0.88