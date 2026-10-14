# Dual-role retrain (S5 recon + cls=none + head)

S5 skip decoder — frozen recon / SEG mask
plain no-skip decoder (init none) + LesionPresenceHead(μ) — presence
keep S5 mask iff P(head)>=tau else blank

best VAL AUROC (ckpt)=0.8098

## TEST
- TEST_S5_raw: ne=0.6637 fp=0.790
- TEST_plain_raw: ne=0.5923 fp=0.407
- TEST_gate@plainFloor_0.22: ne=0.5995 fp=0.383 auroc=0.7820
- TEST_gate@0.20: ne=0.6184 fp=0.428 auroc=0.7820
- TEST_gate@0.30: ne=0.5456 fp=0.266 auroc=0.7820
- TEST_gate@0.50: ne=0.4331 fp=0.156 auroc=0.7820
- TEST_gate@bal_0.22: ne=0.5995 fp=0.383 auroc=0.7820