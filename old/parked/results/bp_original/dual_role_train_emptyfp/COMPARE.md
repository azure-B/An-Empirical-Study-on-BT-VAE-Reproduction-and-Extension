# Dual-role retrain (S5 recon + cls=emptyfp + head)

S5 skip decoder — frozen recon / SEG mask
plain no-skip decoder (init emptyfp) + LesionPresenceHead(μ) — presence
keep S5 mask iff P(head)>=tau else blank

best VAL AUROC (ckpt)=0.7994

## TEST
- TEST_S5_raw: ne=0.6637 fp=0.790
- TEST_plain_raw: ne=0.5804 fp=0.380
- TEST_gate@plainFloor_0.11: ne=0.5498 fp=0.278 auroc=0.7882
- TEST_gate@0.20: ne=0.5160 fp=0.225 auroc=0.7882
- TEST_gate@0.30: ne=0.4912 fp=0.198 auroc=0.7882
- TEST_gate@0.50: ne=0.4570 fp=0.153 auroc=0.7882
- TEST_gate@bal_0.16: ne=0.5271 fp=0.243 auroc=0.7882