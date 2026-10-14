# BT-VAE Results Log

_Auto-updated: 2026-09-22 12:39:24_

## Active focus

AG×γ + (1−γ)×ungated skip blend, γ_init=0.01 (COVID).

## Current run - Attention Gate × gamma blend γ₀=0.01 (COVID)

| Item | Value |
|---|---|
| Result dir | `results/skiponly_covid_attn_gamma01/` |
| Model | `attn_skip_gamma`: x+=γ·AG+(1−γ)·ungated, γ_init=0.01 |
| Protocol | reuse BT encoder → SEG ≤400, early-stop patience=40 |
| Phase | **done** |
| Epoch | **95** |
| Updated at | 2026-09-22 12:39:10 |
| Best val Dice | 0.8303 |
| SEG bad epochs | 40 |
| Last val Dice | 0.8010 |
| Last val IoU | 0.7497 |
| Last val loss | 0.3118 |
| History epochs | 95 (last=95) |
| History best val Dice | 0.8303 @ ep 55 |

### TEST (batch-avg)

- Dice `0.7663` | IoU `0.7033` | loss `0.2976`

### TEST (per-image KPI)

| Metric | Value |
|---|---|
| n / empty GT | 94 / 40 |
| mean Dice (all) | 0.7718 |
| median Dice (all) | 0.9095 |
| **mean Dice (nonempty)** | **0.7694** |
| **median Dice (nonempty)** | **0.8278** |
| mean IoU (all) | 0.7103 |

### Report snapshot

- best val Dice `0.8303` @ ep `55`
- last epoch `95` val Dice `0.8010`
- SEG best ckpt ep `55` dice `0.8303`
- best gammas (1=AG,0=additive): `{'g4': 0.3932265341281891, 'g3': 0.48460373282432556, 'g2': 0.9967834949493408, 'g1': 0.9842025637626648, 'g0': 0.0}`
- status gammas: `{'g4': 0.3932265341281891, 'g3': 0.48460373282432556, 'g2': 0.9967834949493408, 'g1': 0.9842025637626648, 'g0': 0.0}`

### References

- original/ (no skips): mean~0.76, median~0.84
- skiponly_covid_base32 (additive skips): nonempty mean `0.7829`, median `0.8391` (batch TEST Dice `0.7629`)
- skiponly_covid_attn (pure AG): nonempty mean `0.7787`, median `0.8299` (batch TEST Dice `0.7729`)

## Archived experiments

| Experiment | Dataset / notes | TEST Dice | TEST IoU | Extra KPI |
|---|---|---:|---:|---|
| `skiponly_covid_attn_gamma` | COVID AG×γ blend γ_init=0.5, SEG ES patience=40 | 0.7718 | 0.7066 | nonempty mean/med 0.7766/0.8341 |
| `skiponly_covid_attn` | COVID pure AG skips, base32 lat1024, ep400 no ES | 0.7729 | 0.7060 | nonempty mean/med 0.7787/0.8299 |
| `skiponly_covid_base32` | COVID additive skips, base32 lat1024, ~ep400+ | 0.7629 | 0.6957 | nonempty mean/med 0.7829/0.8391 |
| `skiponly_lat1024_base32` | BP SkipOnly base32 lat1024 | 0.5995 | 0.5630 | - |
| `skiponly_lat1024` | BP SkipOnly (earlier) | 0.4524 | 0.4031 | - |
| `archive_skip_base16` | SkipOnly base16 | 0.4524 | 0.4031 | - |
| `archive_lat512` | MS-GSF / lat512 archive | 0.6501 | 0.6170 | - |
| `archive_base16` | earlier base16 | 0.3680 | 0.3178 | - |
| `ms_gsf_bt_vae` | MS-GSF (unstable late) | 0.5420 | 0.4906 | - |

## Layout

```
results/
  RESULTS.md                 # this file (live)
  skiponly_covid_attn_gamma01/ # current AG×gamma γ_init=0.01
  skiponly_covid_attn_gamma/   # prior γ_init=0.5 blend
  skiponly_covid_attn/         # prior pure-AG COVID run
  archive/                   # prior experiments
```

## Changelog

- 2026-09-22 12:39:24: refreshed from `run_status.json` / reports
