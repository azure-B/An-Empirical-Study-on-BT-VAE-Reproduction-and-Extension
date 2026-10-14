# COVID Dice vs labeled-data ratio

Train subset only. Val/test = full covid_8_1_1. BT encoder shared.
TEST dice = mean over slices, deterministic μ.
Each section is its own curve. A later model is a new section.

## none

| ratio | TEST dice | nonempty | empty FP | best ep |
|---:|---:|---:|---:|---:|
| 0.1 | 0.5046 | 0.4895 | 0.475 | 121 |
| 0.3 | 0.6399 | 0.5953 | 0.300 | 136 |
| 0.5 | 0.6864 | 0.6208 | 0.225 | 118 |
| 0.7 | 0.7903 | 0.6536 | 0.025 | 240 |
| 0.9 | 0.8097 | 0.6688 | 0.000 | 202 |

## none_cls

| ratio | TEST dice | nonempty | empty FP | best ep |
|---:|---:|---:|---:|---:|
| 0.1 | 0.6934 | 0.4663 | 0.000 | 19 |
| 0.5 | 0.7692 | 0.5982 | 0.000 | 10 |

## s5

| ratio | TEST dice | nonempty | empty FP | best ep |
|---:|---:|---:|---:|---:|
| 0.1 | 0.6627 | 0.7092 | 0.400 | 79 |
| 0.3 | 0.7449 | 0.7226 | 0.225 | 133 |
| 0.5 | 0.7925 | 0.7684 | 0.175 | 108 |
| 0.7 | 0.8058 | 0.7546 | 0.125 | 124 |
| 0.9 | 0.8185 | 0.7766 | 0.125 | 118 |

## s5_cls

| ratio | TEST dice | nonempty | empty FP | best ep |
|---:|---:|---:|---:|---:|
| 0.1 | 0.7685 | 0.6895 | 0.125 | 19 |
| 0.3 | 0.8011 | 0.6908 | 0.050 | 36 |
| 0.5 | 0.7659 | 0.6295 | 0.050 | 17 |
| 0.7 | 0.8299 | 0.7225 | 0.025 | 37 |
| 0.9 | 0.8428 | 0.7634 | 0.050 | 28 |
