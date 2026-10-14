# BP train_30 val / test (re-eval 2026-09-28)

Same loaders, thr=0.5, empty-empty Dice=1. Raw: `COMPARE.json`.

## TEST (n=564, 230 nonempty / 334 empty)

| Experiment | Dice | Nonempty | Empty FP | Nonempty P | Nonempty R |
|---|---:|---:|---:|---:|---:|
| none | 0.599 | 0.572 | 0.383 | 0.829 | 0.551 |
| s5 skip | 0.396 | 0.663 | 0.787 | 0.777 | 0.650 |
| region seed | 0.536 | 0.650 | 0.542 | 0.806 | 0.639 |
| encoder freeze | 0.398 | 0.677 | 0.793 | 0.727 | 0.685 |
| presence τ=0.20 | 0.650 | 0.533 | 0.269 | 0.860 | 0.514 |
| neighbor isolated | 0.642 | 0.513 | 0.269 | — | — |
| neighbor small | 0.625 | 0.568 | 0.335 | — | — |
| BT proto | 0.243 | 0.478 | 0.919 | 0.576 | 0.472 |
| empty-FP FT | 0.675 | 0.437 | 0.162 | 0.896 | 0.428 |

## VAL (n=563, 227 nonempty / 336 empty)

| Experiment | Dice | Nonempty | Empty FP |
|---|---:|---:|---:|
| none | 0.608 | 0.644 | 0.417 |
| s5 skip | 0.419 | 0.699 | 0.771 |
| region seed | 0.538 | 0.661 | 0.545 |
| encoder freeze | 0.381 | 0.650 | 0.801 |
| presence τ=0.20 | 0.665 | 0.601 | 0.292 |
| neighbor isolated | 0.629 | 0.560 | 0.324 |
| neighbor small | 0.622 | 0.644 | 0.393 |
| BT proto | 0.247 | 0.490 | 0.917 |
| empty-FP FT | 0.671 | 0.471 | 0.193 |
