# Race simulator: replay check (phase 4)

Generated 2026-11-03 19:56 UTC. Each finished dry race is rebuilt from its own fitted pace and tyre model, with the strategies, Safety Cars/VSCs, retirements and lap-1 order that actually happened; overtaking, dirty air and pit loss come from components fitted only on earlier races. Buscombe's standard: the model must reproduce the real race before it earns an opinion about a fake one.

Wet or mixed-condition races skipped: 6.

## Overall

| index | races | spearman | pos_in_10_90 | gap_mae_s | gap_in_10_90 | winner_correct |
|---|---|---|---|---|---|---|
| dry races | 58 | 0.900 | 0.703 | 10.066 | 0.579 | 0.759 |

## By season

| season | races | spearman | pos_in_10_90 | gap_mae_s | gap_in_10_90 | winner_correct |
|---|---|---|---|---|---|---|
| 2024.000 | 20.000 | 0.922 | 0.740 | 8.667 | 0.544 | 0.750 |
| 2025.000 | 21.000 | 0.853 | 0.688 | 9.406 | 0.628 | 0.762 |
| 2026.000 | 17.000 | 0.933 | 0.680 | 12.527 | 0.560 | 0.765 |

## Metrics

- **spearman**: rank correlation of simulated median position vs actual order
- **pos_in_10_90**: share of drivers whose actual position is inside the simulated 10-90% range (ideal ~0.8)
- **gap_mae_s**: mean |simulated median gap - actual gap| to the winner, lead-lap finishers (s)
- **gap_in_10_90**: share of lead-lap finishers whose actual gap is inside the simulated 10-90% range (ideal ~0.8)
- **winner_correct**: simulated favourite (best median position) actually won

## Largest gap errors

| season | round | spearman | pos_in_10_90 | gap_mae_s | gap_in_10_90 | winner_correct |
|---|---|---|---|---|---|---|
| 2026.000 | 9.000 | 0.732 | 0.364 | 82.566 | 0.059 | 1.000 |
| 2025.000 | 10.000 | 0.607 | 0.550 | 56.630 | 0.000 | 0.000 |
| 2024.000 | 11.000 | 0.690 | 0.350 | 29.370 | 0.000 | 0.000 |
| 2025.000 | 6.000 | 0.542 | 0.263 | 26.040 | 0.250 | 0.000 |
| 2026.000 | 14.000 | 0.977 | 0.682 | 18.483 | 0.167 | 0.000 |
| 2025.000 | 8.000 | 0.923 | 0.900 | 18.307 | 0.800 | 0.000 |
| 2024.000 | 17.000 | 0.610 | 0.300 | 14.872 | 0.455 | 0.000 |
| 2026.000 | 10.000 | 0.978 | 0.810 | 14.665 | 0.071 | 1.000 |
