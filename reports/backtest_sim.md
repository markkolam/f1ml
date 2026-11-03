# Backtest report

Generated 2026-11-03 19:59 UTC. Walk-forward: each race is predicted by a model trained only on earlier races (plus that weekend's practice, qualifying and sprint).

Features: grid_pos, pit_lane_start, quali_pos, quali_gap_pct, teammate_quali_gap_pct, longrun_gap_pct, n_long_runs, is_sprint_weekend, sprint_pos, sprint_pace_pct, race_pace_ewm, team_race_pace_ewm, quali_gap_ewm, finish_ewm, points_ewm, team_dnf_rate, n_prev_races, track_grid_finish_rho, track_mean_places_changed, track_sc_share, track_n_prev, races_into_era.

## Overall

| model | races | spearman | winner_logloss | brier_win | brier_podium | winner_hit | top3_overlap | mae_pos |
|---|---|---|---|---|---|---|---|---|
| grid | 66 | 0.683 | 1.367 | 0.029 | 0.067 | 0.591 | 2.045 | 3.192 |
| lgbm | 66 | 0.705 | 1.317 | 0.030 | 0.064 | 0.561 | 2.015 | 2.970 |
| sim | 66 | 0.686 | 1.369 | 0.029 | 0.064 | 0.561 | 2.015 | 3.143 |

## By season

| model | season | races | spearman | winner_logloss | brier_win | brier_podium | winner_hit | top3_overlap | mae_pos |
|---|---|---|---|---|---|---|---|---|---|
| grid | 2024 | 23 | 0.735 | 1.702 | 0.038 | 0.073 | 0.435 | 2.000 | 2.908 |
| grid | 2025 | 24 | 0.652 | 1.129 | 0.026 | 0.062 | 0.667 | 2.250 | 3.170 |
| grid | 2026 | 19 | 0.657 | 1.261 | 0.023 | 0.066 | 0.684 | 1.842 | 3.565 |
| lgbm | 2024 | 23 | 0.755 | 1.741 | 0.039 | 0.074 | 0.391 | 1.870 | 2.622 |
| lgbm | 2025 | 24 | 0.669 | 1.041 | 0.027 | 0.056 | 0.625 | 2.167 | 3.063 |
| lgbm | 2026 | 19 | 0.690 | 1.152 | 0.024 | 0.062 | 0.684 | 2.000 | 3.273 |
| sim | 2024 | 23 | 0.763 | 1.968 | 0.036 | 0.070 | 0.522 | 1.957 | 2.715 |
| sim | 2025 | 24 | 0.643 | 0.970 | 0.027 | 0.058 | 0.542 | 2.167 | 3.244 |
| sim | 2026 | 19 | 0.646 | 1.148 | 0.024 | 0.064 | 0.632 | 1.895 | 3.533 |

## Metrics

- **spearman**: rank correlation, predicted vs actual order (higher better)
- **winner_logloss**: -log P(actual winner) (lower better; uniform guess over 20 cars = 3.00)
- **brier_win**: Brier score of P(win) (lower better)
- **brier_podium**: Brier score of P(podium) (lower better)
- **winner_hit**: share of races the favourite won
- **top3_overlap**: predicted top-3 drivers who finished top-3 (0-3)
- **mae_pos**: mean |expected position - actual| (lower better)

## Feature importance (final model, % of total gain)

| index | gain_% |
|---|---|
| grid_pos | 32.700 |
| quali_pos | 20.600 |
| quali_gap_pct | 17.200 |
| teammate_quali_gap_pct | 4.700 |
| points_ewm | 4.000 |
| quali_gap_ewm | 3.900 |
| race_pace_ewm | 3.000 |
| n_prev_races | 2.100 |
| finish_ewm | 2.000 |
| team_race_pace_ewm | 1.800 |
| track_mean_places_changed | 1.600 |
| track_grid_finish_rho | 1.300 |
| team_dnf_rate | 1.200 |
| sprint_pace_pct | 1.200 |
| races_into_era | 1.200 |
| longrun_gap_pct | 0.400 |
| track_sc_share | 0.400 |
| n_long_runs | 0.200 |
| track_n_prev | 0.200 |
| sprint_pos | 0.100 |
| is_sprint_weekend | 0.000 |
| pit_lane_start | 0.000 |
