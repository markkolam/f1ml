# Backtest report

Generated 2026-11-08 15:01 UTC. Walk-forward: each race is predicted by a model trained only on earlier races (plus that weekend's practice, qualifying and sprint).

Features: grid_pos, pit_lane_start, quali_pos, quali_gap_pct, teammate_quali_gap_pct, longrun_gap_pct, n_long_runs, is_sprint_weekend, sprint_pos, sprint_pace_pct, race_pace_ewm, team_race_pace_ewm, quali_gap_ewm, finish_ewm, points_ewm, team_dnf_rate, n_prev_races, track_grid_finish_rho, track_mean_places_changed, track_sc_share, track_n_prev, races_into_era.

## Overall

| model | races | spearman | winner_logloss | brier_win | brier_podium | winner_hit | top3_overlap | mae_pos |
|---|---|---|---|---|---|---|---|---|
| grid | 66 | 0.683 | 1.367 | 0.029 | 0.067 | 0.591 | 2.045 | 3.192 |
| lgbm | 66 | 0.706 | 1.311 | 0.030 | 0.064 | 0.561 | 2.061 | 2.976 |
| lgbm_pure | 66 | 0.696 | 1.434 | 0.033 | 0.067 | 0.500 | 2.030 | 3.019 |
| quali | 66 | 0.684 | 1.347 | 0.029 | 0.069 | 0.606 | 1.955 | 3.185 |

## By season

| model | season | races | spearman | winner_logloss | brier_win | brier_podium | winner_hit | top3_overlap | mae_pos |
|---|---|---|---|---|---|---|---|---|---|
| grid | 2024 | 23 | 0.735 | 1.702 | 0.038 | 0.073 | 0.435 | 2.000 | 2.908 |
| grid | 2025 | 24 | 0.652 | 1.129 | 0.026 | 0.062 | 0.667 | 2.250 | 3.170 |
| grid | 2026 | 19 | 0.657 | 1.261 | 0.023 | 0.066 | 0.684 | 1.842 | 3.565 |
| lgbm | 2024 | 23 | 0.761 | 1.722 | 0.038 | 0.073 | 0.435 | 1.957 | 2.626 |
| lgbm | 2025 | 24 | 0.668 | 1.035 | 0.027 | 0.056 | 0.583 | 2.250 | 3.070 |
| lgbm | 2026 | 19 | 0.687 | 1.163 | 0.024 | 0.062 | 0.684 | 1.947 | 3.282 |
| lgbm_pure | 2024 | 23 | 0.751 | 1.823 | 0.041 | 0.078 | 0.348 | 1.913 | 2.671 |
| lgbm_pure | 2025 | 24 | 0.657 | 1.167 | 0.030 | 0.059 | 0.583 | 2.167 | 3.130 |
| lgbm_pure | 2026 | 19 | 0.680 | 1.302 | 0.028 | 0.064 | 0.579 | 2.000 | 3.301 |
| quali | 2024 | 23 | 0.753 | 1.639 | 0.036 | 0.077 | 0.478 | 1.870 | 2.875 |
| quali | 2025 | 24 | 0.614 | 1.232 | 0.028 | 0.064 | 0.667 | 2.167 | 3.262 |
| quali | 2026 | 19 | 0.688 | 1.140 | 0.023 | 0.067 | 0.684 | 1.789 | 3.463 |

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
| grid_pos | 35.200 |
| quali_pos | 20.500 |
| quali_gap_pct | 13.200 |
| teammate_quali_gap_pct | 5.100 |
| quali_gap_ewm | 4.800 |
| race_pace_ewm | 3.600 |
| points_ewm | 3.200 |
| team_race_pace_ewm | 2.600 |
| n_prev_races | 2.300 |
| track_grid_finish_rho | 1.700 |
| track_mean_places_changed | 1.600 |
| finish_ewm | 1.400 |
| team_dnf_rate | 1.100 |
| races_into_era | 1.000 |
| sprint_pace_pct | 1.000 |
| longrun_gap_pct | 0.500 |
| track_sc_share | 0.500 |
| sprint_pos | 0.200 |
| n_long_runs | 0.200 |
| track_n_prev | 0.100 |
| is_sprint_weekend | 0.000 |
| pit_lane_start | 0.000 |
