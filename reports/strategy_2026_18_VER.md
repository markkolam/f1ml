# Strategy briefing: VER, 2026 round 18

Generated 2026-11-08 13:42 UTC.

- **grid**: P2
- **objective**: podium (in the podium fight (P(podium)))
- **baseline**: P(win) 0.18, P(podium) 0.62, expected position 4.7
- **track inputs**: 56 laps, green pit loss 23.3 s, deg S/M/H [0.049, 0.06, 0.053] s/lap, 50% pass threshold 1.83 s/lap

## Plans

Best plan for **VER** on **podium**: **H-S@22**, best in 279 of 1501 simulated races (19%): close: the driver is the casting vote. `vs_best` is the paired difference from the best plan (same simulated races), with its standard error.

| plan | stops | objective | vs_best | vs_best_se | votes | vote_share | exp_points | p_win | p_podium | p_top10 | mean_pos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H-S@22 | 1 | 0.743 | 0.000 | 0.000 | 279 | 0.186 | 16.553 | 0.290 | 0.743 | 0.900 | 4.215 |
| S-H@26 | 1 | 0.700 | -0.043 | 0.009 | 82 | 0.055 | 15.838 | 0.243 | 0.700 | 0.900 | 4.404 |
| M-S@24 | 1 | 0.697 | -0.047 | 0.008 | 156 | 0.104 | 15.777 | 0.235 | 0.697 | 0.900 | 4.413 |
| H-S@26 | 1 | 0.691 | -0.052 | 0.009 | 273 | 0.182 | 15.673 | 0.231 | 0.691 | 0.900 | 4.447 |
| S-H@30 | 1 | 0.648 | -0.095 | 0.011 | 165 | 0.110 | 15.304 | 0.215 | 0.648 | 0.900 | 4.552 |
| S-M@28 | 1 | 0.647 | -0.097 | 0.010 | 37 | 0.024 | 15.176 | 0.203 | 0.647 | 0.900 | 4.581 |
| H-S@30 | 1 | 0.611 | -0.132 | 0.012 | 195 | 0.130 | 14.733 | 0.171 | 0.611 | 0.900 | 4.691 |
| S-M@32 | 1 | 0.603 | -0.140 | 0.012 | 47 | 0.031 | 14.663 | 0.182 | 0.603 | 0.900 | 4.743 |
| M-H@26 | 1 | 0.593 | -0.151 | 0.011 | 14 | 0.009 | 14.261 | 0.154 | 0.593 | 0.900 | 4.853 |
| S-H@34 | 1 | 0.591 | -0.153 | 0.012 | 187 | 0.125 | 14.589 | 0.180 | 0.591 | 0.900 | 4.770 |
| S-M@36 | 1 | 0.541 | -0.203 | 0.013 | 53 | 0.035 | 13.944 | 0.152 | 0.541 | 0.900 | 4.975 |
| H-M@30 | 1 | 0.529 | -0.215 | 0.012 | 13 | 0.008 | 13.523 | 0.115 | 0.529 | 0.900 | 5.067 |

## When does another stop pay?

| index | from_stops | to_stops | medium_deg_s_per_lap | deg_multiplier | plan_after |
|---|---|---|---|---|---|
| 0 | 1 | 2 | 0.099 | 1.640 | S-S-H@19,39 |
| 1 | 2 | 3 | 0.206 | 3.420 | S-S-S-H@14,28,43 |

## If the tyre model is wrong

Regret = objective lost against the best plan in that tyre world (0 = best there).

| plan | all compounds deg alike | as fitted | higher deg (x1.3) | lower deg (x0.7) | soft falls off, hard holds | max_regret |
|---|---|---|---|---|---|---|
| S-H@26 | 0.019 | 0.009 | 0.049 | 0.028 | 0.000 | 0.049 |
| H-S@26 | 0.029 | 0.035 | 0.055 | 0.044 | 0.044 | 0.055 |
| S-H@30 | 0.072 | 0.069 | 0.096 | 0.047 | 0.091 | 0.096 |
| H-S@22 | 0.000 | 0.000 | 0.000 | 0.000 | 0.141 | 0.141 |
| M-S@24 | 0.004 | 0.041 | 0.064 | 0.029 | 0.197 | 0.197 |

## Safety Car on every lap

Share of Safety Car laps on which each plan was best, and its objective averaged over them.

| plan | best_on_share | mean_objective | worst_objective |
|---|---|---|---|
| H-S@22 | 0.765 | 0.711 | 0.335 |
| H-S@26 | 0.157 | 0.692 | 0.435 |
| M-S@24 | 0.059 | 0.685 | 0.380 |
| S-H@26 | 0.020 | 0.671 | 0.395 |

## Rivals' best responses

| driver | round | objective | plan | value | vote_share | changed_after_round_1 | rounds_run |
|---|---|---|---|---|---|---|---|
| ANT | 3 | points | M-S@24 | 19.578 | 0.181 | True | 3 |
| VER | 3 | points | M-S@24 | 15.138 | 0.212 | False | 3 |
| LEC | 3 | points | M-S@24 | 12.730 | 0.220 | True | 3 |
| HAD | 3 | points | M-S@24 | 13.448 | 0.229 | False | 3 |
| RUS | 3 | points | S-H@30 | 10.872 | 0.256 | False | 3 |
| HAM | 3 | points | S-H@30 | 9.725 | 0.235 | False | 3 |

## Teammate split (top pairs)

| index | VER | HAD | split | team_points | p_team_win | p_team_podium |
|---|---|---|---|---|---|---|
| 0 | H-S@22 | S-H@26 | True | 29.940 | 0.389 | 0.913 |
| 1 | H-S@22 | M-S@24 | True | 29.796 | 0.377 | 0.909 |
| 2 | H-S@22 | H-S@26 | False | 29.760 | 0.372 | 0.908 |
| 3 | S-H@26 | H-S@22 | True | 29.725 | 0.383 | 0.909 |
| 4 | H-S@22 | H-S@22 | False | 29.671 | 0.377 | 0.893 |
| 5 | M-S@24 | S-H@26 | True | 29.604 | 0.365 | 0.900 |
| 6 | H-S@26 | H-S@22 | False | 29.581 | 0.364 | 0.905 |
| 7 | S-H@26 | M-S@24 | True | 29.576 | 0.369 | 0.905 |
