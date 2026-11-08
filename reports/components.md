# Component models (phase 3)

Generated 2026-11-08 14:44 UTC from 2023-2026 race data (OpenF1).

## Against Ruth Buscombe's published figures

| quantity | buscombe | f1ml |
|---|---|---|
| Monza 2026 race deg (s/lap) | 0.03 | 0.034 |
| Hungary 2026 green pit loss (s) | 21.9 measured | 22.567 |
| Austin 2026 green pit loss (s) | 20.5-21.5 | 22.363 |
| Mexico 2026 green pit loss (s) | 20.5 | 23.455 |
| Madrid 2026 green pit loss (s) | ~24 (pre-race estimate) | 25.879 |
| Budapest 2026 pass attempts (<=1.2 s, green) | 225 | 221.000 |
| Budapest 2026 passes by next lap | 28 | 27.000 |
| Budapest 2026 50% pass threshold (s/lap) | 1.09 | 1.604 |
| Austin 2026 50% pass threshold (s/lap) | 0.5-0.7 | 1.422 |
| Singapore 50% pass threshold (s/lap) | 2.5 | 3.349 |
| Monaco 50% pass threshold (s/lap) | 4.5 | 7.790 |
| Verstappen pass-curve shift (s/lap) | ~0.2 better than field | 0.277 |
| Dirty-air cost, 2026 era mean (s/lap) | ~0.3 (Norris, Austin) | 0.376 |
| Mexico late-race neutralisation hazard (per lap) | 1.3% | 1.438 |
| Budapest 2026 lap-1 concertina (s) | 11.0 | 11.000 |

Notes: our pass thresholds use the expected (race-average) pace gap, which is smoother than a lap-by-lap gap, so they sit ~0.3-0.6 s above hers; track ordering agrees. Green pit losses run ~1-3 s above her figures: we count the whole in-lap and out-lap against cars that stayed out (cold-tyre out-lap included). VSC/SC pit losses are derived from the measured green loss with her ~0.52 ratio (not measurable from lap sums behind a Safety Car). The restart effect on neutralisation hazard came out negative in this data (no post-restart bump), unlike her flag model.

## Tyre: 2026 deg by track (s/lap per lap of tyre age, temperature-adjusted, pooled)

| track_id | HARD | MEDIUM | SOFT |
|---|---|---|---|
| albert_park | 0.042 | 0.015 | 0.053 |
| americas | 0.048 | 0.053 | 0.048 |
| baku | nan | 0.031 | 0.031 |
| catalunya | 0.084 | 0.149 | 0.070 |
| hungaroring | 0.062 | 0.031 | 0.053 |
| madring | 0.038 | 0.089 | 0.060 |
| marina_bay | nan | 0.050 | 0.060 |
| miami | 0.037 | 0.025 | 0.037 |
| monaco | 0.047 | 0.064 | 0.057 |
| monza | 0.035 | 0.037 | 0.043 |
| red_bull_ring | 0.056 | 0.085 | 0.055 |
| rodriguez | nan | 0.038 | 0.046 |
| shanghai | 0.036 | 0.125 | nan |
| silverstone | 0.045 | 0.014 | nan |
| spa | 0.041 | 0.009 | 0.077 |
| suzuka | 0.060 | 0.049 | nan |
| zandvoort | 0.043 | 0.057 | 0.049 |

Tyre parameters (temp_gamma = extra deg per lap of age per degree C of track temperature):

| param | value | se |
|---|---|---|
| temp_gamma | 0.000 | 0.000 |
| ref_temp:2022_ground_effect | 35.675 | nan |
| ref_temp:2026_active_aero | 41.556 | nan |

## Pit loss, 2026 (s)

| track_id | green | sc | vsc |
|---|---|---|---|
| albert_park | 23.500 | 12.200 | 12.200 |
| americas | 22.400 | 11.600 | 11.600 |
| baku | 23.700 | 12.300 | 12.300 |
| catalunya | 24.100 | 12.500 | 12.500 |
| hungaroring | 22.600 | 11.700 | 11.700 |
| madring | 25.900 | 13.500 | 13.500 |
| marina_bay | 23.500 | 12.200 | 12.200 |
| miami | 19.300 | 10.000 | 10.000 |
| monaco | 21.400 | 11.100 | 11.100 |
| monza | 24.700 | 12.900 | 12.900 |
| red_bull_ring | 21.600 | 11.200 | 11.200 |
| rodriguez | 23.500 | 12.200 | 12.200 |
| sepang | 25.700 | 13.400 | 13.400 |
| shanghai | 24.900 | 12.900 | 12.900 |
| silverstone | 22.700 | 11.800 | 11.800 |
| spa | 22.900 | 11.900 | 11.900 |
| suzuka | 23.300 | 12.100 | 12.100 |
| villeneuve | 25.400 | 13.200 | 13.200 |
| zandvoort | 20.100 | 10.400 | 10.400 |

## Overtaking, 2026: pass thresholds (s/lap of pace advantage)

| track_id | n_attempts | pass_rate | threshold_50 | threshold_90 |
|---|---|---|---|---|
| monza | 244 | 0.210 | 1.040 | 2.320 |
| miami | 140 | 0.160 | 1.160 | 2.200 |
| silverstone | 135 | 0.130 | 1.230 | 2.220 |
| villeneuve | 123 | 0.150 | 1.350 | 2.580 |
| red_bull_ring | 182 | 0.130 | 1.370 | 2.380 |
| americas | 126 | 0.230 | 1.420 | 2.530 |
| shanghai | 179 | 0.170 | 1.470 | 3.080 |
| rodriguez | 323 | 0.060 | 1.510 | 2.470 |
| suzuka | 355 | 0.100 | 1.520 | 2.930 |
| albert_park | 167 | 0.160 | 1.540 | 2.800 |
| catalunya | 149 | 0.130 | 1.540 | 2.700 |
| hungaroring | 221 | 0.120 | 1.600 | 2.540 |
| spa | 175 | 0.140 | 1.740 | 3.290 |
| zandvoort | 226 | 0.110 | 1.980 | 3.280 |
| baku | 189 | 0.110 | 2.350 | 4.320 |
| sepang | 155 | 0.170 | 2.450 | 5.170 |
| marina_bay | 183 | 0.100 | 3.350 | 5.750 |
| madring | 161 | 0.050 | 3.440 | 5.470 |
| monaco | 235 | 0.010 | 7.790 | 11.990 |

Driver pass-curve shifts (s/lap equivalent; positive = converts better than the field):

| driver | logit_shift | pace_equiv_s | n_attempts |
|---|---|---|---|
| VER | 0.552 | 0.277 | 432 |
| SAI | 0.364 | 0.182 | 803 |
| HAM | 0.311 | 0.156 | 942 |
| PER | 0.276 | 0.138 | 544 |
| RUS | 0.276 | 0.138 | 855 |
| LEC | 0.274 | 0.137 | 757 |
| STR | 0.243 | 0.122 | 922 |
| ALO | 0.203 | 0.102 | 880 |
| BEA | 0.163 | 0.082 | 622 |
| PIA | 0.141 | 0.071 | 740 |

## Neutralisation hazard

Coefficients (logit scale):

| param | value |
|---|---|
| intercept | -3.957 |
| lap1 | 2.138 |
| restart | -0.771 |
| pit_activity | 3.513 |
| late | 0.215 |
| race_frac | -0.782 |
| wet | -0.109 |

Per track (probabilities per lap at risk):

| track_id | n_races | obs_events_per_race | p_lap1 | h_green_mid | h_pit_window | h_late |
|---|---|---|---|---|---|---|
| albert_park | 4 | 3.000 | 0.366 | 0.032 | 0.073 | 0.029 |
| americas | 4 | 1.000 | 0.106 | 0.012 | 0.029 | 0.011 |
| bahrain | 3 | 0.667 | 0.089 | 0.009 | 0.022 | 0.008 |
| baku | 4 | 1.000 | 0.171 | 0.014 | 0.032 | 0.012 |
| catalunya | 4 | 0.750 | 0.083 | 0.009 | 0.021 | 0.008 |
| hungaroring | 4 | 0.250 | 0.065 | 0.007 | 0.016 | 0.006 |
| imola | 2 | 1.000 | 0.118 | 0.012 | 0.029 | 0.011 |
| interlagos | 3 | 2.000 | 0.212 | 0.017 | 0.039 | 0.015 |
| jeddah | 3 | 1.000 | 0.185 | 0.014 | 0.032 | 0.012 |
| losail | 3 | 1.333 | 0.263 | 0.013 | 0.031 | 0.012 |
| madring | 1 | 1.000 | 0.130 | 0.013 | 0.031 | 0.012 |
| marina_bay | 4 | 1.000 | 0.106 | 0.012 | 0.029 | 0.011 |
| miami | 4 | 1.250 | 0.124 | 0.015 | 0.036 | 0.014 |
| monaco | 4 | 0.500 | 0.066 | 0.007 | 0.016 | 0.006 |
| monza | 4 | 0.500 | 0.084 | 0.009 | 0.022 | 0.008 |
| red_bull_ring | 4 | 1.500 | 0.238 | 0.013 | 0.030 | 0.012 |
| rodriguez | 4 | 1.750 | 0.188 | 0.016 | 0.037 | 0.014 |
| sepang | 1 | 2.000 | 0.160 | 0.017 | 0.039 | 0.015 |
| shanghai | 3 | 0.667 | 0.100 | 0.011 | 0.025 | 0.010 |
| silverstone | 4 | 1.500 | 0.121 | 0.015 | 0.035 | 0.013 |
| spa | 4 | 0.750 | 0.144 | 0.010 | 0.024 | 0.009 |
| suzuka | 4 | 1.000 | 0.153 | 0.011 | 0.027 | 0.010 |
| vegas | 2 | 2.000 | 0.259 | 0.020 | 0.048 | 0.019 |
| villeneuve | 4 | 2.000 | 0.144 | 0.019 | 0.045 | 0.017 |
| yas_marina | 3 | 0.333 | 0.080 | 0.008 | 0.019 | 0.007 |
| zandvoort | 4 | 2.250 | 0.134 | 0.017 | 0.041 | 0.016 |

Event mix by era:

| era_id | red | sc | vsc |
|---|---|---|---|
| 2022_ground_effect | 0.030 | 0.590 | 0.390 |
| 2026_active_aero | 0.050 | 0.270 | 0.680 |

## Car pace now (2026, % of lap vs field median; negative = faster)

| team | state_mean | state_sd |
|---|---|---|
| Mercedes | -1.359 | 0.171 |
| Ferrari | -1.244 | 0.171 |
| Red Bull Racing | -1.126 | 0.171 |
| McLaren | -0.986 | 0.172 |
| Alpine | -0.024 | 0.171 |
| Racing Bulls | 0.079 | 0.171 |
| Audi | 0.213 | 0.171 |
| Haas F1 Team | 0.610 | 0.171 |
| Williams | 0.948 | 0.171 |
| Aston Martin | 1.376 | 0.172 |
| Cadillac | 1.938 | 0.172 |
