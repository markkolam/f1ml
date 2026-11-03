# Component models (phase 3)

Generated 2026-11-03 18:13 UTC from 2023-2026 race data (OpenF1).

## Against Ruth Buscombe's published figures

| quantity | buscombe | f1ml |
|---|---|---|
| Monza 2026 race deg (s/lap) | 0.03 | 0.020 |
| Hungary 2026 green pit loss (s) | 21.9 measured | 22.551 |
| Austin 2026 green pit loss (s) | 20.5-21.5 | 22.390 |
| Mexico 2026 green pit loss (s) | 20.5 | 23.459 |
| Madrid 2026 green pit loss (s) | ~24 (pre-race estimate) | 26.367 |
| Budapest 2026 pass attempts (<=1.2 s, green) | 225 | 221.000 |
| Budapest 2026 passes by next lap | 28 | 27.000 |
| Budapest 2026 50% pass threshold (s/lap) | 1.09 | 1.590 |
| Austin 2026 50% pass threshold (s/lap) | 0.5-0.7 | 1.385 |
| Singapore 50% pass threshold (s/lap) | 2.5 | 3.303 |
| Monaco 50% pass threshold (s/lap) | 4.5 | 7.957 |
| Verstappen pass-curve shift (s/lap) | ~0.2 better than field | 0.279 |
| Dirty-air cost, 2026 era mean (s/lap) | ~0.3 (Norris, Austin) | 0.382 |
| Mexico late-race neutralisation hazard (per lap) | 1.3% | 1.438 |
| Budapest 2026 lap-1 concertina (s) | 11.0 | 11.000 |

Notes: our pass thresholds use the expected (race-average) pace gap, which is smoother than a lap-by-lap gap, so they sit ~0.3-0.6 s above hers; track ordering agrees. Green pit losses run ~1-3 s above her figures: we count the whole in-lap and out-lap against cars that stayed out (cold-tyre out-lap included). VSC/SC pit losses are derived from the measured green loss with her ~0.52 ratio (not measurable from lap sums behind a Safety Car). The restart effect on neutralisation hazard came out negative in this data (no post-restart bump), unlike her flag model.

## Tyre: 2026 deg by track (s/lap per lap of tyre age, temperature-adjusted, pooled)

| track_id | HARD | MEDIUM | SOFT |
|---|---|---|---|
| albert_park | 0.048 | 0.026 | 0.059 |
| americas | 0.063 | 0.076 | 0.060 |
| baku | nan | 0.028 | 0.018 |
| catalunya | 0.105 | 0.154 | 0.078 |
| hungaroring | 0.071 | 0.055 | 0.054 |
| madring | 0.018 | 0.041 | 0.018 |
| marina_bay | nan | 0.050 | 0.058 |
| miami | 0.037 | 0.036 | 0.027 |
| monaco | 0.043 | 0.044 | 0.035 |
| monza | 0.027 | 0.026 | 0.033 |
| red_bull_ring | 0.062 | 0.071 | 0.047 |
| rodriguez | nan | 0.045 | 0.041 |
| shanghai | 0.057 | 0.132 | nan |
| silverstone | 0.064 | 0.056 | nan |
| spa | 0.067 | 0.067 | 0.113 |
| suzuka | 0.062 | 0.067 | nan |
| zandvoort | 0.063 | 0.082 | 0.065 |

Tyre parameters (temp_gamma = extra deg per lap of age per degree C of track temperature):

| param | value | se |
|---|---|---|
| temp_gamma | 0.001 | 0.000 |
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
| madring | 26.400 | 13.700 | 13.700 |
| marina_bay | 23.500 | 12.200 | 12.200 |
| miami | 19.300 | 10.000 | 10.000 |
| monaco | 21.300 | 11.100 | 11.100 |
| monza | 24.700 | 12.900 | 12.900 |
| red_bull_ring | 21.400 | 11.100 | 11.100 |
| rodriguez | 23.500 | 12.200 | 12.200 |
| sepang | 25.400 | 13.200 | 13.200 |
| shanghai | 24.900 | 12.900 | 12.900 |
| silverstone | 22.700 | 11.800 | 11.800 |
| spa | 22.800 | 11.900 | 11.900 |
| suzuka | 23.300 | 12.100 | 12.100 |
| villeneuve | 25.200 | 13.100 | 13.100 |
| zandvoort | 20.000 | 10.400 | 10.400 |

## Overtaking, 2026: pass thresholds (s/lap of pace advantage)

| track_id | n_attempts | pass_rate | threshold_50 | threshold_90 |
|---|---|---|---|---|
| monza | 244 | 0.210 | 1.050 | 2.330 |
| miami | 140 | 0.160 | 1.170 | 2.210 |
| silverstone | 135 | 0.130 | 1.260 | 2.250 |
| shanghai | 179 | 0.170 | 1.320 | 2.780 |
| red_bull_ring | 182 | 0.130 | 1.370 | 2.390 |
| americas | 126 | 0.230 | 1.380 | 2.460 |
| catalunya | 149 | 0.130 | 1.500 | 2.650 |
| suzuka | 355 | 0.100 | 1.510 | 2.900 |
| rodriguez | 323 | 0.060 | 1.510 | 2.480 |
| villeneuve | 123 | 0.150 | 1.530 | 2.940 |
| albert_park | 167 | 0.160 | 1.530 | 2.790 |
| hungaroring | 221 | 0.120 | 1.590 | 2.520 |
| spa | 175 | 0.140 | 1.680 | 3.190 |
| zandvoort | 226 | 0.110 | 1.910 | 3.170 |
| sepang | 155 | 0.170 | 2.220 | 4.630 |
| baku | 189 | 0.110 | 2.340 | 4.300 |
| marina_bay | 183 | 0.100 | 3.300 | 5.680 |
| madring | 161 | 0.050 | 3.890 | 6.210 |
| monaco | 235 | 0.010 | 7.960 | 12.260 |

Driver pass-curve shifts (s/lap equivalent; positive = converts better than the field):

| driver | logit_shift | pace_equiv_s | n_attempts |
|---|---|---|---|
| VER | 0.565 | 0.279 | 432 |
| SAI | 0.387 | 0.191 | 803 |
| PER | 0.312 | 0.154 | 544 |
| HAM | 0.289 | 0.143 | 942 |
| LEC | 0.266 | 0.131 | 757 |
| STR | 0.257 | 0.127 | 922 |
| RUS | 0.227 | 0.112 | 855 |
| ALO | 0.188 | 0.093 | 880 |
| PIA | 0.148 | 0.073 | 740 |
| BEA | 0.140 | 0.069 | 622 |

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
| Mercedes | -1.386 | 0.171 |
| Ferrari | -1.277 | 0.171 |
| Red Bull Racing | -1.182 | 0.171 |
| McLaren | -1.022 | 0.172 |
| Alpine | -0.063 | 0.171 |
| Racing Bulls | 0.081 | 0.171 |
| Audi | 0.157 | 0.171 |
| Haas F1 Team | 0.609 | 0.171 |
| Williams | 0.954 | 0.171 |
| Aston Martin | 1.397 | 0.172 |
| Cadillac | 1.974 | 0.172 |
