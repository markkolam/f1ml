# f1ml — F1 race & strategy modelling stack

Phase 1 (data foundation) of the design in `docs/F1_ML_model_design.md`:
config for eras/seasons/tracks, ingestion of 2020+ timing data, lap tagging,
a per-race fuel/tyre/car fit, long-run extraction, and an idempotent
`f1ml update` command.

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```
Network access:
- `api.jolpi.ca` (Jolpica) is always needed. It provides the calendar, results and pit stops.
- Timing data comes from one of two sources:
  - `livetiming.formula1.com` (FastF1), which covers 2018 onwards and is preferred;
  - `api.openf1.org` (OpenF1), used as the fallback, which covers 2023 onwards only.

`--source auto` (the default) uses FastF1 if livetiming is reachable and OpenF1 otherwise.
`f1ml status` shows which source each round came from. When livetiming becomes
reachable, re-ingest the OpenF1 rounds for consistency with
`f1ml update --season <y> --source fastf1 --force`.

What the OpenF1 source derives differently:
- **Track status:** rebuilt from race-control messages.
- **Tyre life:** taken from stints.
- **Deleted laps:** parsed from track-limits messages.
- **Not available:** per-lap position, finish-line speed and telemetry window features.

## Everyday use
```bash
f1ml update --season 2026 --round latest     # after each race weekend
f1ml update --season 2026 --round 21 --telemetry   # + per-lap "window" features (slow)
f1ml backfill --from 2020 --to 2026          # first-time load
f1ml status
f1ml query "SELECT driver, avg(lap_time_car_fuel_corr_s) FROM clean_laps WHERE season=2026 AND round=21 AND session='R' AND is_clean_free_air GROUP BY 1 ORDER BY 2"
```
Re-running is safe: finished steps are recorded in `data/manifest/` and skipped.
`--force` re-downloads and re-cleans; `--reclean` re-runs cleaning and fits from stored raw data,
which you need after changing cleaning or fit code or settings.

## Adding things
| What | How |
|---|---|
| New race | `f1ml update --season <y> --round latest` |
| New season | `f1ml season-init 2027` (creates `config/seasons/2027.yaml`), then `f1ml update --season 2027` |
| Rule change | Close the current era (`last_season`) and add `config/eras/<id>.yaml` with `transfer_from: <previous era>`. If it brings new physics, add a module in `src/f1ml/regs/` decorated with `@register("<name>")` and list it under `plugins`. |
| Layout/surface change | Add a `versions` entry to the track in `config/tracks/registry.yaml` |
| New venue | Add it to the registry (`f1ml tracks-check --season <y>` lists unknown ones) |
| Tyre C-numbers | `config/pirelli/<season>.yaml` (see the README there) |

## Data layout
`data/<raw|clean|features>/<table>/season=YYYY/round=RR[/session=CODE]/part.parquet`,
every table checked against `src/f1ml/contracts.py`. DuckDB views are named
`<layer>_<table>`, with `clean_laps` and `raw_laps` etc.

Tables in phase 1:
- **raw:** `events`, `laps`, `results`, `weather`, `race_control`, `pit_stops`
- **clean:** `laps`, i.e. raw laps plus:
  - tags: `is_clean`, `is_clean_free_air`, `is_dirty_air`, `is_restart_lap`, ...
  - `gap_ahead_s`
  - weather
  - track version, era, C-number
  - fuel-corrected and car+fuel-corrected lap times
- **features:**
  - `fit_params`: fuel effect, deg and base pace per compound, car offsets per race/sprint
  - `long_runs`: best clean run of 5+ laps per stint, raw and fuel-adjusted deg
  - `lap_window`: share of lap time at 70–150 km/h, with `--telemetry`

## Phase 2: pre-race model and backtest
```bash
f1ml backtest --from 2024 --to 2026            # walk-forward, writes reports/backtest.md
f1ml predict --season 2026 --round 20          # after `f1ml update` has ingested FP + qualifying
```
- **Features** (`src/f1ml/features/prerace.py`): one row per race and driver, using only pre-race information.
  - Weekend: grid and qualifying position, gap to pole and to teammate, Friday long-run gap, sprint result and pace.
  - Form: exponentially weighted race pace from the phase-1 fits, qualifying gap, finishes, points, team DNF rate.
  - Track history: grid-to-finish correlation ("Saturday or Sunday track"), places changed, share of races with a Safety Car.
  - Era: races since the current rules began.
- **Models** (`src/f1ml/models/`):
  - `lgbm` (default): LightGBM LambdaRank with monotone constraints, blended with the grid order. The blend weights are fitted on held-out races.
  - `lgbm_pure`: the ranker without the blend.
  - `grid` and `quali`: baselines.

  Every model's scores become probabilities through a Plackett-Luce finishing-order model, whose scale is calibrated on held-out races.
- **Backtest** (`src/f1ml/eval/`): each race is predicted by a model trained only on earlier races. Metrics are rank correlation, winner log loss, Brier scores for win and podium, favourite hit rate, top-3 overlap and position error.

## Phase 3: component models
```bash
f1ml components            # fits all components, writes features/comp_* tables + reports/components.md
```
Each component is fitted from race data per track and era. Thin evidence is pulled toward the era average by empirical-Bayes pooling (`src/f1ml/components/pooling.py`). `reports/components.md` compares each component with Ruth Buscombe's published figures.

| Component | What it estimates | Main tables |
|---|---|---|
| `tyre` | Deg and base pace per track x era x compound. Wet races are excluded. A track-temperature slope is learnt within track across years. | `comp_tyre_track`, `comp_tyre_params` |
| `pitloss` | Green pit loss measured against cars that stayed out. VSC/SC losses are *derived* with her ~0.52 ratio, because they can't be measured from lap sums. | `comp_pitloss_track` |
| `overtake` | Pass S-curve per track (her "pass ladder"): attempts within 1.2 s under green, with per-driver conversion and the dirty-air cost. | `comp_overtake_track`, `comp_overtake_driver`, `comp_dirty_air_track` |
| `flags` | Per-lap SC/VSC/red hazard covering lap 1, restarts, pit-window activity, late race, wet and track effects, plus the event-type mix per era. | `comp_flags_track`, `comp_flags_coefs` |
| `start` | Lap-1 places gained by grid slot, grid-side effect per track, lap-1 concertina. | `comp_start_*` |
| `pace` | Kalman local-level tracker of team and driver pace with one-step-ahead predictions (leakage-free). | `comp_pace_*` |

`fit_components(store, as_of=date)` fits every component on races before a date, for walk-forward use in the race simulator (phase 4).

### Fuel-dependent tyre model
Buscombe: "On full tanks, deg and fuel cancel, and the three compounds collapse into one tyre ... The compound is worth nothing at lap 5 and everything at lap 50." The race fit and tyre component now include it:

    lap time = car + base[c] + base_fuel[c] * f - fuel * (lap - 1) + (deg[c] + deg_fuel * f) * tyre age

Here f is the fuel fraction, 1 on lap 1 and 0 on the last lap. `base[c]` is the compound's pace on an empty tank. `base_fuel[c]` is how much its advantage over the Medium changes by a full tank. The simulator and the plan search use the same formula; the plan search uses an exact closed-form stint time.

What the data say (2023 - Mexico 2026):

| | Soft vs Medium, empty tank | Change to a full tank |
|---|---|---|
| Mexico 2026 | -0.58 s | +0.68 |
| Austin 2026 | -0.46 s | +0.60 |

- At those two tracks the Soft's advantage nearly vanishes on full fuel, as she says. Her figure for Mexico is "0.57 seconds a lap faster than the Medium at equal age".
- Across all races the effect points the same way but is noisy: the Soft's fuel term is positive in 58% of race estimates.
- The 2026 era mean is +0.14 and the 2022-25 era mean is +0.05.
- The Hard shows no consistent effect.
- Wear is slightly higher on heavy fuel (`deg_fuel` about +0.01).

The effect is only identifiable when a compound is run at different fuel loads at different ages. Within one race that is often weak, so the per-track values lean on pooling.

Known limits of phase 3:
- There is no tyre cliff or graining term yet; deg is linear in tyre age within a stint (with the fuel terms above).
- Compounds are keyed by label until Pirelli C-numbers are configured.
- Pass thresholds use the expected (race-average) pace gap.

## Phase 4: race simulator
```bash
f1ml simulate --season 2026 --round 19                               # pre-race: odds per driver
f1ml simulate --season 2026 --round 19 --strategy "VER=M-H@30"       # force a car's plan
f1ml whatif --season 2026 --round 18 --strategy "VER=S-M-S@13,40"    # replay a past race with a changed plan
f1ml calibrate --from 2024 --to 2026                                 # replay check, reports/simulator.md
f1ml backtest --models grid,lgbm,sim --report reports/backtest_sim.md
```
- **Engine** (`src/f1ml/sim/engine.py`): a lap-by-lap Monte Carlo of every car, vectorised over simulations (2,000 races in under a second). Each lap it handles:
  - lap time: car pace + compound base + deg x age, minus fuel burn, plus noise;
  - dirty air;
  - pass attempts on the track's S-curve, using the expected pace gap and the driver's conversion shift;
  - pit loss by condition;
  - SC/VSC/red flags drawn from the hazard model, with field compression at SC restarts and a free tyre change under a red flag;
  - reactive stops under a cheap SC/VSC;
  - random retirements.
- **Strategies** (`sim/strategy.py`): plan strings such as `S-M-H@14,38`. The default plans are the fastest 1-, 2- and 3-stop sequences from the stint-time model. Each car draws one of the plans within 8 s of the best.
- **Inputs** (`sim/params.py`):
  - *Pre-race*: components fitted only on earlier races, plus car pace from the pace tracker blended 50/50 with this weekend's qualifying gap.
  - *Replay*: the race's own fitted pace and tyres, plus the strategies, neutralisations, retirements and lap-1 order that actually happened.
- **Replay check** (`sim/replay.py`): Buscombe's standard, "reproduce the real race before it earns an opinion about a fake one". `f1ml whatif` builds on it.

Results (2024 - Mexico 2026):
- **Replay check, 58 dry races** (`reports/simulator.md`):
  - finishing-order rank correlation 0.90;
  - the simulated favourite won 76% of the time;
  - 70% of actual positions fell inside the simulated 10-90% range (ideal ~80%, so the ranges are a little narrow);
  - median gap error 10 s, with the worst races decided by collisions (Austria 2024, Canada 2025).
- **Pre-race backtest, 66 races** (`reports/backtest_sim.md`, with the fuel-dependent tyre model): `sim` scores winner log loss 1.382 against `lgbm`'s 1.311 and rank correlation 0.688 against 0.706. With the linear tyre model it was 1.369 against 1.317.
  - It is better than `lgbm` on winners in 2025 and 2026, and worse in 2024, when it only had 2023 history.
  - It does **not** replace `lgbm` as the default predictor. It is the engine for strategy and what-if questions (phase 5).

Known limits of phase 4:
- Lapped traffic and blue flags are not modelled.
- Wet and mixed races are skipped in replays.
- Compound pace depends on fuel load only linearly. There is no "window" term from telemetry (share of the lap at 70-150 km/h), because OpenF1 car data isn't ingested.
- Other cars do not react to a changed strategy in `whatif`.

## Phase 5: strategy layer
```bash
f1ml strategy --season 2026 --round 18 --driver VER                          # plan vote, crossover, tyre-model regret
f1ml strategy --season 2026 --round 18 --driver VER --sc-sweep --rivals 6 --teammate
f1ml strategy-check --from 2024 --to 2026                                    # recommended vs what winners ran
```
`f1ml strategy` writes a briefing to `reports/strategy_<season>_<round>_<driver>.md`. Each part follows one of Buscombe's ideas:

| Part | What it does | Buscombe |
|---|---|---|
| Plan vote | Every candidate plan for the driver is run through the *same* simulated races (common random numbers), showing the share of races each plan wins and the paired difference from the best with its SE | "if it's 299 against one, we go with the data; 150 vs 150, the driver is the casting vote" |
| Objective | points, P(win), P(podium), P(top 10) or position; `auto` picks one from the driver's situation | "the best strategy depends on who you are" |
| Crossover | The deg level at which another stop starts to pay | "the line between a one-stop and a two-stop is a tyre deg of about 0.07s a lap" |
| Tyre-model regret | Plans re-run under lower, higher, "soft falls off" and "all equal" deg; the minimax-regret plan is the robust one | "same car, same pit loss, two tyre models" |
| SC on every lap | A Safety Car forced on each lap in turn; share of laps on which each plan was best | "I ran one Safety Car on every lap from 3 to 53" |
| Rivals | Iterated best responses among the drivers around the focus driver | "build your race on what your competitors will do" |
| Teammate | All plan pairs for the two cars: team points and P(either wins), split vs same plan | "teams that pair strategies win championships" |

History check (`f1ml strategy-check`, 59 dry races 2024 - Mexico 2026): the pre-race recommendation for each eventual winner, compared with what they ran.

| | Linear tyre model | Fuel-dependent tyre model |
|---|---|---|
| Starting compound matches | 31% | **47%** |
| Exact compound sequence matches | 10% | **19%** |
| Winner's sequence in the top 3 | 19% | **31%** |
| Number of stops matches | 54% | 51% |
| Best plan's average vote share | 13% | 16% |

The fuel-dependent tyre model made recommendations noticeably more realistic. The replay check was unchanged (rank correlation 0.90).

Agreement is still partial. Winners' "actual" plans include opportunistic Safety Car stops that no pre-race plan contains. Real teams also start on the Medium more often: 48 of 59 winners did, against 29 of the model's recommendations.

Known limits of phase 5:
- Tyre-set inventory (which new sets each car has left) is not in the public data, so plans assume any compound is available. For the same reason, rivals' tyre beliefs can't be back-calculated from the sets they kept.
- Plans are fixed sequences with reactive SC stops. There is no mid-race re-planning.
- Rival responses use expected points for everyone.

## Known limits of phase 1
- `fuel_effect` combines fuel burn and track evolution, which both scale with race lap.
- Deg is linear per compound within a race fit. Phase 3 adds a temperature slope and pooling, but not graining or cliffs.
- Practice fuel loads are unknown. Long-run deg uses the configured prior fuel effect.
- Regulation plugins declare their data needs only. Their simulator hooks are phase 6.
