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

Known limits of phase 3:
- There is no tyre cliff or graining term yet; deg is linear in tyre age within a stint.
- Compounds are keyed by label until Pirelli C-numbers are configured.
- Pass thresholds use the expected (race-average) pace gap.

## Known limits of phase 1
- `fuel_effect` combines fuel burn and track evolution, which both scale with race lap.
- Deg is linear per compound within a race fit. Phase 3 adds a temperature slope and pooling, but not graining or cliffs.
- Practice fuel loads are unknown. Long-run deg uses the configured prior fuel effect.
- Regulation plugins declare their data needs only. Their simulator hooks are phase 6.
