# F1 Race & Strategy ML Model — Draft Design (v2, refined from Ruth Buscombe's Substack)

## Context
**Goal:** an F1 machine-learning system that is as realistic as possible. Three things must be easy to do:
1. add each new race;
2. add each new **season**;
3. absorb **regulation changes** (2026-style resets) without rewriting the model.

**Research basis:** I read the full text of 27 posts from Ruth Buscombe's Substack, Jul–Nov 2026. She is ex-Head of Race Strategy at Sauber, previously Ferrari and Haas, and publishes "Method" notes for her race models. Posts read include:
- The Pit Lane That Never Was
- What If? Austin
- What If? Singapore
- Hards or softs? 2 or 3?
- How we go racing at a new track, parts 1, 2 and 2.5
- The best strategy depends on who you are
- Tow chicken
- The racing gods are in the rain
- Baku briefings
- Not our first rodeo
- Mexico is Monaco

**Starting point:** `/home/user` is empty, so this is greenfield.

**Guiding principle (hers):**
> "The model must reproduce the real race to the second before it earns an opinion about a fake one."

## 1. What a real strategist measures → what the model must contain
| Her concept (with her numbers) | Model requirement |
|---|---|
| Deg regressed on **green, clean laps, car and fuel taken out**, top-10 cars (e.g. 557 laps). Typical deg 0.02–0.10 s/lap; 0.36 s/lap is a **cliff** | Hierarchical tyre model fitted on fuel- and car-corrected clean laps; cliff/graining terms |
| Deg is **temperature-aware**: Soft 0.10 at 52 °C → 0.075 at 48 °C | Track-temperature covariate per compound |
| **Graining:** front graining is a phase (8–10 laps, then recovers); rear graining is a step and doesn't recover. It is driven by fuel load, tyre bulk/surface temperature and banking load | Graining state in the tyre model (onset, recovery), conditioned on fuel and track energy |
| Fuel effect is about 0.03–0.065 s/lap per lap, track-dependent. On full tanks **deg and fuel cancel** and compounds "collapse into one tyre" | Fuel × compound interaction. The compound advantage grows as fuel burns |
| **"The window":** % of lap time spent at 70–150 km/h, where the tyre is the limiter. Monaco 29%, Singapore 22%, Austin 16%, Monza 6%. It is about 30% on a green lap, 37% on lap 1 and about 70% on a restart lap | Telemetry-derived track/era feature; it scales the compound advantage and restart value |
| **Push vs manage:** managing costs 0.3 s/lap and cuts deg by 30%; the 2026 car "can't manage by lifting" | Driver-mode parameter in the tyre and overtake models (era-dependent) |
| C-numbers change character between seasons (the C3 is 0.27 s faster than the C2 in 2023–25, but 0.44 s *slower* in 2026) | Model compounds by **Pirelli C-number × season**, never by Soft/Medium/Hard labels |
| **Overtaking S-curve**, built from "car within 1.2 s under green → pass by next lap" events. Budapest: 0/152 below 0.25 s, 1 in 4 at 0.25–0.75, 2/3 at 0.75–1.25, 50% point 1.09 s/lap. Per-track threshold: Austin 0.5–0.7, most tracks 0.9–1.5, Singapore 2.5, Monaco 4.5 | Logistic pass model per track, partially pooled; a **driver-specific offset** (Verstappen about 0.2 s below the field); a "no fight" upper threshold |
| Dirty air costs about 0.3 s/lap (Norris, Austin). The tow is worth 0.3–0.4 s and about 15 km/h at Monza | Following penalty to pace and deg; tow/energy effect on straights |
| **Restarts:** a softer compound behind a harder one passes within two laps far more often (Singapore 4/6). Old tyres (>15 laps) gain places only 6% of the time | Separate restart-pass model |
| **Pit loss** is "a range, not a number". It is measured from the moment the car leaves the racing line, relative to staying out, and re-tuned through the weekend. Green 20.5–24 s; VSC about 10.5–12 s; SC about 11–13 s | Per-track pit-loss distribution by green/VSC/SC, updated by FP pit practice and the first real stops |
| **Flags are causal, not random:** external causes (flat rate), retirements (lap 1, late in hot races, after stops, early/late season), accidents (lap 1, restarts, pit window where 1- and 2-stoppers mix, final push). Mexico example: 4 of 9 races had a lap-1 SC/VSC, then a 1.3%/lap hazard after lap 30. Every flag is mapped by corner | Per-lap SC/VSC/red hazard built from those cause components, plus a corner-location map. New tracks: "the sum of the parents, not the average" |
| **Lap 1:** concertina (11.0 s nose-to-tail at Budapest); a pit-lane start joins the gearbox by S2; grid-side effect; the launch is power-limited, so compound matters little (about 70 m of 1,080 m); pole's lap-1 lead rate varies by track | Start model: positions by grid slot and side, concertina, pit-lane-start join |
| **Pace estimation:** two readings, "Friday-weighted" (FP2 long run + quali) vs "Saturday-rebased". Fuel loads are unpublished, which makes them "the biggest unknown" | Latent car+driver pace with **latent unknown fuel load** per FP run |
| **Weather:** wet→slick crossover at about 110% of dry lap time, with only about a 2-lap window. Drying rate (% per minute) depends on humidity and the track–air temperature gap. When weather models disagree, "that is the strategy question" | Weather scenario ensemble; crossover and drying-rate model |
| **Crossover thresholds:** 1-stop vs 2-stop is decided by deg vs pit loss/laps (Austin about 0.07 s/lap). A tenth of setup = 5.6 s per race | Analytic threshold calculator next to the simulator |
| 2026: no DRS; active-aero Straight Mode zones; battery deployment maps (350 vs 250 kW zones), clipping, energy-starved tracks; tow saves battery | Pluggable **regulation module** (energy/aero), see §3 |

**How she decides:**
- **Simulations vote:** "299 against one, go with the data; 150 vs 150, the driver is the casting vote." She runs 2,500–3,000 races per option.
- **"Safety Car on every lap" sweeps:** e.g. S-H-H ahead in 59% of SC laps vs 18% for a 3-stop.
- **"The best strategy depends on who you are":** P8 needs variance, the leader needs robustness.
- **Game theory:** "Build your race on what your competitors will do, not on what they'd do with your tyre model." She back-calculates rivals' tyre beliefs from the sets they kept and hedges by **splitting the two cars**.
- **Tyre-set inventory** (which new sets are left after qualifying) shapes strategy: keep a new Soft for the end or the restart.
- **Evaluation:** "What If?" counterfactual debriefs, plus betting markets (Kalshi, bookmakers) as an external benchmark.

## 2. Architecture: the "strategist's stack"
This is a hybrid design: learned component models feed a Monte Carlo race simulator, which feeds a strategy/game layer. A direct ML ranker serves as benchmark and blend. Everything is probabilistic (posterior samples flow into the simulator).

**A. Data layer**
- **Sources:**
  - **FastF1:** timing, stints, compounds, tyre life, telemetry, weather, track status, race-control messages, 2018+.
  - **Jolpica-F1:** results back to 1950.
  - **OpenF1:** intervals, pit durations, position; also a live mode.
  - **Pirelli:** C-number nominations, preview pit loss/windows, race reports.
  - **Weather forecasts:** multiple models.
  - **Betting odds:** benchmark only.
- **Storage:** `raw → clean → features` Parquet, partitioned `season/round/session`, queried with DuckDB.
- **Schema-versioned data contracts:** new-era fields (e.g. energy) are optional columns.

**B. Cleaning & normalisation**
- **Lap tags:** in/out, lap 1, SC/VSC/yellow, pit, wet, traffic (gap < 1.5 s marks dirty air), restart, outlier.
- **Joint fit** of fuel effect, car offset and track evolution.
- **Telemetry features:** per-lap time in the 70–150 km/h window, corner-zone splits and min speeds.

**C. Component models (Bayesian, NumPyro)**
- **Pace:** latent car+driver pace, with a random walk across rounds for upgrades and latent FP fuel load.
- **Tyre:** f(C-number, season, age, fuel, track_temp, window%, driver mode) + cliff + graining state + dirty-air penalty.
- **Pit loss:** distribution per track × condition; stop-time distribution per team.
- **Overtake:** per-track logistic S-curve, with offsets for driver, top-speed/energy and compound/age; a restart sub-model; tow.
- **Flags:** a causal hazard (external + retirement + accident components) by lap phase, restarts, pit-window mixing, heat, season phase and corner.
- **Start:** grid slot/side, concertina, pit-lane join, T1 incident rate.
- **Reliability:** DNF hazard per PU/team/era, cumulative heat/brake wear.
- **Weather:** rain scenarios, drying rate, ~110% crossover, wind (yaw-sensitive cars).

**D. Race simulator (core)**
- Vectorised lap-by-lap Monte Carlo (NumPy/Numba), all cars, 3k–10k runs per option.
- Simulates traffic, dirty air, passes, stops (compound rules from the era config), SC/VSC/red (free tyre change), restarts, tyre-set inventory and reactive policies (cover the undercut, box under SC if in window).
- **Calibration gate:** replaying a finished race with the actual strategies must reproduce the real gaps within a tolerance before any what-if is trusted.

**E. Strategy & game layer**
- Enumerates plans (stops × C-number order × windows, constrained by the sets left).
- Scores each plan per **position-dependent objective**: E[points], P(win), P(≥ target).
- Best-response iteration against rivals, plus **belief inference** that back-calculates rivals' deg beliefs from kept sets and early stops.
- Teammate split/hedge.
- **Outputs:** vote share ("best in N/3000"), SC-on-every-lap sweep, robustness across tyre-model "camps", and crossover thresholds.

**F. Direct ML predictor**
- LightGBM LambdaMART ranker over pre-race features plus simulator outputs.
- Blended using walk-forward weights; benchmarked against betting markets.

**G. New-track cold start**
- Similarity over layout, window%, energy, surface and altitude picks "parent" tracks.
- Overtake/deg priors come from the parents, with inflated variance. Flag hazard is the **sum** of the parents' hazards.
- Updated session by session (FP1 → FP2 → FP3 → Q). Part of this is the "Saturday or Sunday track" call.

**H. Tools**
- `whatif` counterfactual replay.
- Streamlit dashboard: deg curves, S-curves, pit windows, SC sweep, vote share.
- Optional live mode (OpenF1).

## 3. Built for new seasons and rule changes (key requirement)
Everything season- or rule-specific is **data/config or a plugin, never hard-coded**:
```
config/
  eras/2014_hybrid.yaml, 2017_wide.yaml, 2022_ground_effect.yaml, 2026_active_aero.yaml
  seasons/2026.yaml        # calendar, sprint rounds, teams, drivers (with mid-season swaps), PU suppliers, era ref
  tracks/<track>.yaml      # versioned by layout/surface (e.g. mexico@2026_resurface), pit-lane length, zones
  pirelli/2026.yaml        # C-number nominations per round
```
- **Era config** holds:
  - points system and sprint format
  - grid size
  - tyre range (C1–C6) and compound-usage rules
  - DRS on/off, active-aero zones
  - energy parameters (deployment kW, recharge limits)
  - refuelling, parc fermé and penalty rules
  - a list of enabled **physics plugins**
- **Plugin interface** (`RegulationModule`): hooks such as `lap_time_adjust`, `overtake_modifier`, `energy_state`, `start_model`. Examples: `DRSModule` (2011–2025), `ActiveAeroEnergyModule` (2026+). A future rule change means a new YAML file plus, if needed, a new module. The simulator core stays unchanged.
- **Cross-era transfer:** parameters are hierarchical, with era-level effects. Previous-era posteriors become down-weighted priors for a new era. The transfer weight is learned from how well the old era predicted the first rounds of the new one (2022 and 2026 resets give two test cases). Everything is stored per C-number × season, since compound behaviour changes.
- **Adding a race:** `f1ml update --season 2026 --round 22`. This is idempotent: ingest → validate contracts → features → incremental posterior update → calibration gate → backtest report.
- **Adding a season:** add `seasons/2027.yaml` (and an era file if the rules change), then `f1ml update --season 2027`.
- **Model registry (MLflow):** every fit is tagged with era, season and round, so any past state can be reproduced.

## 4. Repo layout
```
f1ml/
  config/ (eras, seasons, tracks, pirelli)
  src/f1ml/ingest/     fastf1, jolpica, openf1, pirelli, weather, odds
  src/f1ml/clean/      lap tagging, fuel/car/evolution normalisation, telemetry window features
  src/f1ml/models/     pace, tyre, pitloss, overtake, flags, start, reliability, weather, ranker
  src/f1ml/regs/       RegulationModule base + drs.py, active_aero_energy_2026.py
  src/f1ml/sim/        race engine, inventory, policies
  src/f1ml/strategy/   enumerator, objectives, game/belief inference, sweeps, thresholds
  src/f1ml/eval/       walk-forward backtest, calibration gate, market benchmark
  src/f1ml/cli.py      update | backtest | simulate | whatif | new-track
  app/dashboard.py
  tests/
```
**Stack:** Python 3.12, FastF1, polars, DuckDB, NumPyro, LightGBM, Numba, MLflow, Typer, Streamlit, pytest.

## 5. Roadmap
1. **Data foundation:** config schema (eras, seasons, tracks), ingestion 2018–2026, lap tagging, fuel/car normalisation, `f1ml update`.
2. **Baseline:** LightGBM ranker plus a walk-forward backtest against betting odds.
3. **Component models:** tyre (C-number × season, temperature, graining), pace, pit loss, overtake S-curves, causal flag hazard, start.
4. **Simulator:** with the calibration gate on 2024–2026 races.
5. **Strategy/game layer:** vote share, SC sweeps, belief inference, teammate split.
6. **Regulation plugins and cold start:** DRS vs 2026 energy/active-aero modules, era transfer, new-track priors.
7. **Tools:** what-if tool, dashboard, live mode.

## 6. Verification
- **Calibration gate:** replay real races with the actual strategies. Finishing gaps should land within the predictive intervals, mirroring her "to the second" anchor (e.g. Hungary 2026, Austin 2026).
- **Walk-forward backtest** over every round 2023–2026, compared with betting markets. Metrics:
  - win/podium Brier and log loss
  - Spearman on finishing order
  - calibration plots
- **Component checks:**
  - tyre/pace models: held-out stint lap-time MAE
  - overtake model: S-curve vs observed pass rates (e.g. Budapest 50% point ≈1.09 s)
  - pit loss: vs measured
  - flags: predicted vs actual SC frequency by lap phase
- **Regulation-change test:** train on 2022–2025 only and predict early 2026 rounds, checking that the era-transfer weighting beats both "no transfer" and "full transfer".
- **Extensibility smoke test:** add a dummy `seasons/2027.yaml` and a dummy era with a stub plugin. The pipeline should run end-to-end with no code changes outside config and the plugin.

## Known limitations
- Public data only: no team tyre temperatures, fuel loads, setups or energy maps. Her models face the same limit ("No proprietary tools, just maths").
- Small same-era samples after a rule reset mean the model leans on priors and in-weekend updates.
- Outputs are distributions. A good call can still lose.
