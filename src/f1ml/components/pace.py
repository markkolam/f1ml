"""Car and driver pace tracker (local-level Kalman filter across races).

Observations: each race's car+fuel-corrected pace offset from the phase-1 fit,
as % of the field's reference lap (negative = faster). The team (car) level
follows a random walk between races, with extra variance at a regulation reset
(new era) and over the winter break; each driver's deviation from the team mean
is a second, slower random walk. Noise variances are chosen by maximising the
one-step-ahead predictive likelihood, so the filter is tuned to how much car
pace actually moves race to race (upgrades) versus measurement noise.

Outputs the one-step-ahead prediction (mean, SD) for every race, i.e. what the
model believed *before* that race (leakage-free inputs for the simulator), and
the current state for the next race.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from f1ml.components.data import RaceData

PRIOR_VAR = 1.0       # (% of lap)^2 for a team never seen
RESET_VAR = 0.5       # added at a new era (rule reset)
WINTER_VAR_MULT = 4   # winter break counts as this many race-to-race steps


def race_pace(rd: RaceData) -> pd.DataFrame:
    f = rd.fits.dropna(subset=["offset", "reference_offset"]).copy()
    f["pace_pct"] = f["offset"] / f["reference_offset"] * 100
    teams = rd.results[["season", "round", "driver", "team"]]
    f = f.merge(teams, on=["season", "round", "driver"], how="left")
    f = f.merge(rd.events[["season", "round", "event_date", "era_id"]], on=["season", "round"])
    # very large offsets are damaged races (long stops, damage); clip to keep the filter sane
    f = f[f["pace_pct"].abs() < 5]
    return f.sort_values("event_date")


def _filter(obs: pd.DataFrame, key: str, q: float, r: float) -> tuple[pd.DataFrame, float]:
    """Run one local-level filter per `key` over races in date order. Returns predictions and log-lik."""
    state: dict[str, tuple[float, float, str, int]] = {}
    preds, ll = [], 0.0
    for (date, season, era), g in obs.groupby(["event_date", "season", "era_id"], sort=True):
        for _, row in g.iterrows():
            k = row[key]
            if k in state:
                m, v, last_era, last_season = state[k]
                v = v + q * (WINTER_VAR_MULT if season != last_season else 1) + (RESET_VAR if era != last_era else 0)
            else:
                m, v = 0.0, PRIOR_VAR
            y = row["y"]
            s2 = v + r
            ll += -0.5 * (np.log(2 * np.pi * s2) + (y - m) ** 2 / s2)
            preds.append({"season": season, "round": row["round"], key: k, "pred_mean": m, "pred_sd": np.sqrt(s2),
                          "obs": y})
            kgain = v / s2
            state[k] = (m + kgain * (y - m), (1 - kgain) * v, era, season)
    cur = pd.DataFrame([{key: k, "state_mean": m, "state_sd": np.sqrt(v), "era_id": e, "season": s}
                        for k, (m, v, e, s) in state.items()])
    return pd.DataFrame(preds).assign(ll=ll), cur


def fit_pace(rd: RaceData, q_grid=(0.002, 0.005, 0.01, 0.02, 0.05), r_grid=(0.02, 0.05, 0.1, 0.2)) -> dict:
    p = race_pace(rd)
    team = p.groupby(["season", "round", "event_date", "era_id", "team"], as_index=False)["pace_pct"].mean()
    team = team.rename(columns={"pace_pct": "y"})
    best = max(((q, r) for q, r in itertools.product(q_grid, r_grid)),
               key=lambda qr: _filter(team, "team", *qr)[0]["ll"].iloc[0])
    team_pred, team_now = _filter(team, "team", *best)
    p = p.merge(team[["season", "round", "team", "y"]].rename(columns={"y": "team_obs"}),
                on=["season", "round", "team"], how="left")
    p["y"] = p["pace_pct"] - p["team_obs"]
    dbest = max(((q, r) for q, r in itertools.product((0.0005, 0.001, 0.003, 0.01), r_grid)),
                key=lambda qr: _filter(p, "driver", *qr)[0]["ll"].iloc[0])
    drv_pred, drv_now = _filter(p, "driver", *dbest)
    params = pd.DataFrame([{"param": "team_q", "value": best[0]}, {"param": "team_r", "value": best[1]},
                           {"param": "driver_q", "value": dbest[0]}, {"param": "driver_r", "value": dbest[1]}])
    return {"pace_team_pred": team_pred.drop(columns="ll"), "pace_team_now": team_now,
            "pace_driver_pred": drv_pred.drop(columns="ll"), "pace_driver_now": drv_now, "pace_params": params}
