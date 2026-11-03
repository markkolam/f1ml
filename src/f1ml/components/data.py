"""Shared inputs for the component models (phase 3).

`RaceData` holds every race lap (Grand Prix sessions) with the columns the
components need, plus derived per-lap race order and the per-race phase-1
fit (car offsets, compound base pace, deg) used to compute *expected* pace.
Everything can be restricted to races before a date (`as_of`) so the same
fits can be used leakage-free inside walk-forward evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1ml.store import Store

LAP_COLS = [
    "season", "round", "driver", "team", "lap_number", "lap_time_s", "time_s", "pit_in_time_s",
    "pit_out_time_s", "compound", "tyre_life", "stint", "track_status", "is_lap1", "is_in_lap", "is_out_lap",
    "is_neutralised", "is_red", "is_yellow", "is_restart_lap", "laps_since_restart", "is_wet_tyre",
    "is_dirty_air", "is_clean", "is_clean_free_air", "gap_ahead_s", "track_temp", "track_id", "track_version",
    "era_id", "lap_time_car_fuel_corr_s",
]


@dataclass
class RaceData:
    laps: pd.DataFrame        # race laps + `race_pos` (order at the line on that lap)
    fits: pd.DataFrame        # wide: season, round, driver -> offset, plus per-race compound params
    events: pd.DataFrame      # season, round, event_name, event_date, track_id, era_id
    results: pd.DataFrame     # race classification incl. grid

    def races(self) -> pd.DataFrame:
        return self.events[["season", "round", "track_id", "era_id", "event_date"]]


def load_race_data(store: Store, as_of: pd.Timestamp | None = None, session: str = "R") -> RaceData:
    q = store.query
    events = q("SELECT season, round, event_name, event_date, track_id, era_id FROM raw_events")
    events["event_date"] = pd.to_datetime(events["event_date"])
    if as_of is not None:
        events = events[events["event_date"] < as_of]
    keep = events[["season", "round"]]

    laps = q(f"SELECT {', '.join(LAP_COLS)} FROM clean_laps WHERE session = '{session}'")
    laps = laps.merge(keep, on=["season", "round"])
    # order at the line on each lap: rank of the crossing time among cars completing that lap
    laps["race_pos"] = laps.groupby(["season", "round", "lap_number"])["time_s"].rank(method="first")

    fp = q(f"SELECT season, round, param, value FROM features_fit_params WHERE session = '{session}'")
    fp = fp.merge(keep, on=["season", "round"])
    off = fp[fp["param"].str.startswith("offset:")].copy()
    off["driver"] = off["param"].str.slice(len("offset:"))
    fits = off.rename(columns={"value": "offset"})[["season", "round", "driver", "offset"]]
    comp = fp[fp["param"].str.startswith(("deg:", "compound_offset:"))].copy()
    comp[["kind", "compound"]] = comp["param"].str.split(":", n=1, expand=True)
    comp = comp.pivot_table(index=["season", "round", "compound"], columns="kind", values="value").reset_index()
    comp = comp.rename(columns={"deg": "deg", "compound_offset": "base"})
    glob = fp[fp["param"].isin(["fuel_effect", "reference_offset", "sigma"])].pivot_table(
        index=["season", "round"], columns="param", values="value").reset_index()
    fits = fits.merge(glob, on=["season", "round"], how="left")
    laps = laps.merge(comp, on=["season", "round", "compound"], how="left")
    laps = laps.merge(fits[["season", "round", "driver", "offset", "fuel_effect", "reference_offset"]],
                      on=["season", "round", "driver"], how="left")
    # expected lap time (s, relative within the race): car + compound base + deg * age - fuel * lap
    laps["expected_s"] = (laps["offset"] + laps["base"].fillna(0) + laps["deg"] * laps["tyre_life"]
                          - laps["fuel_effect"] * (laps["lap_number"] - 1))

    results = q(f"""SELECT season, round, driver, team, position, grid_position, status, laps
                    FROM raw_results WHERE session = '{session}'""").merge(keep, on=["season", "round"])
    return RaceData(laps=laps, fits=fits, events=events, results=results)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))
