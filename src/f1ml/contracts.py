"""Data contracts: the columns and dtypes each stored table must have.

Required columns must be present (a missing one is an error). Optional columns
are added as nulls when absent, which is how era-specific fields (e.g. 2026
energy data) coexist with older seasons. Unknown extra columns are kept, after
the contracted ones, so a new source field is never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1ml import SCHEMA_VERSION

_DTYPES = {
    "int": "Int64",
    "float": "float64",
    "str": "string",
    "bool": "boolean",
    "datetime": "datetime64[ns]",
}


@dataclass(frozen=True)
class Col:
    name: str
    kind: str
    required: bool = True


KEY = [Col("season", "int"), Col("round", "int"), Col("session", "str")]

CONTRACTS: dict[str, list[Col]] = {
    "events": [
        Col("season", "int"), Col("round", "int"), Col("event_name", "str"), Col("location", "str"),
        Col("country", "str", False), Col("event_format", "str"), Col("event_date", "datetime", False),
        Col("track_id", "str"), Col("track_version", "str"), Col("track_known", "bool"), Col("era_id", "str"),
        Col("sessions", "str"),
    ],
    "laps": KEY + [
        Col("driver", "str"), Col("driver_number", "str"), Col("team", "str", False),
        Col("lap_number", "int"), Col("stint", "float", False),
        Col("lap_time_s", "float"), Col("time_s", "float"), Col("lap_start_time_s", "float", False),
        Col("pit_in_time_s", "float", False), Col("pit_out_time_s", "float", False),
        Col("sector1_s", "float", False), Col("sector2_s", "float", False), Col("sector3_s", "float", False),
        Col("speed_i1", "float", False), Col("speed_i2", "float", False),
        Col("speed_fl", "float", False), Col("speed_st", "float", False),
        Col("compound", "str"), Col("tyre_life", "float"), Col("fresh_tyre", "bool", False),
        Col("track_status", "str"), Col("position", "float", False),
        Col("deleted", "bool", False), Col("is_accurate", "bool", False), Col("is_personal_best", "bool", False),
        Col("lap_start_date", "datetime", False),
    ],
    "results": KEY + [
        Col("driver_number", "str"), Col("driver", "str"), Col("driver_id", "str", False),
        Col("team", "str", False), Col("team_id", "str", False),
        Col("position", "float", False), Col("classified_position", "str", False),
        Col("grid_position", "float", False), Col("status", "str", False), Col("points", "float", False),
        Col("laps", "float", False), Col("q1_s", "float", False), Col("q2_s", "float", False),
        Col("q3_s", "float", False), Col("time_s", "float", False),
    ],
    "weather": KEY + [
        Col("time_s", "float"), Col("air_temp", "float", False), Col("track_temp", "float", False),
        Col("humidity", "float", False), Col("pressure", "float", False), Col("rainfall", "bool", False),
        Col("wind_direction", "float", False), Col("wind_speed", "float", False),
    ],
    "race_control": KEY + [
        Col("time", "datetime", False), Col("lap", "float", False), Col("category", "str", False),
        Col("message", "str"), Col("status", "str", False), Col("flag", "str", False),
        Col("scope", "str", False), Col("sector", "float", False), Col("racing_number", "str", False),
    ],
    "pit_stops": [
        Col("season", "int"), Col("round", "int"), Col("driver_id", "str"), Col("stop", "int"),
        Col("lap", "int"), Col("duration_s", "float", False), Col("time_of_day", "str", False),
    ],
    "clean_laps": [],  # = laps + tag/derived columns; filled in below
    "fit_params": KEY + [
        Col("param", "str"), Col("value", "float"), Col("se", "float", False),
        Col("n_laps", "int", False), Col("method", "str"),
    ],
    "long_runs": KEY + [
        Col("driver", "str"), Col("team", "str", False), Col("stint", "float"), Col("compound", "str"),
        Col("c_number", "str", False), Col("n_laps", "int"), Col("first_lap", "int"),
        Col("start_tyre_life", "float"), Col("mean_lap_s", "float"), Col("median_lap_s", "float"),
        Col("slope_s_per_lap", "float"), Col("deg_fuel_adj_s_per_lap", "float"),
        Col("mean_track_temp", "float", False),
    ],
    "lap_window": KEY + [
        Col("driver", "str"), Col("lap_number", "int"), Col("window_frac", "float"),
        Col("full_throttle_frac", "float", False), Col("min_speed_kph", "float", False),
    ],
}

CONTRACTS["clean_laps"] = CONTRACTS["laps"] + [
    Col("track_id", "str"), Col("track_version", "str"), Col("era_id", "str"), Col("c_number", "str", False),
    Col("is_lap1", "bool"), Col("is_in_lap", "bool"), Col("is_out_lap", "bool"),
    Col("is_yellow", "bool"), Col("is_neutralised", "bool"), Col("is_red", "bool"),
    Col("is_restart_lap", "bool"), Col("laps_since_restart", "float", False),
    Col("is_wet_tyre", "bool"), Col("rainfall", "bool", False),
    Col("gap_ahead_s", "float", False), Col("is_dirty_air", "bool"), Col("is_outlier", "bool"),
    Col("is_clean", "bool"), Col("is_clean_free_air", "bool"),
    Col("track_temp", "float", False), Col("air_temp", "float", False), Col("humidity", "float", False),
    Col("wind_speed", "float", False),
    Col("lap_time_fuel_corr_s", "float", False), Col("lap_time_car_fuel_corr_s", "float", False),
]


class ContractError(ValueError):
    pass


def conform(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Validate and coerce `df` to the contract of `table`. Returns a new frame."""
    if table not in CONTRACTS:
        raise ContractError(f"no contract for table {table!r}")
    cols = CONTRACTS[table]
    missing = [c.name for c in cols if c.required and c.name not in df.columns]
    if missing:
        raise ContractError(f"{table}: missing required columns {missing}")
    out = df.copy()
    for c in cols:
        if c.name not in out.columns:
            null = {"float": np.nan, "datetime": pd.NaT}.get(c.kind, pd.NA)
            out[c.name] = pd.Series(null, index=out.index, dtype=_DTYPES[c.kind])
            continue
        try:
            out[c.name] = _coerce(out[c.name], c.kind)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{table}.{c.name}: cannot coerce to {c.kind}: {exc}") from exc
    known = [c.name for c in cols]
    extras = [c for c in out.columns if c not in known]
    out = out[known + extras]
    out.attrs["schema_version"] = SCHEMA_VERSION
    return out.reset_index(drop=True)


def _coerce(s: pd.Series, kind: str) -> pd.Series:
    if kind == "datetime":
        s = pd.to_datetime(s, errors="coerce")
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s.astype("datetime64[ns]")
    if kind == "int":
        return pd.to_numeric(s, errors="raise").astype("Int64")
    if kind == "float":
        return pd.to_numeric(s, errors="coerce").astype("float64")
    if kind == "bool":
        return s.astype("boolean")
    if kind == "str":
        return s.astype("string")
    raise ContractError(f"unknown kind {kind}")
