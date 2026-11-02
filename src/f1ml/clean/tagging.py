"""Lap tagging: decide which laps say something about pace.

Buscombe measures degradation on "green laps, car and fuel taken out" and on
"clear-air laps only". Each lap therefore gets explicit boolean tags instead of
being dropped, so later models can choose (e.g. the dirty-air penalty model
needs exactly the laps the deg model excludes).

Track-status codes (official timing): 1 green, 2 yellow, 4 Safety Car,
5 red flag, 6 VSC deployed, 7 VSC ending. A lap's status string lists every
status seen during that lap (e.g. "124").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "SUPERSOFT", "ULTRASOFT", "HYPERSOFT", "SUPERHARD"}

WEATHER_COLS = ["track_temp", "air_temp", "humidity", "wind_speed", "rainfall"]


def _has_any(status: pd.Series, codes: str) -> pd.Series:
    s = status.fillna("").astype(str)
    return s.apply(lambda v: any(c in v for c in codes)).astype(bool)


def gap_to_car_ahead(laps: pd.DataFrame) -> pd.Series:
    """Time gap (s) at the line to the previous car to cross it, any lap (lapped cars count:
    they are physical traffic). Crossings that end in the pit lane are not traffic."""
    on_track = laps["pit_in_time_s"].isna() & laps["time_s"].notna()
    cross = laps.loc[on_track, ["time_s", "driver"]].sort_values("time_s", kind="mergesort")
    gaps = cross["time_s"].diff()
    same_car = cross["driver"].eq(cross["driver"].shift())
    gaps[same_car] = np.nan
    return gaps.reindex(laps.index)


def attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """As-of join of the most recent weather sample to each lap end."""
    out = laps.drop(columns=[c for c in WEATHER_COLS if c in laps.columns])
    if weather is None or weather.empty or "time_s" not in weather:
        for c in WEATHER_COLS:
            out[c] = np.nan if c != "rainfall" else pd.NA
        return out
    w = weather[["time_s"] + [c for c in WEATHER_COLS if c in weather.columns]].dropna(subset=["time_s"])
    w = w.sort_values("time_s")
    left = out.reset_index().sort_values("time_s")
    merged = pd.merge_asof(left, w, on="time_s", direction="backward")
    merged = merged.set_index("index").sort_index()
    merged.index.name = None
    return merged


def tag_laps(
    laps: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    *,
    dirty_air_gap_s: float = 1.5,
    outlier_pct: float = 0.05,
    compound_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = laps.copy().reset_index(drop=True)
    df = df.sort_values(["driver", "lap_number"], kind="mergesort").reset_index(drop=True)
    status = df["track_status"].fillna("").astype(str)

    df["is_lap1"] = df["lap_number"].eq(1)
    df["is_in_lap"] = df["pit_in_time_s"].notna()
    df["is_out_lap"] = df["pit_out_time_s"].notna()
    df["is_yellow"] = _has_any(status, "2")
    df["is_neutralised"] = _has_any(status, "467")
    df["is_red"] = _has_any(status, "5")

    # Restart lap: first non-neutralised lap after a neutralised/red one, per driver.
    stopped = (df["is_neutralised"] | df["is_red"])
    prev_stopped = stopped.groupby(df["driver"]).shift(fill_value=False).astype(bool)
    df["is_restart_lap"] = prev_stopped & ~stopped
    restart_id = df["is_restart_lap"].astype(int).groupby(df["driver"]).cumsum()
    lap_at_restart = df["lap_number"].where(df["is_restart_lap"]).groupby([df["driver"], restart_id]).transform("first")
    df["laps_since_restart"] = (df["lap_number"] - lap_at_restart).astype("float64")

    comp = df["compound"].fillna("UNKNOWN").astype(str).str.upper()
    df["compound"] = comp
    df["is_wet_tyre"] = comp.isin(WET_COMPOUNDS)
    cmap = {k.upper(): v for k, v in (compound_map or {}).items()}
    df["c_number"] = comp.map(cmap) if cmap else pd.NA

    df = attach_weather(df, weather if weather is not None else pd.DataFrame())

    df["gap_ahead_s"] = gap_to_car_ahead(df)
    df["is_dirty_air"] = df["gap_ahead_s"].lt(dirty_air_gap_s).fillna(False).astype(bool)

    base_ok = (
        df["lap_time_s"].notna() & ~df["is_lap1"] & ~df["is_in_lap"] & ~df["is_out_lap"]
        & ~df["is_yellow"] & ~df["is_neutralised"] & ~df["is_red"] & ~df["is_wet_tyre"]
    )
    stint_key = [df["driver"], df["stint"].fillna(-1)]
    stint_median = df["lap_time_s"].where(base_ok).groupby(stint_key).transform("median")
    too_slow = df["lap_time_s"] > stint_median * (1 + outlier_pct)
    inaccurate = df["is_accurate"].eq(False).fillna(False) if "is_accurate" in df else False
    deleted = df["deleted"].eq(True).fillna(False) if "deleted" in df else False
    df["is_outlier"] = (too_slow.fillna(False) | inaccurate | deleted).astype(bool)

    raining = df["rainfall"].eq(True).fillna(False).astype(bool) if "rainfall" in df else False
    df["is_clean"] = (base_ok & ~df["is_outlier"] & ~raining).astype(bool)
    df["is_clean_free_air"] = (df["is_clean"] & ~df["is_dirty_air"]).astype(bool)
    return df
