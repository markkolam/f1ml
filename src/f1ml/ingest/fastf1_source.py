"""FastF1-backed source: official timing (laps, stints, tyres, track status),
results, weather and race-control messages, 2018+.

Needs network access to livetiming.formula1.com (and api.jolpi.ca, which
FastF1 uses for some results/schedule data).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.ingest.base import ScheduledEvent, SessionData

log = logging.getLogger(__name__)


def _td_s(s: pd.Series) -> pd.Series:
    """Timedelta series -> float seconds (NaT -> NaN)."""
    if pd.api.types.is_timedelta64_dtype(s):
        return s.dt.total_seconds()
    return pd.to_numeric(s, errors="coerce")


def map_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """FastF1 Laps -> f1ml `laps` contract columns (keys added by the pipeline)."""
    out = pd.DataFrame({
        "driver": laps["Driver"].astype("string"),
        "driver_number": laps["DriverNumber"].astype("string"),
        "team": laps.get("Team"),
        "lap_number": laps["LapNumber"],
        "stint": laps.get("Stint"),
        "lap_time_s": _td_s(laps["LapTime"]),
        "time_s": _td_s(laps["Time"]),
        "lap_start_time_s": _td_s(laps["LapStartTime"]) if "LapStartTime" in laps else np.nan,
        "pit_in_time_s": _td_s(laps["PitInTime"]) if "PitInTime" in laps else np.nan,
        "pit_out_time_s": _td_s(laps["PitOutTime"]) if "PitOutTime" in laps else np.nan,
        "sector1_s": _td_s(laps["Sector1Time"]) if "Sector1Time" in laps else np.nan,
        "sector2_s": _td_s(laps["Sector2Time"]) if "Sector2Time" in laps else np.nan,
        "sector3_s": _td_s(laps["Sector3Time"]) if "Sector3Time" in laps else np.nan,
        "speed_i1": laps.get("SpeedI1"),
        "speed_i2": laps.get("SpeedI2"),
        "speed_fl": laps.get("SpeedFL"),
        "speed_st": laps.get("SpeedST"),
        "compound": laps["Compound"].fillna("UNKNOWN").astype("string").str.upper(),
        "tyre_life": laps["TyreLife"],
        "fresh_tyre": laps.get("FreshTyre"),
        "track_status": laps["TrackStatus"].fillna("").astype("string"),
        "position": laps.get("Position"),
        "deleted": laps.get("Deleted"),
        "is_accurate": laps.get("IsAccurate"),
        "is_personal_best": laps.get("IsPersonalBest"),
        "lap_start_date": laps.get("LapStartDate"),
    })
    # Laps without a lap number are unusable; laps without a time are kept (in/out laps,
    # red-flag laps) because tagging needs them, but lap_time_s stays NaN.
    out = out[out["lap_number"].notna()]
    out = out[out["time_s"].notna()]
    return out.reset_index(drop=True)


def map_results(res: pd.DataFrame) -> pd.DataFrame:
    def col(name):
        return res[name] if name in res else pd.Series(pd.NA, index=res.index)

    return pd.DataFrame({
        "driver_number": col("DriverNumber").astype("string"),
        "driver": col("Abbreviation").astype("string"),
        "driver_id": col("DriverId"),
        "team": col("TeamName"),
        "team_id": col("TeamId"),
        "position": col("Position"),
        "classified_position": col("ClassifiedPosition").astype("string"),
        "grid_position": col("GridPosition"),
        "status": col("Status"),
        "points": col("Points"),
        "laps": col("Laps"),
        "q1_s": _td_s(col("Q1")),
        "q2_s": _td_s(col("Q2")),
        "q3_s": _td_s(col("Q3")),
        "time_s": _td_s(col("Time")),
    }).reset_index(drop=True)


def map_weather(w: pd.DataFrame) -> pd.DataFrame:
    if w is None or len(w) == 0:
        return pd.DataFrame(columns=["time_s"])
    return pd.DataFrame({
        "time_s": _td_s(w["Time"]),
        "air_temp": w.get("AirTemp"),
        "track_temp": w.get("TrackTemp"),
        "humidity": w.get("Humidity"),
        "pressure": w.get("Pressure"),
        "rainfall": w.get("Rainfall"),
        "wind_direction": w.get("WindDirection"),
        "wind_speed": w.get("WindSpeed"),
    }).reset_index(drop=True)


def map_race_control(rc: pd.DataFrame) -> pd.DataFrame:
    if rc is None or len(rc) == 0:
        return pd.DataFrame(columns=["message"])

    def col(name):
        return rc[name] if name in rc else pd.Series(pd.NA, index=rc.index)

    return pd.DataFrame({
        "time": col("Time"),
        "lap": col("Lap"),
        "category": col("Category"),
        "message": col("Message").astype("string"),
        "status": col("Status"),
        "flag": col("Flag"),
        "scope": col("Scope"),
        "sector": col("Sector"),
        "racing_number": col("RacingNumber").astype("string"),
    }).reset_index(drop=True)


class FastF1Source:
    name = "fastf1"

    def __init__(self, cache_dir: Path, window_band: tuple[float, float] = (70.0, 150.0)):
        import fastf1

        self._ff1 = fastf1
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        self.window_band = window_band

    def schedule(self, season: int) -> list[ScheduledEvent]:
        sched = self._ff1.get_event_schedule(season, include_testing=False)
        events = []
        for _, row in sched.iterrows():
            sessions = []
            for i in range(1, 6):
                name = row.get(f"Session{i}")
                if isinstance(name, str) and name and name.lower() != "none":
                    ts = row.get(f"Session{i}DateUtc")
                    sessions.append((name, pd.Timestamp(ts) if pd.notna(ts) else None))
            events.append(ScheduledEvent(
                season=season,
                round=int(row["RoundNumber"]),
                event_name=str(row["EventName"]),
                location=str(row["Location"]),
                country=str(row.get("Country", "")),
                event_format=str(row.get("EventFormat", "conventional")),
                event_date=pd.Timestamp(row["EventDate"]) if pd.notna(row.get("EventDate")) else None,
                sessions=sessions,
            ))
        return events

    def load_session(self, event: ScheduledEvent, session_name: str, code: str,
                     telemetry: bool = False) -> SessionData:
        s = self._ff1.get_session(event.season, event.round, session_name)
        s.load(laps=True, telemetry=telemetry, weather=True, messages=True)
        # FastF1 logs (rather than raises) when the timing API is unreachable.
        if getattr(s, "_laps", None) is None:
            raise ConnectionError(
                f"FastF1 could not load timing data for {event.season} R{event.round} {session_name}; "
                "check network access to livetiming.formula1.com and api.jolpi.ca"
            )
        laps = map_laps(s.laps)
        window = None
        if telemetry:
            from f1ml.clean.telemetry import lap_window_table
            window = lap_window_table(s.laps, *self.window_band)
        total = getattr(s, "total_laps", None)
        return SessionData(
            code=code,
            laps=laps,
            results=map_results(s.results),
            weather=map_weather(s.weather_data),
            race_control=map_race_control(s.race_control_messages),
            total_laps=int(total) if total else None,
            lap_window=window,
        )
