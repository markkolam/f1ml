"""Test fixtures. `synthetic_race` builds a race with KNOWN parameters so tests can
check the cleaning/fit code recovers them; it is test data, not F1 data."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1ml.ingest.base import ScheduledEvent, SessionData

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRUE_FUEL = 0.05
TRUE_DEG = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.025}
COMPOUND_PACE = {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.4}  # compound offset (absorbed by driver/stint in fit? no: see note)


def synthetic_race(n_drivers: int = 12, n_laps: int = 50, seed: int = 0, sc_laps=(20, 21, 22),
                   noise: float = 0.12, compound_offsets: bool = False) -> tuple[pd.DataFrame, dict]:
    """Laps in the f1ml `laps` contract (without keys). One stop per car, mixed strategies.

    With compound_offsets=False every compound has the same base pace, so the
    linear fit's model is exactly the generating model.
    """
    rng = np.random.default_rng(seed)
    drivers = [f"D{i:02d}" for i in range(n_drivers)]
    offsets = {d: 90.0 + 0.15 * i for i, d in enumerate(drivers)}
    strategies = [("MEDIUM", "HARD"), ("SOFT", "HARD"), ("HARD", "MEDIUM"), ("SOFT", "MEDIUM")]
    rows = []
    for i, d in enumerate(drivers):
        c1, c2 = strategies[i % len(strategies)]
        stop = 14 + (i * 3) % 18
        t = 2.0 * i  # grid spread at the start
        life = 0
        for lap in range(1, n_laps + 1):
            comp, stint = (c1, 1) if lap <= stop else (c2, 2)
            life = life + 1 if lap != stop + 1 else 1
            base = offsets[d] - TRUE_FUEL * (lap - 1) + TRUE_DEG[comp] * life
            if compound_offsets:
                base += COMPOUND_PACE[comp]
            status = "1"
            lt = base + rng.normal(0, noise)
            if lap in sc_laps:
                status, lt = "4", base + 25.0
            if lap == 1:
                lt += 6.0
            if lap == stop:
                lt += 4.0
            if lap == stop + 1:
                lt += 18.0
            start = t
            t += lt
            rows.append({
                "driver": d, "driver_number": str(i + 1), "team": f"T{i // 2}",
                "lap_number": lap, "stint": float(stint), "lap_time_s": lt, "time_s": t,
                "lap_start_time_s": start,
                "pit_in_time_s": t - 1.0 if lap == stop else np.nan,
                "pit_out_time_s": start + 1.0 if lap == stop + 1 else np.nan,
                "compound": comp, "tyre_life": float(life), "fresh_tyre": True,
                "track_status": status, "position": np.nan, "deleted": False, "is_accurate": True,
            })
    truth = {"fuel": TRUE_FUEL, "deg": TRUE_DEG, "offsets": offsets}
    return pd.DataFrame(rows), truth


def synthetic_weather(laps: pd.DataFrame) -> pd.DataFrame:
    t = np.arange(0, laps["time_s"].max() + 60, 60.0)
    return pd.DataFrame({"time_s": t, "air_temp": 25.0, "track_temp": np.linspace(45, 40, len(t)),
                         "humidity": 50.0, "pressure": 1010.0, "rainfall": False,
                         "wind_direction": 180.0, "wind_speed": 2.0})


class FakeSource:
    """In-memory Source for pipeline tests."""

    name = "fake"

    def __init__(self, season: int = 2024, location: str = "Budapest"):
        self.season = season
        self.location = location
        self.calls: list[tuple[int, str]] = []

    def schedule(self, season):
        past = pd.Timestamp("2024-07-21 13:00")
        future = pd.Timestamp("2099-01-01")
        return [
            ScheduledEvent(season, 1, "Test GP", self.location, "Hungary", "conventional", past,
                           [("Practice 2", pd.Timestamp("2024-07-19 15:00")), ("Race", past)]),
            ScheduledEvent(season, 2, "Future GP", "Nowhere City", "X", "conventional", future,
                           [("Race", future)]),
        ]

    def load_session(self, event, session_name, code, telemetry=False):
        self.calls.append((event.round, code))
        laps, _ = synthetic_race(seed=event.round + (0 if code == "R" else 7),
                                 n_laps=50 if code == "R" else 20, sc_laps=(20, 21, 22) if code == "R" else ())
        res = pd.DataFrame({"driver_number": laps["driver_number"].unique(),
                            "driver": laps["driver"].unique()})
        return SessionData(code=code, laps=laps, results=res, weather=synthetic_weather(laps),
                           race_control=pd.DataFrame({"message": ["GREEN LIGHT - PIT EXIT OPEN"]}),
                           total_laps=50 if code == "R" else None)


class FakePits:
    def pit_stops(self, season, round_):
        return pd.DataFrame({"season": [season], "round": [round_], "driver_id": ["d00"], "stop": [1],
                             "lap": [14], "duration_s": [21.9], "time_of_day": ["14:30:00"]})


@pytest.fixture
def project(tmp_path) -> Path:
    """A copy of the real config in a temp project root (data goes to tmp)."""
    shutil.copytree(PROJECT_ROOT / "config", tmp_path / "config")
    return tmp_path
