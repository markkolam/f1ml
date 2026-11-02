"""Source abstraction: anything that can provide a season schedule and session data.

The pipeline only talks to this interface, so FastF1 can be swapped or
complemented (OpenF1 live mode in a later phase) and tests can run on fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

# Canonical session codes.
SESSION_CODES = ("FP1", "FP2", "FP3", "SQ", "S", "Q", "R")


def session_code(name: str, event_format: str = "conventional") -> str | None:
    """Map a timing-data session name to a canonical code.

    'Sprint Qualifying' was the name of the sprint *race* in 2021 (event format
    'sprint'), and of the sprint *shootout* from 2024; the event format decides.
    """
    n = (name or "").strip().lower()
    fmt = (event_format or "").strip().lower()
    mapping = {
        "practice 1": "FP1", "practice 2": "FP2", "practice 3": "FP3",
        "qualifying": "Q", "race": "R", "sprint": "S",
        "sprint shootout": "SQ",
    }
    if n == "sprint qualifying":
        return "S" if fmt == "sprint" else "SQ"
    return mapping.get(n)


@dataclass
class ScheduledEvent:
    season: int
    round: int
    event_name: str
    location: str
    country: str
    event_format: str
    event_date: pd.Timestamp | None
    #: [(session name, session start UTC)]
    sessions: list[tuple[str, pd.Timestamp | None]] = field(default_factory=list)
    #: stable circuit id (Jolpica circuitId) when the source knows it; preferred over `location`
    circuit_id: str | None = None
    #: source-specific handles (e.g. OpenF1 meeting/session keys)
    extra: dict = field(default_factory=dict)

    def session_codes(self) -> list[tuple[str, str, pd.Timestamp | None]]:
        out = []
        for name, ts in self.sessions:
            code = session_code(name, self.event_format)
            if code:
                out.append((code, name, ts))
        return out


@dataclass
class SessionData:
    """One session, already mapped to the f1ml contracts (minus season/round/session keys)."""

    code: str
    laps: pd.DataFrame
    results: pd.DataFrame
    weather: pd.DataFrame
    race_control: pd.DataFrame
    total_laps: int | None = None
    lap_window: pd.DataFrame | None = None  # only when telemetry was requested


class Source(Protocol):
    name: str

    def schedule(self, season: int) -> list[ScheduledEvent]: ...

    def load_session(self, event: ScheduledEvent, session_name: str, code: str,
                     telemetry: bool = False) -> SessionData: ...


class PitStopSource(Protocol):
    def pit_stops(self, season: int, round_: int) -> pd.DataFrame: ...
