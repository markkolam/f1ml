"""Jolpica-F1 (Ergast successor) client: season calendar (official round numbers,
circuit ids), classified results (grid, status) and pit-stop records.
"""

from __future__ import annotations

import time

import pandas as pd
import requests


class JolpicaClient:
    def __init__(self, base_url: str = "https://api.jolpi.ca/ergast/f1", min_interval_s: float = 0.3,
                 session: requests.Session | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.min_interval_s = min_interval_s
        self.http = session or requests.Session()
        self.timeout = timeout
        self._last = 0.0

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(5):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            r = self.http.get(f"{self.base_url}/{path}", params=params, timeout=self.timeout)
            self._last = time.monotonic()
            if r.status_code == 429 and attempt < 4:
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    def _paged(self, path: str, extract) -> list:
        items, offset, limit = [], 0, 100
        while True:
            data = self._get(path, {"limit": limit, "offset": offset})["MRData"]
            batch = extract(data)
            items.extend(batch)
            total = int(data.get("total", 0))
            offset += limit
            if offset >= total or not batch:
                return items

    # ------------------------------------------------------------------ calendar
    def races(self, season: int) -> list[dict]:
        return self._paged(f"{season}.json", lambda d: d.get("RaceTable", {}).get("Races", []))

    # ------------------------------------------------------------------ results
    def results(self, season: int, round_: int, sprint: bool = False) -> pd.DataFrame:
        key = "SprintResults" if sprint else "Results"
        path = f"{season}/{round_}/{'sprint' if sprint else 'results'}.json"

        def extract(d):
            races = d.get("RaceTable", {}).get("Races", [])
            return races[0].get(key, []) if races else []

        rows = self._paged(path, extract)
        return pd.DataFrame([{
            "driver_number": str(r.get("number")),
            "driver": r.get("Driver", {}).get("code"),
            "driver_id": r.get("Driver", {}).get("driverId"),
            "team_id": r.get("Constructor", {}).get("constructorId"),
            "grid_position": float(r["grid"]) if r.get("grid") not in (None, "") else float("nan"),
            "classified_position": r.get("positionText"),
            "status": r.get("status"),
        } for r in rows], columns=["driver_number", "driver", "driver_id", "team_id", "grid_position",
                                    "classified_position", "status"])

    def pit_stops(self, season: int, round_: int) -> pd.DataFrame:
        def extract(d):
            races = d.get("RaceTable", {}).get("Races", [])
            return races[0].get("PitStops", []) if races else []

        rows = self._paged(f"{season}/{round_}/pitstops.json", extract)
        df = pd.DataFrame([{
            "season": season,
            "round": round_,
            "driver_id": r.get("driverId"),
            "stop": int(r.get("stop")),
            "lap": int(r.get("lap")),
            "duration_s": _duration_s(r.get("duration")),
            "time_of_day": r.get("time"),
        } for r in rows])
        if df.empty:
            df = pd.DataFrame(columns=["season", "round", "driver_id", "stop", "lap", "duration_s", "time_of_day"])
        return df


def _duration_s(v) -> float:
    """'22.345' or '1:02.345' (long stops / red flags) -> seconds."""
    if v is None:
        return float("nan")
    s = str(v)
    try:
        if ":" in s:
            m, sec = s.split(":", 1)
            return int(m) * 60 + float(sec)
        return float(s)
    except ValueError:
        return float("nan")
