"""OpenF1-backed source (2023+), used when livetiming.formula1.com (FastF1) is
unreachable. Calendar and round numbers come from Jolpica; session data from
OpenF1 (laps, stints, pit, weather, race control, drivers, session results).

Differences from the FastF1 source, all handled here:
- per-lap track status is rebuilt from race-control messages
  (SC/VSC/red from deployment to TRACK CLEAR; sector yellows until sector CLEAR);
- tyre life comes from stints (tyre_age_at_start + laps into the stint);
- lap end times are lap date_start + lap_duration (lap 1: next lap's start);
- deleted laps are parsed from "... DELETED - TRACK LIMITS ... LAP n" messages;
- no per-lap position, finish-line speed or accuracy flag (left null);
- no telemetry window features (car_data is too heavy to pull per lap here).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from f1ml.ingest.base import ScheduledEvent, SessionData
from f1ml.ingest.jolpica import JolpicaClient

log = logging.getLogger(__name__)

OPENF1_FIRST_SEASON = 2023
FINAL_AFTER = pd.Timedelta(hours=8)   # weekend data is cached only once the last session is this old
BASE_URL = "https://api.openf1.org/v1"


# ============================================================================ client
class OpenF1Client:
    def __init__(self, cache_dir: Path | None = None, min_interval_s: float = 0.35,
                 session: requests.Session | None = None, timeout: float = 120.0):
        self.http = session or requests.Session()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_interval_s = min_interval_s
        self.timeout = timeout
        self._last = 0.0

    def _cache_path(self, endpoint: str, params: dict) -> Path | None:
        # Only queries pinned to a meeting/session are cacheable; callers decide whether the
        # data is final yet (see OpenF1Source: a weekend in progress is never cached).
        if not self.cache_dir or not ({"meeting_key", "session_key"} & params.keys()):
            return None
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_dir / endpoint / f"{key}.json"

    def get(self, endpoint: str, *, cache: bool = True, refresh: bool = False, **params) -> list[dict]:
        """GET an OpenF1 endpoint. `cache`: read/write the on-disk cache; `refresh`: ignore any
        cached copy and overwrite it."""
        cp = self._cache_path(endpoint, params) if cache else None
        if cp and cp.exists() and not refresh:
            return json.loads(cp.read_text())
        for attempt in range(6):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            r = self.http.get(f"{BASE_URL}/{endpoint}", params=params, timeout=self.timeout)
            self._last = time.monotonic()
            if r.status_code in (429, 502, 503, 504) and attempt < 5:
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            if r.status_code == 404:
                data = []  # OpenF1 answers 404 {"detail": "No results found."} for empty queries
            else:
                r.raise_for_status()
                data = r.json()
            if isinstance(data, dict):
                data = []
            if cp:
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(data))
            return data
        raise RuntimeError("unreachable")


# ============================================================================ parsing
def _utc(s) -> pd.Series | pd.Timestamp:
    out = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    if isinstance(out, pd.Series):
        return out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out.tz_convert("UTC").tz_localize(None) if out is not pd.NaT else out


def status_timeline(rc: pd.DataFrame) -> tuple[np.ndarray, list[frozenset]]:
    """Change points (datetime64) and the set of active status codes after each one.

    Codes: 2 yellow (any sector), 4 SC, 5 red, 6 VSC deployed, 7 VSC ending.
    """
    if rc is None or rc.empty:
        return np.array([], dtype="datetime64[ns]"), []
    rc = rc.assign(_t=_utc(rc["date"])).dropna(subset=["_t"]).sort_values("_t", kind="mergesort")
    sc = red = track_yellow = False
    vsc: str | None = None
    sectors: set = set()
    times, states = [], []
    for _, m in rc.iterrows():
        cat = str(m.get("category") or "")
        flag = str(m.get("flag") or "").upper()
        scope = str(m.get("scope") or "")
        msg = str(m.get("message") or "").upper()
        if cat == "SafetyCar":
            # 2023-25 wording "VIRTUAL SAFETY CAR DEPLOYED"; 2026 wording "VSC DEPLOYED"
            if "VIRTUAL SAFETY CAR DEPLOYED" in msg or msg.startswith("VSC DEPLOYED"):
                vsc = "6"
            elif "VIRTUAL SAFETY CAR ENDING" in msg or msg.startswith("VSC ENDING"):
                vsc = "7"
            elif "SAFETY CAR DEPLOYED" in msg:
                sc = True
        elif msg.startswith("RED FLAG"):
            # 2026: "RED FLAG - RACE SUSPENDED" arrives with category "Other", not "Flag"
            red = True
        elif cat == "Flag":
            if flag == "RED":
                red = True
            elif flag in ("CLEAR", "GREEN") and scope == "Track" or flag == "CHEQUERED":
                sc = red = track_yellow = False
                vsc = None
                sectors.clear()
            elif flag in ("YELLOW", "DOUBLE YELLOW"):
                if scope == "Sector" and pd.notna(m.get("sector")):
                    sectors.add(int(m["sector"]))
                else:
                    track_yellow = True
            elif flag == "CLEAR" and scope == "Sector" and pd.notna(m.get("sector")):
                sectors.discard(int(m["sector"]))
        else:
            continue
        codes = set()
        if sectors or track_yellow:
            codes.add("2")
        if sc:
            codes.add("4")
        if red:
            codes.add("5")
        if vsc:
            codes.add(vsc)
        times.append(m["_t"])
        states.append(frozenset(codes))
    return np.array(times, dtype="datetime64[ns]"), states


def lap_status(starts: pd.Series, ends: pd.Series, times: np.ndarray, states: list[frozenset]) -> list[str]:
    """FastF1-style status string per lap: every status active at any point of the lap."""
    out = []
    s_arr = starts.to_numpy(dtype="datetime64[ns]")
    e_arr = ends.to_numpy(dtype="datetime64[ns]")
    for s, e in zip(s_arr, e_arr):
        if np.isnat(e):
            out.append("")
            continue
        if np.isnat(s):
            s = e
        i0 = int(np.searchsorted(times, s, side="right")) - 1
        i1 = int(np.searchsorted(times, e, side="left")) - 1
        codes: set = set()
        if i0 < 0:
            codes.add("1")
        for i in range(max(i0, 0), i1 + 1):
            codes |= states[i] or {"1"}
        if i1 < 0:
            codes.add("1")
        out.append("".join(sorted(codes)) or "1")
    return out


_DELETED = re.compile(r"CAR (\d+) \([A-Z]+\) TIME [\d:.]+ DELETED.*?\bLAP (\d+)")


def deleted_laps(rc: pd.DataFrame) -> set[tuple[str, int]]:
    out = set()
    if rc is None or rc.empty:
        return out
    for msg in rc["message"].dropna().astype(str):
        m = _DELETED.search(msg.upper())
        if m and "REINSTATED" not in msg.upper():
            out.add((m.group(1), int(m.group(2))))
    for msg in rc["message"].dropna().astype(str):  # reinstated laps cancel a deletion
        mm = re.search(r"CAR (\d+) \([A-Z]+\) TIME [\d:.]+ .*REINSTATED.*?\bLAP (\d+)", msg.upper())
        if mm:
            out.discard((mm.group(1), int(mm.group(2))))
    return out


def build_laps(laps: pd.DataFrame, stints: pd.DataFrame, pit: pd.DataFrame, rc: pd.DataFrame,
               drivers: pd.DataFrame, t0: pd.Timestamp, is_race: bool) -> pd.DataFrame:
    """OpenF1 frames for ONE session -> f1ml `laps` contract (keys added by the pipeline)."""
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["driver", "driver_number", "lap_number", "lap_time_s", "time_s",
                                     "compound", "tyre_life", "track_status"])
    L = laps.copy()
    L["driver_number"] = L["driver_number"].astype(int).astype(str)
    L["lap_number"] = L["lap_number"].astype(int)
    L = L.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    start = _utc(L["date_start"])
    dur = pd.to_numeric(L["lap_duration"], errors="coerce")
    end = start + pd.to_timedelta(dur, unit="s")
    next_start = start.groupby(L["driver_number"]).shift(-1)
    end = end.fillna(next_start)
    start = start.fillna(end - pd.to_timedelta(dur, unit="s"))

    # tyres from stints
    L["stint"] = np.nan
    L["compound"] = "UNKNOWN"
    L["tyre_life"] = np.nan
    L["fresh_tyre"] = pd.NA
    if stints is not None and not stints.empty:
        S = stints.copy()
        S["driver_number"] = S["driver_number"].astype(int).astype(str)
        for _, s in S.iterrows():
            lo = int(s["lap_start"]) if pd.notna(s.get("lap_start")) else 1
            hi = int(s["lap_end"]) if pd.notna(s.get("lap_end")) else 10_000
            m = (L["driver_number"] == s["driver_number"]) & L["lap_number"].between(lo, hi)
            age0 = float(s["tyre_age_at_start"]) if pd.notna(s.get("tyre_age_at_start")) else np.nan
            L.loc[m, "stint"] = float(s["stint_number"])
            L.loc[m, "compound"] = str(s.get("compound") or "UNKNOWN").upper()
            L.loc[m, "tyre_life"] = age0 + (L.loc[m, "lap_number"] - lo) + 1
            L.loc[m, "fresh_tyre"] = bool(age0 == 0) if np.isfinite(age0) else pd.NA

    # pit in/out
    out_lap = L["is_pit_out_lap"].fillna(False).astype(bool)
    next_is_out = out_lap.groupby(L["driver_number"]).shift(-1, fill_value=False).astype(bool)
    pit_keys = set()
    pit_time = {}
    if pit is not None and not pit.empty:
        for _, p in pit.iterrows():
            if pd.notna(p.get("lap_number")) and pd.notna(p.get("driver_number")):
                k = (str(int(p["driver_number"])), int(p["lap_number"]))
                pit_keys.add(k)
                pit_time[k] = _utc(p.get("date"))
    key = list(zip(L["driver_number"], L["lap_number"]))
    in_lap = pd.Series([k in pit_keys for k in key], index=L.index) | next_is_out
    if not is_race:  # the final lap of a practice/quali run back to the garage
        last = L.groupby("driver_number")["lap_number"].transform("max") == L["lap_number"]
        in_lap |= last
    to_s = lambda ts: (ts - t0).dt.total_seconds()  # noqa: E731
    time_s = to_s(end)
    pit_in = pd.Series([pit_time.get(k) for k in key], index=L.index, dtype="datetime64[ns]")
    pit_in_s = to_s(pit_in).where(in_lap, np.nan).fillna(time_s.where(in_lap))

    times, states = status_timeline(rc)
    dels = deleted_laps(rc)
    drv = {}
    if drivers is not None and not drivers.empty:
        for _, d in drivers.iterrows():
            drv[str(int(d["driver_number"]))] = (d.get("name_acronym"), d.get("team_name"))

    out = pd.DataFrame({
        "driver": [drv.get(n, (n, None))[0] or n for n in L["driver_number"]],
        "driver_number": L["driver_number"],
        "team": [drv.get(n, (None, None))[1] for n in L["driver_number"]],
        "lap_number": L["lap_number"],
        "stint": L["stint"],
        "lap_time_s": dur,
        "time_s": time_s,
        "lap_start_time_s": to_s(start),
        "pit_in_time_s": pit_in_s,
        "pit_out_time_s": to_s(start).where(out_lap),
        "sector1_s": pd.to_numeric(L.get("duration_sector_1"), errors="coerce"),
        "sector2_s": pd.to_numeric(L.get("duration_sector_2"), errors="coerce"),
        "sector3_s": pd.to_numeric(L.get("duration_sector_3"), errors="coerce"),
        "speed_i1": pd.to_numeric(L.get("i1_speed"), errors="coerce"),
        "speed_i2": pd.to_numeric(L.get("i2_speed"), errors="coerce"),
        "speed_fl": np.nan,
        "speed_st": pd.to_numeric(L.get("st_speed"), errors="coerce"),
        "compound": L["compound"],
        "tyre_life": L["tyre_life"],
        "fresh_tyre": L["fresh_tyre"],
        "track_status": lap_status(start, end, times, states),
        "position": np.nan,
        "deleted": [k in dels for k in key],
        "is_accurate": pd.NA,
        "is_personal_best": pd.NA,
        "lap_start_date": start,
    })
    return out[out["time_s"].notna()].reset_index(drop=True)


def build_results(res: pd.DataFrame, drivers: pd.DataFrame, jolpica: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["driver_number", "driver", "driver_id", "team", "team_id", "position", "classified_position",
            "grid_position", "status", "points", "laps", "q1_s", "q2_s", "q3_s", "time_s"]
    if res is None or res.empty:
        return pd.DataFrame(columns=cols)
    drv = {str(int(d["driver_number"])): d for _, d in drivers.iterrows()} if drivers is not None else {}
    rows = []
    for _, r in res.iterrows():
        n = str(int(r["driver_number"]))
        dur = r.get("duration")
        q = list(dur) if isinstance(dur, (list, tuple, np.ndarray)) else None
        status = "DSQ" if r.get("dsq") else "DNS" if r.get("dns") else "DNF" if r.get("dnf") else "Finished"
        d = drv.get(n, {})
        rows.append({
            "driver_number": n, "driver": d.get("name_acronym") if len(d) else None, "driver_id": None,
            "team": d.get("team_name") if len(d) else None, "team_id": None,
            "position": r.get("position"), "classified_position": None, "grid_position": np.nan,
            "status": status, "points": r.get("points"), "laps": r.get("number_of_laps"),
            "q1_s": q[0] if q and len(q) > 0 else np.nan, "q2_s": q[1] if q and len(q) > 1 else np.nan,
            "q3_s": q[2] if q and len(q) > 2 else np.nan,
            "time_s": np.nan if q is not None else dur,
        })
    out = pd.DataFrame(rows, columns=cols)
    if jolpica is not None and not jolpica.empty:
        j = jolpica.set_index("driver_number")
        for c in ("driver_id", "team_id", "grid_position", "classified_position"):
            out[c] = out["driver_number"].map(j[c]).combine_first(out[c]) if c in j else out[c]
        out["status"] = out["driver_number"].map(j["status"]).fillna(out["status"])
    for c in ("q1_s", "q2_s", "q3_s", "time_s", "position", "points", "laps", "grid_position"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_weather(w: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    if w is None or w.empty:
        return pd.DataFrame(columns=["time_s"])
    return pd.DataFrame({
        "time_s": (_utc(w["date"]) - t0).dt.total_seconds(),
        "air_temp": w.get("air_temperature"), "track_temp": w.get("track_temperature"),
        "humidity": w.get("humidity"), "pressure": w.get("pressure"),
        "rainfall": pd.to_numeric(w.get("rainfall"), errors="coerce").fillna(0) > 0,
        "wind_direction": w.get("wind_direction"), "wind_speed": w.get("wind_speed"),
    })


def build_race_control(rc: pd.DataFrame) -> pd.DataFrame:
    if rc is None or rc.empty:
        return pd.DataFrame(columns=["message"])
    return pd.DataFrame({
        "time": _utc(rc["date"]), "lap": rc.get("lap_number"), "category": rc.get("category"),
        "message": rc["message"].astype("string"), "status": None, "flag": rc.get("flag"),
        "scope": rc.get("scope"), "sector": rc.get("sector"),
        "racing_number": rc.get("driver_number").map(lambda v: str(int(v)) if pd.notna(v) else None)
        if "driver_number" in rc else None,
    })


# ============================================================================ source
class OpenF1Source:
    name = "openf1"

    def __init__(self, cache_dir: Path, jolpica: JolpicaClient, client: OpenF1Client | None = None):
        self.client = client or OpenF1Client(Path(cache_dir) / "openf1")
        self.jolpica = jolpica
        self._meeting_cache: dict[tuple[int, str], pd.DataFrame] = {}

    def schedule(self, season: int) -> list[ScheduledEvent]:
        races = self.jolpica.races(season)
        sessions = pd.DataFrame(self.client.get("sessions", year=season)) if season >= OPENF1_FIRST_SEASON \
            else pd.DataFrame()
        if not sessions.empty:
            sessions = sessions[~sessions.get("is_cancelled", False).fillna(False).astype(bool)]
            sessions["_t"] = _utc(sessions["date_start"])
        events = []
        for r in races:
            race_t = pd.Timestamp(f"{r['date']} {r.get('time', '12:00:00Z').rstrip('Z')}")
            evs = sessions[(sessions["_t"] >= race_t - pd.Timedelta(days=4))
                           & (sessions["_t"] <= race_t + pd.Timedelta(hours=6))] if not sessions.empty else sessions
            names = list(evs["session_name"]) if not evs.empty else []
            fmt = "sprint_qualifying" if "Sprint Qualifying" in names else \
                "sprint_shootout" if "Sprint Shootout" in names else "sprint" if "Sprint" in names else "conventional"
            loc = r["Circuit"]["Location"]
            events.append(ScheduledEvent(
                season=season, round=int(r["round"]), event_name=r["raceName"],
                location=loc.get("locality", ""), country=loc.get("country", ""), event_format=fmt,
                event_date=race_t,
                sessions=[(s["session_name"], s["_t"]) for _, s in evs.sort_values("_t").iterrows()],
                circuit_id=r["Circuit"]["circuitId"],
                extra={
                    "meeting_key": int(evs["meeting_key"].iloc[0]) if not evs.empty else None,
                    "session_keys": {s["session_name"]: int(s["session_key"]) for _, s in evs.iterrows()},
                },
            ))
        return events

    def _now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC").tz_localize(None)

    def _final(self, event: ScheduledEvent) -> bool:
        """A weekend's data is treated as final (cacheable) only once its last session started
        more than FINAL_AFTER ago. Caching earlier froze e.g. a post-qualifying snapshot of the
        weekend's laps, so the race was later 'missing' (bug fixed Nov 2026)."""
        starts = [pd.Timestamp(ts) for _, ts in event.sessions if ts is not None]
        if not starts:
            return False
        last = max(starts)
        last = last.tz_convert("UTC").tz_localize(None) if last.tzinfo is not None else last
        return last + FINAL_AFTER < self._now()

    def _meeting(self, meeting_key: int, endpoint: str, cache: bool = True, refresh: bool = False) -> pd.DataFrame:
        k = (meeting_key, endpoint)
        if refresh or k not in self._meeting_cache:
            if len(self._meeting_cache) > 40:
                self._meeting_cache.clear()
            self._meeting_cache[k] = pd.DataFrame(
                self.client.get(endpoint, cache=cache, refresh=refresh, meeting_key=meeting_key))
        return self._meeting_cache[k]

    def load_session(self, event: ScheduledEvent, session_name: str, code: str,
                     telemetry: bool = False) -> SessionData:
        if event.season < OPENF1_FIRST_SEASON:
            raise ValueError(f"OpenF1 has no data before {OPENF1_FIRST_SEASON}")
        sk = event.extra.get("session_keys", {}).get(session_name)
        mk = event.extra.get("meeting_key")
        if sk is None or mk is None:
            raise LookupError(f"no OpenF1 session for {event.season} R{event.round} {session_name}")
        t0 = dict(event.sessions)[session_name]

        final = self._final(event)
        endpoints = ("laps", "stints", "pit", "race_control", "drivers", "weather")

        def of(endpoint, refresh=False):
            df = self._meeting(mk, endpoint, cache=final, refresh=refresh)
            return df[df["session_key"] == sk] if not df.empty and "session_key" in df else df

        laps, stints, pit, rc, drivers, weather = (of(e) for e in endpoints)
        if laps.empty:
            # A copy saved earlier in the weekend can lack later sessions: refetch the whole
            # weekend once, overwriting stale cache files.
            laps, stints, pit, rc, drivers, weather = (of(e, refresh=True) for e in endpoints)
        if laps.empty:
            raise LookupError(f"OpenF1 returned no laps for {event.season} R{event.round} {session_name} "
                              "(not published yet? OpenF1 usually has a race within a few hours)")
        is_race = code in ("R", "S")
        res = pd.DataFrame(self.client.get("session_result", cache=final, session_key=sk))
        jres = None
        if is_race:
            try:
                jres = self.jolpica.results(event.season, event.round, sprint=(code == "S"))
            except requests.RequestException as exc:
                log.warning("jolpica results unavailable: %s", exc)
        results = build_results(res, drivers, jres)
        if telemetry:
            log.info("OpenF1 source: telemetry window features not supported; skipping")
        total = int(results["laps"].max()) if is_race and results["laps"].notna().any() else None
        return SessionData(
            code=code,
            laps=build_laps(laps, stints, pit, rc, drivers, t0, is_race),
            results=results,
            weather=build_weather(weather, t0),
            race_control=build_race_control(rc),
            total_laps=total,
        )
