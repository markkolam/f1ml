"""OpenF1 parsing tests on small hand-built frames shaped like the API responses."""

import numpy as np
import pandas as pd
import pytest

from f1ml.contracts import conform
from f1ml.ingest.openf1_source import (build_laps, build_results, deleted_laps, lap_status,
                                       status_timeline)

T0 = pd.Timestamp("2024-11-03 15:00:00")


def iso(sec):
    return (T0 + pd.to_timedelta(sec, unit="s")).isoformat() + "+00:00"


def rc_frame(rows):
    return pd.DataFrame([dict(zip(["date", "category", "flag", "scope", "sector", "message"], r)) for r in rows])


RC = rc_frame([
    (iso(0), "Flag", "GREEN", "Track", None, "GREEN LIGHT - PIT EXIT OPEN"),
    (iso(150), "Flag", "DOUBLE YELLOW", "Sector", 7, "DOUBLE YELLOW IN TRACK SECTOR 7"),
    (iso(160), "Flag", "CLEAR", "Sector", 7, "CLEAR IN TRACK SECTOR 7"),
    (iso(300), "SafetyCar", None, None, None, "SAFETY CAR DEPLOYED"),
    (iso(500), "SafetyCar", None, None, None, "SAFETY CAR IN THIS LAP"),
    (iso(560), "Flag", "CLEAR", "Track", None, "TRACK CLEAR"),
    (iso(700), "SafetyCar", None, None, None, "VIRTUAL SAFETY CAR DEPLOYED"),
    (iso(760), "SafetyCar", None, None, None, "VIRTUAL SAFETY CAR ENDING"),
    (iso(775), "Flag", "CLEAR", "Track", None, "TRACK CLEAR"),
    (iso(900), "Flag", "RED", "Track", None, "RED FLAG"),
    (iso(1000), "Other", None, None, None, "CAR 1 (VER) TIME 1:29.677 DELETED - TRACK LIMITS AT TURN 4 LAP 5 12:56:34"),
    (iso(1001), "Other", None, None, None, "CAR 4 (NOR) TIME 1:30.000 DELETED - TRACK LIMITS AT TURN 4 LAP 6 12:57:00"),
    (iso(1002), "Other", None, None, None, "CAR 4 (NOR) TIME 1:30.000 REINSTATED - LAP 6"),
])


def test_status_timeline_and_lap_status():
    times, states = status_timeline(RC)
    s = pd.Series(T0 + pd.to_timedelta([10, 140, 320, 540, 680, 765, 800, 910], unit="s"))
    e = pd.Series(T0 + pd.to_timedelta([100, 200, 400, 600, 740, 770, 850, 950], unit="s"))
    got = lap_status(s, e, times, states)
    assert got == ["1", "12", "4", "14", "16", "7", "1", "5"]


def test_deleted_laps_with_reinstatement():
    assert deleted_laps(RC) == {("1", 5)}


def test_build_laps_tyres_pits_and_status():
    laps = pd.DataFrame([
        # driver 1: lap1 has no start/duration (as in OpenF1), pits at end of lap 3
        {"driver_number": 1, "lap_number": 1, "date_start": None, "lap_duration": None, "is_pit_out_lap": False},
        {"driver_number": 1, "lap_number": 2, "date_start": iso(100), "lap_duration": 90.0, "is_pit_out_lap": False},
        {"driver_number": 1, "lap_number": 3, "date_start": iso(190), "lap_duration": 110.0, "is_pit_out_lap": False},
        {"driver_number": 1, "lap_number": 4, "date_start": iso(300), "lap_duration": 120.0, "is_pit_out_lap": True},
        {"driver_number": 1, "lap_number": 5, "date_start": iso(420), "lap_duration": 91.0, "is_pit_out_lap": False},
    ])
    stints = pd.DataFrame([
        {"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 3, "compound": "MEDIUM", "tyre_age_at_start": 2},
        {"driver_number": 1, "stint_number": 2, "lap_start": 4, "lap_end": 5, "compound": "HARD", "tyre_age_at_start": 0},
    ])
    pit = pd.DataFrame([{"driver_number": 1, "lap_number": 3, "date": iso(285), "pit_duration": 22.0}])
    drivers = pd.DataFrame([{"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"}])
    out = build_laps(laps, stints, pit, RC, drivers, T0, is_race=True)
    assert list(out.lap_number) == [1, 2, 3, 4, 5]
    assert out.time_s.iloc[0] == pytest.approx(100.0)            # lap 1 ends where lap 2 starts
    assert list(out.compound) == ["MEDIUM"] * 3 + ["HARD"] * 2
    assert list(out.tyre_life) == [3, 4, 5, 1, 2]               # used set: age 2 at start
    assert out.pit_in_time_s.notna().tolist() == [False, False, True, False, False]
    assert out.pit_out_time_s.notna().tolist() == [False, False, False, True, False]
    assert out.track_status.iloc[3] == "4" and out.track_status.iloc[1] == "12"   # SC from lap start
    assert out.deleted.tolist() == [False, False, False, False, True]
    assert (out.driver == "VER").all()
    conform(out.assign(season=2024, round=21, session="R"), "laps")


def test_build_results_race_and_quali():
    drivers = pd.DataFrame([{"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"},
                            {"driver_number": 4, "name_acronym": "NOR", "team_name": "McLaren"}])
    race = pd.DataFrame([{"position": 1, "driver_number": 1, "number_of_laps": 69, "points": 25.0,
                          "dnf": False, "dns": False, "dsq": False, "duration": 6000.1},
                         {"position": None, "driver_number": 4, "number_of_laps": 30, "points": 0.0,
                          "dnf": True, "dns": False, "dsq": False, "duration": None}])
    jol = pd.DataFrame({"driver_number": ["1", "4"], "driver": ["VER", "NOR"], "driver_id": ["max_verstappen", "norris"],
                        "team_id": ["red_bull", "mclaren"], "grid_position": [17.0, 1.0],
                        "classified_position": ["1", "R"], "status": ["Finished", "Accident"]})
    r = build_results(race, drivers, jol)
    assert r.grid_position.tolist() == [17.0, 1.0] and r.status.tolist() == ["Finished", "Accident"]
    assert r.time_s.iloc[0] == 6000.1
    q = build_results(pd.DataFrame([{"position": 1, "driver_number": 4, "number_of_laps": 20, "dnf": False,
                                     "dns": False, "dsq": False, "duration": [70.1, 69.5, 69.0]}]), drivers, None)
    assert (q.q1_s[0], q.q2_s[0], q.q3_s[0]) == (70.1, 69.5, 69.0) and np.isnan(q.time_s[0])
    conform(r.assign(season=2024, round=21, session="R"), "results")


def test_status_timeline_2026_wording():
    """2026 race control says 'VSC DEPLOYED' and files red flags under category 'Other'."""
    rc = rc_frame([
        (iso(0), "Flag", "GREEN", "Track", None, "GREEN LIGHT - PIT EXIT OPEN"),
        (iso(100), "SafetyCar", None, None, None, "VSC DEPLOYED"),
        (iso(150), "SafetyCar", None, None, None, "VSC ENDING"),
        (iso(160), "Flag", "CLEAR", "Track", None, "TRACK CLEAR"),
        (iso(300), "Other", None, None, None, "RED FLAG - RACE SUSPENDED"),
        (iso(400), "Other", None, None, None, "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 5 (BOR) - VSC INFRINGEMENT"),
    ])
    times, states = status_timeline(rc)
    s = pd.Series(T0 + pd.to_timedelta([110, 152, 200, 310, 410], unit="s"))
    e = pd.Series(T0 + pd.to_timedelta([140, 158, 250, 350, 450], unit="s"))
    assert lap_status(s, e, times, states) == ["6", "7", "1", "5", "5"]


class _FakeHTTP:
    """Serves meeting-level OpenF1 data for sessions 10 (qualifying) and 11 (race)."""

    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url.rsplit("/", 1)[-1], dict(params)))
        ep = url.rsplit("/", 1)[-1]
        rows = []
        if ep == "laps":
            for sk in (10, 11):
                for lap in (1, 2, 3):
                    rows.append({"session_key": sk, "driver_number": 1, "lap_number": lap,
                                 "date_start": iso(100 * lap), "lap_duration": 90.0, "is_pit_out_lap": False})
        if ep == "stints":
            rows = [{"session_key": sk, "driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 3,
                     "compound": "MEDIUM", "tyre_age_at_start": 0} for sk in (10, 11)]

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self_inner):
                return rows
        return R()


def _source(tmp_path, now):
    import pandas as pd

    from f1ml.ingest.base import ScheduledEvent
    from f1ml.ingest.openf1_source import OpenF1Client, OpenF1Source

    class NoJolpica:
        def results(self, *a, **k):
            return pd.DataFrame()

    http = _FakeHTTP()
    src = OpenF1Source(tmp_path, NoJolpica(), client=OpenF1Client(tmp_path / "openf1", min_interval_s=0, session=http))
    src._now = lambda: now
    ev = ScheduledEvent(2026, 20, "Test GP", "Somewhere", "X", "conventional", T0,
                        [("Qualifying", T0 - pd.Timedelta(days=1)), ("Race", T0)],
                        extra={"meeting_key": 1, "session_keys": {"Qualifying": 10, "Race": 11}})
    return src, ev, http


def test_stale_post_qualifying_cache_is_refreshed(tmp_path):
    """Regression: a weekend-level cache saved after qualifying lacked the race, so the race
    'had no laps' (LookupError) after the weekend. It must refetch and heal the cache."""
    import hashlib
    import json

    src, ev, http = _source(tmp_path, now=T0 + pd.Timedelta(days=2))
    key = hashlib.sha1(json.dumps({"meeting_key": 1}, sort_keys=True).encode()).hexdigest()[:16]
    stale = tmp_path / "openf1" / "laps" / f"{key}.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps([{"session_key": 10, "driver_number": 1, "lap_number": 1, "date_start": iso(100),
                                  "lap_duration": 90.0, "is_pit_out_lap": False}]))
    data = src.load_session(ev, "Race", "R")
    assert len(data.laps) == 3
    assert any(x["session_key"] == 11 for x in json.loads(stale.read_text()))   # cache healed


def test_weekend_in_progress_is_not_cached(tmp_path):
    src, ev, http = _source(tmp_path, now=T0 - pd.Timedelta(hours=12))   # before the race
    src.load_session(ev, "Qualifying", "Q")
    assert not list((tmp_path / "openf1").rglob("*.json"))
