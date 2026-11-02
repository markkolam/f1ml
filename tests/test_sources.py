import numpy as np
import pandas as pd

from f1ml.contracts import conform
from f1ml.ingest.fastf1_source import map_laps, map_results
from f1ml.ingest.jolpica import JolpicaClient, _duration_s


def test_map_fastf1_laps_to_contract():
    td = pd.to_timedelta
    ff1 = pd.DataFrame({
        "Time": td([95.0, 185.0, np.nan], unit="s"), "Driver": ["VER", "VER", "VER"], "DriverNumber": ["1"] * 3,
        "LapTime": td([95.0, 90.0, np.nan], unit="s"), "LapNumber": [1.0, 2.0, np.nan], "Stint": [1.0, 1.0, 1.0],
        "PitOutTime": td([np.nan] * 3, unit="s"), "PitInTime": td([np.nan] * 3, unit="s"),
        "Sector1Time": td([30.0, 29.0, np.nan], unit="s"), "Sector2Time": td([30.0] * 3, unit="s"),
        "Sector3Time": td([30.0] * 3, unit="s"), "SpeedI1": [300.0] * 3, "SpeedI2": [280.0] * 3,
        "SpeedFL": [290.0] * 3, "SpeedST": [320.0] * 3, "IsPersonalBest": [False, True, False],
        "Compound": ["MEDIUM", "MEDIUM", None], "TyreLife": [1.0, 2.0, 3.0], "FreshTyre": [True] * 3,
        "Team": ["Red Bull Racing"] * 3, "LapStartTime": td([0.0, 95.0, 185.0], unit="s"),
        "LapStartDate": pd.to_datetime(["2024-07-21 13:03"] * 3), "TrackStatus": ["1", "12", "1"],
        "Position": [1.0, 1.0, np.nan], "Deleted": [False, False, None], "IsAccurate": [False, True, False],
    })
    out = map_laps(ff1)
    assert len(out) == 2                                  # lap without number dropped
    assert out.lap_time_s.tolist() == [95.0, 90.0] and out.sector1_s.tolist() == [30.0, 29.0]
    conform(out.assign(season=2024, round=13, session="R"), "laps")


def test_map_results():
    res = pd.DataFrame({"DriverNumber": ["1"], "Abbreviation": ["VER"], "TeamName": ["Red Bull Racing"],
                        "Position": [1.0], "GridPosition": [2.0], "Q3": pd.to_timedelta([75.1], unit="s"),
                        "Status": ["Finished"], "Points": [25.0]})
    out = conform(map_results(res).assign(season=2024, round=1, session="R"), "results")
    assert out.q3_s[0] == 75.1 and out.q1_s.isna().all()


def test_jolpica_duration_and_paging():
    assert _duration_s("22.345") == 22.345 and _duration_s("1:02.5") == 62.5 and np.isnan(_duration_s(None))

    class Resp:
        def __init__(self, payload):
            self.payload, self.status_code = payload, 200

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Http:
        def get(self, url, params, timeout):
            off = params["offset"]
            stops = [{"driverId": f"d{i}", "stop": "1", "lap": str(10 + i), "duration": "22.1", "time": "14:00"}
                     for i in range(off, min(off + 100, 150))]
            return Resp({"MRData": {"total": "150", "RaceTable": {"Races": [{"PitStops": stops}]}}})

    df = JolpicaClient(min_interval_s=0, session=Http()).pit_stops(2024, 13)
    assert len(df) == 150 and df.duration_s.iloc[0] == 22.1
