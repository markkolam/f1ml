"""Component-model tests on synthetic data with known generating parameters (test data, not F1 data)."""

import numpy as np
import pandas as pd
import pytest

from f1ml.clean import tag_laps
from f1ml.components.flags import fit_hazard, hazard_rows
from f1ml.components.overtake import fit_pass_model
from f1ml.components.pace import _filter
from f1ml.components.pitloss import measure_stops, stop_condition
from f1ml.components.pooling import dl_pool, hierarchical
from f1ml.components.tyre import fit_temperature
from tests.conftest import synthetic_race


def test_dl_pool_fixed_and_random_effects():
    mu, se, tau2 = dl_pool([1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
    assert mu == pytest.approx(1.0) and tau2 == 0.0 and se == pytest.approx(0.1 / np.sqrt(3))
    mu, se, tau2 = dl_pool([0.0, 2.0], [0.1, 0.1])
    assert mu == pytest.approx(1.0) and tau2 > 1.5


def test_hierarchical_shrinks_single_race_tracks_more():
    rng = np.random.default_rng(0)
    rows = []
    for t in range(12):
        true = 0.05 + rng.normal(0, 0.01)
        for _ in range(4 if t < 10 else 1):
            rows.append({"track": f"T{t}", "era": "E", "v": true + rng.normal(0, 0.01), "se": 0.003})
    rows.append({"track": "OUT", "era": "E", "v": 0.20, "se": 0.003})  # one freak race at a one-off track
    out = hierarchical(pd.DataFrame(rows), "v", "se", ["track", "era"], ["era"]).set_index("track")
    assert abs(out.loc["OUT", "post_mean"] - 0.05) < abs(out.loc["OUT", "raw_mean"] - 0.05)
    assert out.loc["T0", "post_sd"] < out.loc["T11", "post_sd"]


def _race_frame(**kw):
    laps, _ = synthetic_race(**kw)
    t = tag_laps(laps).assign(season=2030, round=1, track_id="test", era_id="E", offset=0.0)
    return t


def test_pit_loss_recovers_in_plus_out_lap_cost():
    # synthetic stops add +4 s to the in-lap and +18 s to the out-lap
    stops = measure_stops(_race_frame(n_drivers=14, sc_laps=(), noise=0.05))
    assert len(stops) >= 10
    assert stops["loss_s"].median() == pytest.approx(22.0, abs=0.8)
    assert set(stops["condition"]) == {"green"}
    assert stop_condition("14") == "sc" and stop_condition("6") == "vsc" and stop_condition("15") == "red"


def test_pass_model_recovers_track_thresholds():
    rng = np.random.default_rng(1)
    rows = []
    for track, thr in [("easy", 0.6), ("mid", 1.1), ("hard", 2.0)]:
        b = 2.5
        for _ in range(4000):
            d = rng.uniform(-1, 3)
            p = 1 / (1 + np.exp(-b * (d - thr)))
            rows.append({"track_id": track, "era_id": "E", "attacker": f"D{rng.integers(10)}", "delta": d,
                         "passed": float(rng.random() < p)})
    out = fit_pass_model(pd.DataFrame(rows))["overtake_track"].set_index("track_id")
    for track, thr in [("easy", 0.6), ("mid", 1.1), ("hard", 2.0)]:
        assert out.loc[track, "threshold_50"] == pytest.approx(thr, abs=0.15), track


def test_hazard_rows_detect_events_and_restarts():
    laps = _race_frame(n_drivers=6, n_laps=30, sc_laps=(10, 11))
    rows = hazard_rows(laps)
    ev = rows[rows["event"] == 1]
    assert list(ev["lap"]) == [10]                      # SC starts on lap 10
    assert 11 not in set(rows["lap"])                   # lap 11 is not at risk (already neutralised)
    assert rows.set_index("lap").loc[[13, 14], "restart"].eq(1).all()  # 1-2 laps after the lap-12 restart
    assert rows.set_index("lap").loc[1, "lap1"] == 1


def test_hazard_fit_finds_lap1_and_pit_effects():
    rng = np.random.default_rng(2)
    rows = []
    for race in range(2000):
        for lap in range(1, 51):
            pit = 0.3 if 15 <= lap <= 20 else 0.0
            z = -4.5 + 2.0 * (lap == 1) + 3.0 * pit
            rows.append({"season": 2030, "round": race, "era_id": "E", "type": "sc",
                         "track_id": f"T{race % 10}", "lap1": float(lap == 1), "restart": 0.0, "pit_activity": pit,
                         "late": float(lap > 40), "race_frac": lap / 50, "wet": 0.0,
                         "event": float(rng.random() < 1 / (1 + np.exp(-z)))})
    c = fit_hazard(pd.DataFrame(rows))["flags_coefs"].set_index("param")["value"]
    assert c["lap1"] == pytest.approx(2.0, abs=0.4) and c["pit_activity"] == pytest.approx(3.0, abs=0.8)


def test_pace_filter_tracks_a_random_walk_and_is_calibrated():
    rng = np.random.default_rng(3)
    rows, level = [], {"A": -1.0, "B": 0.0, "C": 1.0}
    for i in range(60):
        for t in level:
            level[t] += rng.normal(0, np.sqrt(0.01))
            rows.append({"event_date": pd.Timestamp("2030-01-01") + pd.to_timedelta(7 * i, unit="D"),
                         "season": 2030, "era_id": "E", "round": i + 1, "team": t,
                         "y": level[t] + rng.normal(0, np.sqrt(0.05))})
    pred, now = _filter(pd.DataFrame(rows), "team", q=0.01, r=0.05)
    late = pred[pred["round"] > 10]
    z = (late["obs"] - late["pred_mean"]) / late["pred_sd"]
    assert 0.8 < z.std() < 1.2
    assert np.corrcoef(late["pred_mean"], late["obs"])[0, 1] > 0.9
    assert set(now["team"]) == {"A", "B", "C"}


def test_temperature_slope_learnt_within_track():
    rng = np.random.default_rng(4)
    rows = []
    for t in range(15):
        base = rng.uniform(0.02, 0.12)        # track effect, deliberately uncorrelated with temperature
        t_mean = rng.uniform(25, 50)
        for y in range(3):
            temp = t_mean + rng.normal(0, 5)
            rows.append({"track_id": f"T{t}", "era_id": "E", "compound": "MEDIUM", "track_temp": temp,
                         "deg": base + 0.004 * (temp - t_mean) + rng.normal(0, 0.002), "deg_se": 0.002})
    gamma, se, _ = fit_temperature(pd.DataFrame(rows))
    assert gamma == pytest.approx(0.004, abs=3 * se + 0.0005)
