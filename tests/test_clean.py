import numpy as np
import pandas as pd
import pytest

from f1ml.clean import apply_corrections, extract_long_runs, fit_race, tag_laps
from f1ml.clean.telemetry import window_stats
from tests.conftest import TRUE_DEG, TRUE_FUEL, synthetic_race, synthetic_weather


@pytest.fixture
def race():
    laps, truth = synthetic_race()
    return tag_laps(laps, synthetic_weather(laps)), truth


def test_tags(race):
    df, _ = race
    assert df.loc[df.lap_number == 1, "is_lap1"].all()
    sc = df[df.lap_number.isin([20, 21, 22])]
    assert sc["is_neutralised"].all() and not sc["is_clean"].any()
    assert df.loc[df.lap_number == 23, "is_restart_lap"].all()
    assert (df.loc[df.lap_number == 25, "laps_since_restart"] == 2).all()
    assert df["is_in_lap"].sum() == df["driver"].nunique() == df["is_out_lap"].sum()
    assert not df.loc[df.is_in_lap | df.is_out_lap, "is_clean"].any()
    assert df["track_temp"].notna().all()
    # dirty air flags exist and are excluded from the free-air set only
    assert df["is_dirty_air"].any()
    assert not (df.is_clean_free_air & df.is_dirty_air).any()


def test_gap_ahead_counts_physical_traffic():
    laps = pd.DataFrame({
        "driver": ["A", "B", "C", "A"], "lap_number": [5, 5, 4, 6], "time_s": [100.0, 100.8, 103.0, 190.0],
        "pit_in_time_s": [np.nan, np.nan, np.nan, np.nan], "pit_out_time_s": np.nan, "lap_time_s": 90.0,
        "track_status": "1", "compound": "SOFT", "tyre_life": 5.0, "stint": 1.0,
    })
    df = tag_laps(laps)
    gaps = df.set_index(["driver", "lap_number"])["gap_ahead_s"]
    assert gaps[("B", 5)] == pytest.approx(0.8)
    assert gaps[("C", 4)] == pytest.approx(2.2)     # lapped car C is behind B on the road
    assert df.set_index(["driver", "lap_number"]).loc[("B", 5), "is_dirty_air"]


def test_wet_and_outliers():
    laps, _ = synthetic_race(n_drivers=2, n_laps=10, sc_laps=())
    laps.loc[(laps.driver == "D00") & (laps.lap_number == 5), "lap_time_s"] += 10  # a mistake
    laps.loc[laps.driver == "D01", "compound"] = "INTERMEDIATE"
    df = tag_laps(laps)
    assert df.set_index(["driver", "lap_number"]).loc[("D00", 5), "is_outlier"]
    assert not df.loc[df.driver == "D01", "is_clean"].any()


def test_fit_recovers_fuel_and_deg(race):
    df, truth = race
    fit = fit_race(df)
    assert fit is not None
    assert fit.fuel_effect == pytest.approx(TRUE_FUEL, abs=0.01)
    for c, d in TRUE_DEG.items():
        assert fit.deg[c] == pytest.approx(d, abs=0.012), c
    # relative car offsets recovered (0.15 s steps)
    assert fit.offsets["D05"] - fit.offsets["D00"] == pytest.approx(0.75, abs=0.08)
    assert {"fuel_effect", "sigma", "deg:SOFT", "offset:D00"} <= set(fit.params.param)


def test_corrections(race):
    df, _ = race
    fit = fit_race(df)
    out = apply_corrections(df, fit, 50)
    clean = out[out.is_clean_free_air]
    # after removing fuel and car, remaining variation is mostly tyre deg + noise
    resid = clean.lap_time_car_fuel_corr_s - clean.compound.map(fit.deg) * clean.tyre_life
    assert resid.std() < 0.25
    assert apply_corrections(df, None, None)["lap_time_fuel_corr_s"].isna().all()


def test_fit_needs_enough_laps():
    laps, _ = synthetic_race(n_drivers=2, n_laps=10, sc_laps=())
    assert fit_race(tag_laps(laps), min_laps=60) is None


def test_long_runs():
    laps, _ = synthetic_race(n_drivers=4, n_laps=40, sc_laps=(), noise=0.02)
    lr = extract_long_runs(tag_laps(laps), min_laps=5, fuel_effect_prior=TRUE_FUEL)
    assert len(lr) >= 4 and (lr.n_laps >= 5).all()
    # raw slope = deg - fuel; adding the fuel prior gives the deg back
    for _, r in lr.iterrows():
        assert r.deg_fuel_adj_s_per_lap == pytest.approx(TRUE_DEG[r.compound], abs=0.02)


def test_window_stats():
    t = np.arange(0, 100.0, 1.0)
    v = np.where(t < 30, 100.0, 250.0)   # 30 s in the 70-150 band out of 99 s
    th = np.where(t < 30, 40.0, 100.0)
    s = window_stats(t, v, th)
    assert s["window_frac"] == pytest.approx(30 / 99)
    assert s["full_throttle_frac"] == pytest.approx(69 / 99)
    assert s["min_speed_kph"] == 100.0


def test_fit_separates_compound_pace_from_deg():
    from tests.conftest import COMPOUND_PACE

    laps, _ = synthetic_race(n_drivers=16, compound_offsets=True, seed=3)
    fit = fit_race(tag_laps(laps))
    ref = [c for c, v in fit.compound_offsets.items() if v == 0.0][0]
    for c, d in TRUE_DEG.items():
        assert fit.deg[c] == pytest.approx(d, abs=0.012), c
        assert fit.compound_offsets[c] == pytest.approx(COMPOUND_PACE[c] - COMPOUND_PACE[ref], abs=0.12), c
