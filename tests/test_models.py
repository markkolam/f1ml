"""Model tests on synthetic races with a known generating process (test data, not F1 data)."""

import numpy as np
import pandas as pd
import pytest

from f1ml.eval import race_metrics, walk_forward
from f1ml.features.prerace import ewm_by_key, finish_order
from f1ml.models import LGBMRaceRanker, LogPositionPL, fit_tau, pl_probabilities
from f1ml.models.ranker import _orders


def synthetic_season(n_races=30, n_drivers=20, noise=0.6, seed=0):
    """Latent pace per driver; grid = pace + quali noise; finish = pace + race noise."""
    rng = np.random.default_rng(seed)
    pace = np.sort(rng.normal(0, 1, n_drivers))
    rows = []
    for r in range(n_races):
        q = pace + rng.normal(0, noise, n_drivers)
        race = pace + rng.normal(0, noise, n_drivers)
        grid = q.argsort().argsort() + 1
        fin = race.argsort().argsort() + 1
        for i in range(n_drivers):
            rows.append({"season": 2030, "round": r + 1, "date": pd.Timestamp("2030-03-01") + pd.to_timedelta(7 * r, unit="D"),
                         "driver": f"D{i:02d}", "team": f"T{i // 2}", "grid_pos": float(grid[i]),
                         "quali_pos": float(grid[i]), "quali_gap_pct": float(q[i] - q.min()),
                         "race_pace_ewm": float(pace[i] + rng.normal(0, 0.2)), "finish_order": float(fin[i]),
                         "points": 0.0})
    return pd.DataFrame(rows)


def test_finish_order_ranks_dnfs_by_laps_and_dns_last():
    res = pd.DataFrame({"position": [2, 1, np.nan, np.nan, np.nan], "laps": [50, 50, 30, 45, 0],
                        "status": ["Finished", "Finished", "Retired", "Accident", "Did not start"],
                        "grid_position": [1, 2, 3, 4, 5]})
    assert finish_order(res).tolist() == [2, 1, 4, 3, 5]


def test_ewm_weights_recent_races_more():
    h = pd.DataFrame({"driver": ["A"] * 3, "date": pd.to_datetime(["2030-01-01", "2030-01-08", "2030-01-15"]),
                      "x": [10.0, 10.0, 1.0]})
    v = ewm_by_key(h, "driver", "x", halflife=1.0)["A"]
    assert 1.0 < v < 5.5      # dominated by the latest value


def test_pl_probabilities_are_coherent():
    p = pl_probabilities(np.array([2.0, 1.0, 0.0, -1.0]), tau=1.0, n_sims=20000)
    assert p.p_win.sum() == pytest.approx(1.0)
    assert p.p_podium.sum() == pytest.approx(3.0, abs=1e-9)
    assert (np.diff(p.p_win) < 0).all() and (np.diff(p.exp_pos) > 0).all()
    assert p.p_win.iloc[0] == pytest.approx(np.exp(2) / np.exp([2, 1, 0, -1]).sum())


def test_fit_tau_recovers_true_temperature():
    rng = np.random.default_rng(1)
    true_tau, races = 1.5, []
    for _ in range(300):
        s = rng.normal(0, 1, 12)
        order = np.argsort(-(true_tau * s + rng.gumbel(size=12)))
        races.append(s[order])
    assert fit_tau(races) == pytest.approx(true_tau, rel=0.15)


def test_lgbm_beats_grid_baseline_when_pace_signal_exists():
    df = synthetic_season()
    train, test = df[df["round"] <= 24], df[df["round"] > 24]
    feats = ["grid_pos", "quali_gap_pct", "race_pace_ewm"]
    lg = LGBMRaceRanker(feats, n_estimators=150).fit(train)
    gr = LogPositionPL("grid_pos").fit(train)
    def mean_rho(m):
        p = pd.concat([test, m.predict(test, n_sims=2000)], axis=1)
        return np.mean([race_metrics(g)["spearman"] for _, g in p.groupby(["season", "round"])])
    assert mean_rho(lg) > mean_rho(gr)
    assert lg.tau > 0 and set(lg.importance().index) == set(feats)


def test_walk_forward_never_trains_on_the_future():
    df = synthetic_season(n_races=14)
    seen = []

    class Spy(LogPositionPL):
        def fit(self, d):
            seen.append(d["date"].max())
            return super().fit(d)

    res = walk_forward(df, {"spy": lambda: Spy("grid_pos")}, [2030], min_train_races=10, n_sims=500)
    assert len(res.per_race) == 4
    test_dates = sorted(df[df["round"] > 10]["date"].unique())
    assert all(s < t for s, t in zip(seen, test_dates))
    assert {"spearman", "winner_logloss", "brier_win"} <= set(res.per_race.columns)


def test_orders_follow_finish():
    df = pd.DataFrame({"season": 1, "round": 1, "finish_order": [3, 1, 2]})
    assert _orders(df, np.array([0.3, 0.1, 0.2]))[0].tolist() == [0.1, 0.2, 0.3]


def test_blend_fits_weights_and_keeps_probabilities_coherent():
    df = synthetic_season(n_races=24)
    m = LGBMRaceRanker(["grid_pos", "quali_gap_pct", "race_pace_ewm"], n_estimators=80, tau_top_k=3,
                       blend_with="grid_pos").fit(df[df["round"] <= 20])
    a, b = m.blend_coef
    assert np.isfinite(a) and np.isfinite(b) and m.tau == 1.0
    p = m.predict(df[df["round"] > 20], n_sims=1000)
    assert p.groupby(df.loc[p.index, "round"])["p_win"].sum().round(6).eq(1.0).all()
