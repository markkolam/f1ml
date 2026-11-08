"""Strategy-layer tests on hand-built races with known answers."""

import numpy as np
import pandas as pd
import pytest

from f1ml.sim import RaceParams, Strategy
from f1ml.strategy import (auto_objective, crossover, deg_robustness, evaluate_plans, objective_value, sc_sweep,
                           sc_sweep_summary, strategy_candidates, team_split)

COMP = ["SOFT", "MEDIUM", "HARD"]


def params(C=6, laps=40, **kw):
    base = dict(drivers=[f"D{i}" for i in range(C)], grid=np.arange(1, C + 1), pace_s=np.linspace(-0.4, 0.4, C),
                pace_sd_s=np.full(C, 0.05), compounds=COMP, base_s=np.array([-0.3, 0.0, 0.3]),
                deg_s=np.array([0.08, 0.05, 0.03]), laps=laps, ref_lap_s=90.0, fuel_s_per_lap=0.05,
                lap_noise_s=0.2, hz_intercept=-4.0, pit_loss_s={0: 21.0, 1: 11.0, 2: 11.0})
    base.update(kw)
    return RaceParams(**base)


def test_objectives():
    pos = np.array([1, 2, 3, 11])
    assert objective_value(pos, "points").tolist() == [25, 18, 15, 0]
    assert objective_value(pos, "win").tolist() == [1, 0, 0, 0]
    assert objective_value(pos, "top10").tolist() == [1, 1, 1, 0]
    with pytest.raises(ValueError):
        objective_value(pos, "vibes")


def test_obviously_better_plan_wins_the_vote():
    p = params(deg_s=np.array([0.02, 0.01, 0.005]))       # very low deg: extra stops are pure loss
    base = [[Strategy(1, [(20, 2)])] for _ in range(6)]
    good = Strategy(1, [(20, 2)])
    bad = Strategy(0, [(10, 1), (20, 2), (30, 0)])
    ev = evaluate_plans(p, base, "D2", [bad, good], "position", n_sims=400)
    t = ev.table
    assert t["plan"].iloc[0] == "M-H@20" and t["vote_share"].iloc[0] > 0.9
    assert t["vs_best"].iloc[0] == 0 and (t["vs_best"] <= 0).all()


def test_common_random_numbers_identical_plans_tie_exactly():
    p = params()
    base = [[Strategy(1, [(20, 2)])] for _ in range(6)]
    s = Strategy(1, [(18, 2)])
    ev = evaluate_plans(p, base, "D0", [s, Strategy(1, [(18, 2)])], "points", n_sims=300)
    assert np.array_equal(ev.values[0], ev.values[1])


def test_crossover_matches_the_analytic_threshold():
    # equal base and deg on every compound, 60 laps, 20 s pit loss:
    # 1 stop: 2 * d * 30*31/2 ; 2 stops: 3 * d * 20*21/2 + 20  -> switch at d = 20 / (930 - 630)
    p = params(C=1, laps=60, base_s=np.zeros(3), deg_s=np.full(3, 0.01), pit_loss_s={0: 20.0, 1: 10.0, 2: 10.0})
    cx = crossover(p, max_stops=2, scale=np.linspace(1, 12, 2201))
    first = cx[(cx.from_stops == 1) & (cx.to_stops == 2)].iloc[0]
    assert first["medium_deg_s_per_lap"] == pytest.approx(20 / 300, abs=0.001)


def test_candidates_include_early_and_late_versions():
    p = params()
    cands = strategy_candidates(p, n_sequences=3, shifts=(-4, 4))
    best = cands[0]
    laps = {tuple(l for l, _ in c.stops) for c in cands if c.start_compound == best.start_compound
            and [x for _, x in c.stops] == [x for _, x in best.stops]}
    l0 = best.stops[0][0]
    assert {(l0 - 4,), (l0,), (l0 + 4,)} <= laps or len(best.stops) > 1


def test_robustness_regret_is_zero_for_the_best_plan_in_each_world():
    p = params()
    base = [[Strategy(1, [(20, 2)])] for _ in range(6)]
    cands = strategy_candidates(p, n_sequences=3, shifts=())
    rob = deg_robustness(p, base, "D1", cands, "position", n_sims=200)
    world_cols = [c for c in rob.columns if c != "max_regret"]
    assert (rob[world_cols].min(axis=0) == 0).all() and (rob[world_cols] >= 0).all().all()


def test_sc_sweep_and_team_split_shapes():
    p = params(laps=30)
    base = [[Strategy(1, [(15, 2)])] for _ in range(6)]
    cands = [Strategy(1, [(12, 2)]), Strategy(1, [(20, 2)])]
    sw = sc_sweep(p, base, "D3", cands, "position", laps=range(5, 9), n_sims=50)
    assert len(sw) == 4 * 2 and set(sw["sc_lap"]) == {5, 6, 7, 8}
    summ = sc_sweep_summary(sw)
    assert set(summ.index) == {"M-H@12", "M-H@20"}
    ts = team_split(p, base, "D0", "D1", cands, n_sims=80)
    assert len(ts) == 4 and ts["team_points"].is_monotonic_decreasing


def test_auto_objective_follows_the_situation():
    p = params(C=10, grid=np.array([6, 1, 2, 3, 4, 5, 7, 8, 9, 10]), pace_s=np.linspace(-0.5, 0.5, 10))
    s = pd.Series({"p_win": 0.1, "p_podium": 0.3, "p_top10": 0.95})
    obj, _ = auto_objective(p, "D0", s)                    # fastest car starting P6
    assert obj == "win"
    assert auto_objective(p, "D1", pd.Series({"p_win": 0.5, "p_podium": 0.8, "p_top10": 1.0}))[0] == "points"
