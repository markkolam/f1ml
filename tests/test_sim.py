"""Simulator tests on hand-built races where the right answer is known."""

import numpy as np
import pytest

from f1ml.sim import GREEN, RED, SC, Overrides, RaceParams, Strategy, candidate_plans, parse_plan, plan_text, simulate
from f1ml.sim.strategy import stint_time

COMP = ["SOFT", "MEDIUM", "HARD"]


def params(C=2, laps=20, **kw):
    base = dict(drivers=[f"D{i}" for i in range(C)], grid=np.arange(1, C + 1), pace_s=np.zeros(C),
                pace_sd_s=np.zeros(C), compounds=COMP, base_s=np.array([-0.5, 0.0, 0.4]),
                deg_s=np.array([0.1, 0.05, 0.02]), laps=laps, ref_lap_s=90.0, fuel_s_per_lap=0.05,
                lap_noise_s=0.0, hz_intercept=-50.0, hz_lap1=0.0, pit_loss_sd_s=0.0, pit_loss_s={0: 20.0, 1: 10.0, 2: 10.0},
                concertina_s=0.0, start_gain_sd=np.zeros(C), start_gain_mean=np.zeros(C))
    base.update(kw)
    return RaceParams(**base)


def test_single_car_race_time_matches_the_lap_formula():
    p = params(C=1, laps=20)
    s = Strategy(COMP.index("MEDIUM"), [(10, COMP.index("HARD"))])
    r = simulate(p, [s], n_sims=3, seed=0, stop_jitter=0)
    # lap 1: ref + start + base + deg*1; laps 2..20: ref + base + deg*age - fuel*(lap-1); +20 s pit loss on lap 10
    t = 90.0 + 6.0 + 0.0 + 0.05 * 1
    for lap in range(2, 21):
        if lap <= 10:
            t += 90.0 + 0.0 + 0.05 * lap - 0.05 * (lap - 1)
        else:
            t += 90.0 + 0.4 + 0.02 * (lap - 10) - 0.05 * (lap - 1)
    t += 20.0
    assert r.time[0, 0] == pytest.approx(t, abs=1e-6)
    assert (r.n_stops == 1).all()


def test_overtaking_curve_controls_whether_a_faster_car_gets_by():
    # D0 starts ahead but is 1 s/lap slower
    hard = params(C=2, laps=30, pace_s=np.array([1.0, 0.0]), pass_a=-40.0, pass_b=0.1, concertina_s=2.0)
    easy = params(C=2, laps=30, pace_s=np.array([1.0, 0.0]), pass_a=2.0, pass_b=5.0, concertina_s=2.0)
    st = [Strategy(1, [(15, 2)]), Strategy(1, [(15, 2)])]
    stuck = simulate(hard, st, n_sims=50, seed=1, stop_jitter=0)
    free = simulate(easy, st, n_sims=50, seed=1, stop_jitter=0)
    assert (stuck.position[:, 1] == 2).all()          # held behind all race
    assert (free.position[:, 1] == 1).mean() > 0.9


def test_safety_car_closes_the_field_and_cheapens_stops():
    p = params(C=3, laps=20, pace_s=np.array([0.0, 1.0, 2.0]), pass_a=-40.0, pass_b=0.1, restart_gap_s=0.8)
    sched = np.zeros(21, np.int8)
    sched[8:11] = SC
    st = [Strategy(1, [(9, 2)]) for _ in range(3)]
    r = simulate(p, st, n_sims=2, seed=0, overrides=Overrides(neutral=sched, no_reactive=True), stop_jitter=0)
    r_green = simulate(p, st, n_sims=2, seed=0, overrides=Overrides(neutral=np.zeros(21, np.int8)), stop_jitter=0)
    # after the SC the gaps are reset, so the field finishes much closer than under green
    spread = r.time[0].max() - r.time[0].min()
    spread_green = r_green.time[0].max() - r_green.time[0].min()
    assert spread < spread_green


def test_red_flag_gives_a_free_tyre_change():
    p = params(C=1, laps=20)
    sched = np.zeros(21, np.int8)
    sched[9] = RED
    s = Strategy(1, [(12, 2)])
    r = simulate(p, [s], n_sims=1, seed=0, overrides=Overrides(neutral=sched, no_reactive=True), stop_jitter=0)
    assert r.n_stops[0, 0] == 0                        # the change happened under the red flag, for free


def test_retirements_are_classified_by_laps_completed():
    p = params(C=3, laps=20)
    st = [Strategy(1, [(10, 2)]) for _ in range(3)]
    ov = Overrides(neutral=np.zeros(21, np.int8), retire_lap=np.array([np.inf, 5, 12]))
    r = simulate(p, st, n_sims=1, seed=0, overrides=ov, stop_jitter=0)
    assert r.position[0].tolist() == [1, 3, 2]
    assert r.laps_done[0].tolist() == [20, 5, 12]


def test_simulation_is_reproducible_with_a_seed():
    p = params(C=6, laps=30, lap_noise_s=0.3, hz_intercept=-3.0, pace_sd_s=np.full(6, 0.1),
               start_gain_sd=np.full(6, 1.5))
    st = [Strategy(1, [(15, 2)]) for _ in range(6)]
    a = simulate(p, st, n_sims=200, seed=7)
    b = simulate(p, st, n_sims=200, seed=7)
    assert np.array_equal(a.position, b.position) and np.allclose(a.time, b.time, equal_nan=True)
    assert (a.neutral_laps > GREEN).any()              # hazard on: some neutralisations happen


def test_plan_parsing_round_trip():
    s = parse_plan("s-m-h@14,38", COMP)
    assert s.start_compound == 0 and s.stops == [(14, 1), (38, 2)]
    assert plan_text(s, COMP) == "S-M-H@14,38"
    with pytest.raises(ValueError):
        parse_plan("S-M-H@14", COMP)


def test_candidate_plans_find_the_analytic_optimum():
    # one stop, two compounds with the same base and deg: best split is the midpoint
    p = params(C=1, laps=40, base_s=np.array([0.0, 0.0, 0.0]), deg_s=np.array([0.2, 0.1, 0.1]),
               pit_loss_s={0: 100.0, 1: 50.0, 2: 50.0})
    t, best = candidate_plans(p, max_stops=2)[0]
    assert len(best.stops) == 1 and best.start_compound in (1, 2) and best.stops[0][0] == 20
    assert t == pytest.approx(2 * stint_time(0.0, 0.1, 20) + 100.0)


def test_new_era_falls_back_to_the_previous_era():
    import pandas as pd

    from f1ml.sim.params import _era, _track_row

    df = pd.DataFrame({"era_id": ["2017_wide_cars", "2022_ground_effect", "2022_ground_effect"],
                       "track_id": ["monza", "monza", "spa"], "v": [1, 2, 3]})
    assert _era(df, "2022_ground_effect") == "2022_ground_effect"
    assert _era(df, "2026_active_aero") == "2022_ground_effect"     # rule reset: start from last era
    assert _track_row(df, "monza", "2026_active_aero")["v"] == 2
    assert _track_row(df, "madring", "2026_active_aero") is None
