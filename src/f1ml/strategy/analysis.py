"""Strategy analyses built on plan evaluation.

- `sc_sweep`: Buscombe's "I ran one Safety Car on every lap from 3 to 53, for
  each strategy against the three rivals": force a Safety Car on each lap in
  turn and see which plan comes out ahead, and how often.
- `deg_robustness`: "Same car, same pit loss, two tyre models ... Nothing changed
  except what you believe about deg." Re-evaluate plans under alternative tyre
  beliefs and report each plan's regret; the minimax-regret plan is the robust one.
- `crossover`: "the line between a one-stop and a two-stop is a tyre deg of about
  0.07s a lap": the deg level at which adding a stop starts to pay.
- `best_response`: "build your race on what your competitors will do": drivers in
  turn switch to their best answer to everyone else's current plan, until stable.
- `team_split`: "Teams that pair strategies win championships. Teams that copy
  each other onto the same tyre share the same result."
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd

from f1ml.sim.engine import GREEN, SC, VSC, Overrides, RaceParams, Strategy, simulate
from f1ml.sim.strategy import candidate_plans, plan_text
from f1ml.strategy.evaluate import DEFAULT_POINTS, evaluate_plans, objective_value, points_of


# ---------------------------------------------------------------------------- Safety Car on every lap
def sc_sweep(p: RaceParams, base: list, driver: str, candidates: list[Strategy], objective: str = "points",
             laps: range | None = None, kind: int = SC, duration: int | None = None, n_sims: int = 300,
             seed: int = 0) -> pd.DataFrame:
    """Objective of each plan with a forced neutralisation starting on each lap (no other incidents)."""
    laps = laps or range(3, p.laps - 2)
    dur = duration or int(np.median(p.durations.get(kind, [4])))
    rows = []
    for lap in laps:
        sched = np.full(p.laps + 1, GREEN, np.int8)
        sched[lap:min(lap + dur, p.laps + 1)] = kind
        ev = evaluate_plans(p, base, driver, candidates, objective, n_sims=n_sims, seed=seed,
                            overrides=Overrides(neutral=sched))
        t = ev.table.set_index("plan")
        for plan in t.index:
            rows.append({"sc_lap": lap, "plan": plan, "objective": t.loc[plan, "objective"],
                         "mean_pos": t.loc[plan, "mean_pos"], "best": plan == ev.table["plan"].iloc[0]})
    return pd.DataFrame(rows)


def sc_sweep_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per plan: share of SC laps on which it was best, and its mean objective across SC laps."""
    g = sweep.groupby("plan")
    out = pd.DataFrame({"best_on_share": g["best"].mean(), "mean_objective": g["objective"].mean(),
                        "worst_objective": g["objective"].min()})
    return out.sort_values("best_on_share", ascending=False)


# ---------------------------------------------------------------------------- tyre-model robustness
DEG_SCENARIOS = {
    "as fitted": {"SOFT": 1.0, "MEDIUM": 1.0, "HARD": 1.0},
    "lower deg (x0.7)": {"SOFT": 0.7, "MEDIUM": 0.7, "HARD": 0.7},
    "higher deg (x1.3)": {"SOFT": 1.3, "MEDIUM": 1.3, "HARD": 1.3},
    "soft falls off, hard holds": {"SOFT": 1.5, "MEDIUM": 1.0, "HARD": 0.7},
    "all compounds deg alike": "equal",
}


def _scenario_params(p: RaceParams, scen) -> RaceParams:
    q = copy.copy(p)
    deg = p.deg_s.copy()
    if scen == "equal":
        deg[:] = deg.mean()
    else:
        for i, c in enumerate(p.compounds):
            deg[i] *= scen.get(c, 1.0)
    q.deg_s = deg
    return q


def deg_robustness(p: RaceParams, base: list, driver: str, candidates: list[Strategy], objective: str = "points",
                   n_sims: int = 1000, seed: int = 0, scenarios: dict | None = None) -> pd.DataFrame:
    """Objective of each plan under each tyre belief, and its regret vs the best plan in that world.
    Rivals keep the plans they would pick under the *fitted* model (their beliefs don't change)."""
    scenarios = scenarios or DEG_SCENARIOS
    rows = []
    for name, scen in scenarios.items():
        ev = evaluate_plans(_scenario_params(p, scen), base, driver, candidates, objective, n_sims, seed)
        best = ev.table["objective"].max()
        for _, r in ev.table.iterrows():
            rows.append({"scenario": name, "plan": r["plan"], "objective": r["objective"],
                         "regret": best - r["objective"]})
    df = pd.DataFrame(rows)
    wide = df.pivot(index="plan", columns="scenario", values="regret")
    wide["max_regret"] = wide.max(axis=1)
    return wide.sort_values("max_regret")


# ---------------------------------------------------------------------------- crossover thresholds
def crossover(p: RaceParams, reference: str = "MEDIUM", max_stops: int = 3,
              scale: np.ndarray | None = None) -> pd.DataFrame:
    """Scale every compound's deg together and record where the best stop count changes.
    Reported as the reference compound's deg (s/lap) at each switch."""
    scale = np.linspace(0.2, 4.0, 191) if scale is None else scale
    ref_i = p.compounds.index(reference)
    prev, rows = None, []
    for m in scale:
        q = copy.copy(p)
        q.deg_s = p.deg_s * m
        best = candidate_plans(q, max_stops=max_stops, top=1)[0][1]
        n = len(best.stops)
        if prev is not None and n != prev[0]:
            rows.append({"from_stops": prev[0], "to_stops": n, f"{reference.lower()}_deg_s_per_lap": q.deg_s[ref_i],
                         "deg_multiplier": m, "plan_after": plan_text(best, p.compounds)})
        prev = (n, m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- rival responses
def best_response(p: RaceParams, base: list, drivers: list[str], objectives: dict[str, str] | None = None,
                  n_sims: int = 600, rounds: int = 3, seed: int = 0, n_candidates: int = 6) -> pd.DataFrame:
    """Iterated best response among `drivers`; others keep their default plan pools."""
    from f1ml.strategy.evaluate import strategy_candidates

    cands = strategy_candidates(p, n_sequences=n_candidates, shifts=())
    current = list(base)
    choice = {d: None for d in drivers}
    history = []
    for rnd in range(1, rounds + 1):
        changed = 0
        for d in drivers:
            obj = (objectives or {}).get(d, "points")
            ev = evaluate_plans(p, current, d, cands, obj, n_sims=n_sims, seed=seed + rnd)
            plan = ev.table["plan"].iloc[0]
            if plan != choice[d]:
                changed += 1
            choice[d] = plan
            current[p.drivers.index(d)] = [cands[[plan_text(c, p.compounds) for c in cands].index(plan)]]
            history.append({"round": rnd, "driver": d, "objective": obj, "plan": plan,
                            "value": ev.table["objective"].iloc[0], "vote_share": ev.table["vote_share"].iloc[0]})
        if changed == 0:
            break
    h = pd.DataFrame(history)
    final = h.groupby("driver").tail(1).set_index("driver").loc[drivers]
    first = h[h["round"] == 1].set_index("driver")["plan"]
    final["changed_after_round_1"] = final["plan"] != first.reindex(final.index)
    final["rounds_run"] = h["round"].max()
    return final.reset_index()


# ---------------------------------------------------------------------------- teammate split
def team_split(p: RaceParams, base: list, d1: str, d2: str, candidates: list[Strategy], n_sims: int = 1000,
               seed: int = 0, points_table=DEFAULT_POINTS) -> pd.DataFrame:
    """Team expected points and P(either wins) for every pair of plans for the two cars."""
    k1, k2 = p.drivers.index(d1), p.drivers.index(d2)
    rows = []
    for c1, c2 in itertools.product(candidates, repeat=2):
        strat = list(base)
        strat[k1], strat[k2] = [c1], [c2]
        r = simulate(p, strat, n_sims=n_sims, seed=seed, exact_cars=frozenset({k1, k2}))
        a, b = r.position[:, k1], r.position[:, k2]
        t1, t2 = plan_text(c1, p.compounds), plan_text(c2, p.compounds)
        rows.append({d1: t1, d2: t2, "split": t1.split("@")[0] != t2.split("@")[0],
                     "team_points": (points_of(a, points_table) + points_of(b, points_table)).mean(),
                     "p_team_win": ((a == 1) | (b == 1)).mean(),
                     "p_team_podium": ((a <= 3) | (b <= 3)).mean()})
    return pd.DataFrame(rows).sort_values("team_points", ascending=False).reset_index(drop=True)
