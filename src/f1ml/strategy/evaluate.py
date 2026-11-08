"""Plan evaluation for one driver: objectives, candidate plans, and the vote.

Every candidate plan for the focus driver is run through the *same* simulated
races (common random numbers: same Safety Cars, same lap noise, same rival plan
draws), so differences between plans are the plan, not luck.

Buscombe: "Normally, if it's 299 against one, we go with the data. If it's 150
against 150, the driver is the casting vote." The vote share here is the share
of simulated races in which each plan gave the focus driver the best outcome
under the chosen objective.

"The best strategy depends on who you are": objectives are expected points,
P(win), P(podium), P(points) or mean position, and `auto_objective` picks one
from the driver's situation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1ml.sim.engine import Overrides, RaceParams, Strategy, simulate
from f1ml.sim.strategy import candidate_plans, plan_text

DEFAULT_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
OBJECTIVES = ("points", "win", "podium", "top10", "position")


def points_of(position: np.ndarray, table=DEFAULT_POINTS) -> np.ndarray:
    pts = np.zeros(max(int(position.max()) + 2, len(table) + 1))
    pts[1:len(table) + 1] = table
    return pts[position]


def objective_value(position: np.ndarray, objective: str, table=DEFAULT_POINTS) -> np.ndarray:
    """Per-simulation value of an outcome (higher is better)."""
    if objective == "points":
        return points_of(position, table).astype(float)
    if objective == "win":
        return (position == 1).astype(float)
    if objective == "podium":
        return (position <= 3).astype(float)
    if objective == "top10":
        return (position <= 10).astype(float)
    if objective == "position":
        return -position.astype(float)
    raise ValueError(f"unknown objective {objective!r} (choose from {OBJECTIVES})")


def auto_objective(p: RaceParams, driver: str, baseline: pd.Series) -> tuple[str, str]:
    """Pick an objective from the driver's situation, Buscombe-style. Returns (objective, reason)."""
    k = p.drivers.index(driver)
    pace_rank = int((p.pace_s < p.pace_s[k]).sum()) + 1
    grid = int(p.grid[k])
    if baseline["p_win"] >= 0.25:
        return "points", "a favourite: bank the result (expected points)"
    if pace_rank <= 3 and grid >= 5:
        return "win", f"a top-{pace_rank} car starting P{grid}: the safe strategy doesn't win, attack (P(win))"
    if baseline["p_podium"] >= 0.2:
        return "podium", "in the podium fight (P(podium))"
    if 0.15 <= baseline["p_top10"] <= 0.85:
        return "top10", "on the edge of the points (P(top 10))"
    return "points", "default (expected points)"


def strategy_candidates(p: RaceParams, n_sequences: int = 6, shifts: tuple[int, ...] = (-4, 4),
                        max_stops: int = 3) -> list[Strategy]:
    """Best plan per compound sequence, plus early/late versions of the best few (undercut / extend)."""
    plans = [s for _, s in candidate_plans(p, max_stops=max_stops, top=n_sequences)]
    out = list(plans)
    for s in plans[:3]:
        for d in shifts:
            stops = [(min(max(lap + d, 3), p.laps - 3), c) for lap, c in s.stops]
            if all(b[0] > a[0] for a, b in zip(stops, stops[1:])):
                out.append(Strategy(s.start_compound, stops))
    seen, uniq = set(), []
    for s in out:
        key = (s.start_compound, tuple(s.stops))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


@dataclass
class Evaluation:
    driver: str
    objective: str
    table: pd.DataFrame            # one row per plan, best first
    values: np.ndarray             # (n_plans, N) per-simulation objective values
    positions: np.ndarray          # (n_plans, N)


def evaluate_plans(p: RaceParams, base: list, driver: str, candidates: list[Strategy], objective: str = "points",
                   n_sims: int = 2000, seed: int = 0, overrides: Overrides | None = None,
                   points_table=DEFAULT_POINTS) -> Evaluation:
    """Run each candidate plan for `driver` through the same simulated races."""
    k = p.drivers.index(driver)
    pos, vals, times = [], [], []
    for cand in candidates:
        strat = list(base)
        strat[k] = [cand]
        r = simulate(p, strat, n_sims=n_sims, seed=seed, overrides=overrides, exact_cars=frozenset({k}))
        pos.append(r.position[:, k])
        times.append(np.where(np.isfinite(r.time[:, k]), r.time[:, k], np.inf))
        vals.append(objective_value(r.position[:, k], objective, points_table))
    pos, vals, times = np.array(pos), np.array(vals), np.array(times)
    # vote: per simulated race, which plan did best (objective, then position, then race time)
    key = vals * 1e9 - pos * 1e4 - np.where(np.isfinite(times), times, 1e7) / 1e4
    best = key == key.max(axis=0, keepdims=True)
    votes = (best / best.sum(axis=0, keepdims=True)).sum(axis=1)
    i_best = int(np.argmax(vals.mean(axis=1)))
    diff = vals - vals[i_best]                          # paired differences vs the best plan
    table = pd.DataFrame({
        "plan": [plan_text(c, p.compounds) for c in candidates],
        "stops": [len(c.stops) for c in candidates],
        "objective": vals.mean(axis=1),
        "vs_best": diff.mean(axis=1),
        "vs_best_se": diff.std(axis=1, ddof=1) / np.sqrt(n_sims),
        "votes": votes.round(0).astype(int),
        "vote_share": votes / n_sims,
        "exp_points": points_of(pos, points_table).mean(axis=1),
        "p_win": (pos == 1).mean(axis=1),
        "p_podium": (pos <= 3).mean(axis=1),
        "p_top10": (pos <= 10).mean(axis=1),
        "mean_pos": pos.mean(axis=1),
    })
    order = np.argsort(-table["objective"].to_numpy(), kind="stable")
    return Evaluation(driver, objective, table.iloc[order].reset_index(drop=True), vals[order], pos[order])
