"""Strategies: parse plan strings and find good default plans.

Plan string: compounds separated by '-', stop laps after '@', e.g.
"M-H@24" (Medium start, Hard from lap 25 after boxing on lap 24) or
"S-M-H@14,38". Letters: S, M, H, I (intermediate), W (wet).

`candidate_plans` enumerates 1-, 2- and 3-stop plans (two different dry compounds
required) and scores each with the deterministic stint-time model

    stint time = n * base + deg * n(n+1)/2        (+ green pit loss per stop)

returning the best few. The simulator draws each car's plan per simulation from
those within `within_s` of the best, so the field does not all run one plan.
"""

from __future__ import annotations

import itertools

import numpy as np

from f1ml.sim.engine import RaceParams, Strategy

LETTER = {"S": "SOFT", "M": "MEDIUM", "H": "HARD", "I": "INTERMEDIATE", "W": "WET"}
DRY = ("SOFT", "MEDIUM", "HARD")


def parse_plan(text: str, compounds: list[str]) -> Strategy:
    text = text.strip().upper()
    seq, _, laps = text.partition("@")
    names = [LETTER.get(x, x) for x in seq.split("-") if x]
    stops = [int(x) for x in laps.split(",") if x.strip()] if laps else []
    if len(stops) != len(names) - 1:
        raise ValueError(f"plan {text!r}: {len(names)} stints need {len(names) - 1} stop laps")
    idx = [compounds.index(n) for n in names]
    return Strategy(start_compound=idx[0], stops=list(zip(stops, idx[1:])))


def plan_text(s: Strategy, compounds: list[str]) -> str:
    short = {v: k for k, v in LETTER.items()}
    seq = [compounds[s.start_compound]] + [compounds[c] for _, c in s.stops]
    txt = "-".join(short.get(c, c) for c in seq)
    return txt + ("@" + ",".join(str(l) for l, _ in s.stops) if s.stops else "")


def stint_time(base: float, deg: float, n: np.ndarray | int) -> np.ndarray:
    n = np.asarray(n, float)
    return n * base + deg * n * (n + 1) / 2


def _stop_combos(L: int, n_stops: int, min_stint: int) -> np.ndarray:
    grid = np.arange(min_stint, L - min_stint + 1)
    combos = np.array(list(itertools.combinations(grid, n_stops)), dtype=int).reshape(-1, n_stops)
    lens = np.diff(np.column_stack([np.zeros(len(combos), int), combos, np.full(len(combos), L)]), axis=1)
    return combos[lens.min(axis=1) >= min_stint]


def candidate_plans(p: RaceParams, max_stops: int = 3, min_stint: int = 6, top: int = 8) -> list[tuple[float, Strategy]]:
    """Best plan per compound sequence by deterministic race time (relative), ascending."""
    L = p.laps
    dry = [p.compounds.index(c) for c in DRY if c in p.compounds]
    loss = p.pit_loss_s[0]
    found: list[tuple[float, Strategy]] = []
    for n_stops in range(1, max_stops + 1):
        combos = _stop_combos(L, n_stops, min_stint)
        if len(combos) == 0:
            continue
        lens = np.diff(np.column_stack([np.zeros(len(combos), int), combos, np.full(len(combos), L)]), axis=1)
        for seq in itertools.product(dry, repeat=n_stops + 1):
            if len(set(seq)) < 2:
                continue
            t = sum(stint_time(p.base_s[c], p.deg_s[c], lens[:, k]) for k, c in enumerate(seq)) + n_stops * loss
            i = int(np.argmin(t))
            found.append((float(t[i]), Strategy(seq[0], list(zip(map(int, combos[i]), seq[1:])))))
    found.sort(key=lambda x: x[0])
    return _dedupe(found)[:top]


def _dedupe(plans):
    """Keep one plan per compound sequence (its best stop laps): variety over near-duplicates."""
    seen, out = set(), []
    for t, s in plans:
        key = (s.start_compound, tuple(c for _, c in s.stops))
        if key not in seen:
            seen.add(key)
            out.append((t, s))
    return out


def default_strategies(p: RaceParams, within_s: float = 8.0, max_stops: int = 3) -> list[list[Strategy]]:
    """Per car, the candidate plans within `within_s` of the best (drawn per simulation)."""
    plans = candidate_plans(p, max_stops=max_stops)
    t0 = plans[0][0]
    pool = [s for t, s in plans if t - t0 <= within_s] or [plans[0][1]]
    return [list(pool) for _ in range(p.n_cars)]
