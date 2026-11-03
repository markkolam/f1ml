"""Lap-by-lap Monte Carlo race engine, vectorised over simulations.

State per (simulation, car): cumulative race time, tyre compound and age, the
next planned stop, retired flag. Each lap:

1. Neutralisation: continue a running SC/VSC/red period, or draw a new one from
   the per-lap hazard (lap 1, restarts, pit-window activity, late race, track).
2. Pit decisions: planned stops, plus a reactive rule (box under SC/VSC when the
   next planned stop is close). A red flag gives a free tyre change.
3. Lap time (green): ref + car pace + compound base + deg x age - fuel x (lap-1)
   + noise, + dirty-air penalty when following, + pit loss on the in-lap.
   VSC: everyone slows by the same factor (gaps kept). SC: everyone runs the SC
   lap, and the field closes up when the SC comes in.
4. Interaction (green laps): front to back, a car that would reach the car ahead
   attempts a pass with the track's S-curve on the *expected* pace gap; if it
   fails it is held min_gap behind.
5. Random retirements.

Times are seconds. Everything random comes from one seeded Generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GREEN, VSC, SC, RED = 0, 1, 2, 3
STATE_NAMES = {GREEN: "green", VSC: "vsc", SC: "sc", RED: "red"}


@dataclass
class Strategy:
    start_compound: int
    stops: list[tuple[int, int]] = field(default_factory=list)  # (in-lap, compound fitted)


@dataclass
class RaceParams:
    drivers: list[str]
    grid: np.ndarray                    # (C,) grid slot 1..C (pit-lane start = C+1)
    pace_s: np.ndarray                  # (C,) car+driver pace offset vs field (s/lap, negative = faster)
    pace_sd_s: np.ndarray               # (C,) uncertainty of that offset (drawn once per simulation)
    compounds: list[str]
    base_s: np.ndarray                  # (K,) compound pace when new, vs MEDIUM
    deg_s: np.ndarray                   # (K,) s/lap per lap of tyre age
    laps: int
    ref_lap_s: float                    # field-median lap time at lap 1 fuel on a new tyre
    fuel_s_per_lap: float
    lap_noise_s: float = 0.3
    lap1_extra_s: float = 6.0           # standing start vs a flying lap
    concertina_s: float = 12.0          # first-to-last spread at the end of lap 1
    pit_loss_s: dict = field(default_factory=lambda: {GREEN: 22.0, VSC: 11.5, SC: 11.5})
    pit_loss_sd_s: float = 1.5
    pass_a: float = -3.0
    pass_b: float = 2.0
    driver_pass_shift: np.ndarray | None = None   # (C,) logit shift per attacker
    dirty_air_s: float = 0.35
    follow_gap_s: float = 1.5
    attempt_gap_s: float = 1.2
    min_gap_s: float = 0.5
    # neutralisation hazard (logit terms) and behaviour
    hz_intercept: float = -4.0
    hz_lap1: float = 2.0
    hz_restart: float = 0.0
    hz_pit: float = 3.0
    hz_late: float = 0.2
    hz_frac: float = -0.8
    type_share: dict = field(default_factory=lambda: {VSC: 0.4, SC: 0.55, RED: 0.05})
    durations: dict = field(default_factory=lambda: {VSC: [2, 2, 3], SC: [3, 4, 4, 5], RED: [1]})
    vsc_factor: float = 1.35
    sc_factor: float = 1.45
    restart_gap_s: float = 0.8
    reactive_window: int = 10           # box under SC/VSC if the next stop is within this many laps
    # start model
    start_gain_mean: np.ndarray | None = None   # (C,) by grid slot index
    start_gain_sd: np.ndarray | None = None
    p_out_lap1: np.ndarray | None = None
    dnf_per_lap: np.ndarray | None = None       # (C,)

    @property
    def n_cars(self) -> int:
        return len(self.drivers)


@dataclass
class Overrides:
    """Replay mode: impose what actually happened instead of drawing it."""
    neutral: np.ndarray | None = None           # (laps+1,) state per lap (index = lap number)
    retire_lap: np.ndarray | None = None        # (C,) last lap completed (np.inf = finished)
    lap1_time: np.ndarray | None = None         # (C,) cumulative time at end of lap 1
    no_reactive: bool = False


@dataclass
class SimResult:
    drivers: list[str]
    position: np.ndarray        # (N, C) finishing position
    time: np.ndarray            # (N, C) race time (inf if retired)
    laps_done: np.ndarray       # (N, C)
    n_stops: np.ndarray         # (N, C)
    neutral_laps: np.ndarray    # (N, laps+1) state per lap

    def summary(self):
        import pandas as pd

        pos = self.position
        return pd.DataFrame({
            "driver": self.drivers,
            "p_win": (pos == 1).mean(0), "p_podium": (pos <= 3).mean(0), "p_top10": (pos <= 10).mean(0),
            "exp_pos": pos.mean(0), "pos_p10": np.percentile(pos, 10, axis=0), "pos_p90": np.percentile(pos, 90, axis=0),
            "p_dnf": (~np.isfinite(self.time)).mean(0), "mean_stops": self.n_stops.mean(0),
        }).sort_values("exp_pos").reset_index(drop=True)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(p: RaceParams, strategies: list[Strategy] | list[list[Strategy]], n_sims: int = 2000,
             seed: int = 0, overrides: Overrides | None = None, stop_jitter: int = 2) -> SimResult:
    """Run `n_sims` races. `strategies` is one Strategy per car, or per car a list of
    candidate strategies (one is drawn per simulation, uniformly)."""
    rng = np.random.default_rng(seed)
    o = overrides or Overrides()
    N, C, L = n_sims, p.n_cars, p.laps
    K = len(p.compounds)
    ar = np.arange(N)

    # ---------------- strategies -> per-sim arrays
    S_MAX = max(len(s.stops) for st in strategies for s in (st if isinstance(st, list) else [st])) + 1
    stop_lap = np.full((N, C, S_MAX), 10_000, dtype=np.int32)
    stop_cmp = np.zeros((N, C, S_MAX), dtype=np.int32)
    comp = np.zeros((N, C), dtype=np.int32)
    for c, st in enumerate(strategies):
        cands = st if isinstance(st, list) else [st]
        pick = rng.integers(len(cands), size=N)
        for ci, s in enumerate(cands):
            m = pick == ci
            comp[m, c] = s.start_compound
            for k, (lap, cmpd) in enumerate(s.stops):
                jit = rng.integers(-stop_jitter, stop_jitter + 1, size=m.sum()) if stop_jitter else 0
                stop_lap[m, c, k] = np.clip(lap + jit, 2, L - 1)
                stop_cmp[m, c, k] = cmpd
    # keep each car's stops ordered after jitter
    order = np.argsort(stop_lap, axis=2)
    stop_lap = np.take_along_axis(stop_lap, order, 2)
    stop_cmp = np.take_along_axis(stop_cmp, order, 2)
    nxt = np.zeros((N, C), dtype=np.int32)

    pace = p.pace_s[None, :] + rng.normal(size=(N, C)) * p.pace_sd_s[None, :]
    shift = p.driver_pass_shift if p.driver_pass_shift is not None else np.zeros(C)
    dnf = p.dnf_per_lap if p.dnf_per_lap is not None else np.zeros(C)
    retired = np.zeros((N, C), bool)
    laps_done = np.zeros((N, C), np.int32)
    n_stops = np.zeros((N, C), np.int32)
    age = np.zeros((N, C))
    pit_hist = np.zeros((N, L + 2))
    neutral = np.zeros((N, L + 1), np.int8)
    state = np.zeros(N, np.int8)
    remaining = np.zeros(N, np.int32)
    since_restart = np.full(N, 99)

    def draw_type(m):
        types = np.array(list(p.type_share.keys()))
        probs = np.array(list(p.type_share.values()), float)
        t = rng.choice(types, size=m.sum(), p=probs / probs.sum())
        dur = np.array([rng.choice(p.durations[tt]) for tt in t])
        return t, dur

    # ---------------- lap 1 (start)
    grid = p.grid.astype(float)
    if o.lap1_time is not None:
        T = np.broadcast_to(o.lap1_time, (N, C)).astype(float).copy()
    else:
        slot = np.clip(grid.astype(int) - 1, 0, C - 1)
        gm = p.start_gain_mean[slot] if p.start_gain_mean is not None else np.zeros(C)
        gs = p.start_gain_sd[slot] if p.start_gain_sd is not None else np.full(C, 1.5)
        score = grid[None, :] - (gm[None, :] + rng.normal(size=(N, C)) * gs[None, :]) + rng.normal(size=(N, C)) * 0.01
        rank = np.argsort(np.argsort(score, axis=1), axis=1)                 # 0 = leader after lap 1
        lap1 = p.ref_lap_s + p.lap1_extra_s + p.deg_s[comp] + p.base_s[comp]
        T = lap1 + p.concertina_s * rank / max(C - 1, 1) + rng.normal(size=(N, C)) * p.lap_noise_s
        if p.p_out_lap1 is not None:
            out1 = rng.random((N, C)) < p.p_out_lap1[slot][None, :]
            retired |= out1
    retired |= ~np.isfinite(T)           # replay: cars that never completed lap 1
    age[:] = 1
    laps_done[~retired] = 1
    T[retired] = np.inf

    # lap-1 neutralisation
    if o.neutral is not None:
        neutral[:, 1] = o.neutral[1]
    else:
        z = p.hz_intercept + p.hz_lap1 + p.hz_frac / L
        ev = rng.random(N) < _sig(z)
        if ev.any():
            t, d = draw_type(ev)
            state[ev], remaining[ev] = t, d
            neutral[ev, 1] = t
    if o.neutral is not None:
        state[:] = o.neutral[1]

    # ---------------- laps 2..L
    for lap in range(2, L + 1):
        # 1. neutralisation state for this lap
        if o.neutral is not None:
            prev = state.copy()
            state[:] = o.neutral[lap]
        else:
            prev = state.copy()
            remaining[state != GREEN] -= 1
            ending = (state != GREEN) & (remaining <= 0)
            state[ending] = GREEN
            green = state == GREEN
            act = pit_hist[:, max(0, lap - 3):lap].sum(1) / C
            # since_restart is 0 on the first green lap; the hazard data's "restart" covers
            # the 1st and 2nd laps *after* it, i.e. since_restart 0 or 1 at draw time
            z = (p.hz_intercept + p.hz_restart * (since_restart <= 1)
                 + p.hz_pit * act + p.hz_late * (lap > 0.8 * L) + p.hz_frac * lap / L)
            ev = green & (prev == GREEN) & (rng.random(N) < _sig(z))
            if ev.any():
                t, d = draw_type(ev)
                state[ev], remaining[ev] = t, d
        neutral[:, lap] = state
        restart_now = (prev != GREEN) & (state == GREEN)
        since_restart = np.where(restart_now, 0, since_restart + 1)

        alive = ~retired
        st = state[:, None]

        # 2. pit decisions (in-lap = this lap)
        planned = stop_lap[ar[:, None], np.arange(C)[None, :], np.minimum(nxt, S_MAX - 1)]
        has_next = nxt < S_MAX
        pit = alive & has_next & (planned == lap)
        if not o.no_reactive:
            cheap = (st == VSC) | (st == SC)
            react = alive & has_next & cheap & (planned > lap) & (planned - lap <= p.reactive_window) & (lap >= 5)
            pit |= react
        red_now = (state == RED)[:, None] & alive
        free_change = red_now & has_next & (planned < 10_000)
        pit_any = pit | free_change

        # 3. lap times
        exp_t = (p.ref_lap_s + pace + p.base_s[comp] + p.deg_s[comp] * (age + 1) - p.fuel_s_per_lap * (lap - 1))
        t = exp_t + rng.normal(size=(N, C)) * p.lap_noise_s
        order = np.argsort(T, axis=1)
        Ts = np.take_along_axis(T, order, 1)
        gap_ahead = np.empty_like(T)
        with np.errstate(invalid="ignore"):     # inf - inf for retired cars -> nan, never "close"
            gaps_sorted = np.diff(Ts, axis=1, prepend=-np.inf)
        np.put_along_axis(gap_ahead, order, gaps_sorted, 1)
        t = np.where((state == GREEN)[:, None] & (gap_ahead < p.follow_gap_s), t + p.dirty_air_s, t)
        t = np.where(st == VSC, np.maximum(t, p.ref_lap_s * p.vsc_factor), t)
        t = np.where((st == SC) | (st == RED), p.ref_lap_s * p.sc_factor, t)
        loss = np.select([st == GREEN, st == VSC], [p.pit_loss_s[GREEN], p.pit_loss_s[VSC]], p.pit_loss_s[SC])
        loss = loss + rng.normal(size=(N, C)) * p.pit_loss_sd_s
        t = t + np.where(pit & ~free_change, np.broadcast_to(loss, (N, C)), 0.0)

        Tc = T + t
        # 4. interaction on green laps
        green_rows = state == GREEN
        if green_rows.any():
            Tadj = Tc.copy()
            exp_k = exp_t
            for k in range(1, C):
                i = order[:, k]
                j = order[:, k - 1]
                ti, tj = Tc[ar, i], Tadj[ar, j]
                pi, pj = pit[ar, i], pit[ar, j]
                # one car pitting: no interaction (it drops back on its own). Both pitting on
                # the same lap: they queue through the pit lane, so order is held.
                reach = green_rows & np.isfinite(ti) & np.isfinite(tj) & (ti < tj + p.min_gap_s) & ~(pi ^ pj)
                if not reach.any():
                    continue
                delta = exp_k[ar, j] - exp_k[ar, i]
                pp = np.where(pi & pj, 0.0, _sig(p.pass_a + p.pass_b * delta + shift[i]))
                passed = rng.random(N) < pp
                new_i = np.where(passed, np.minimum(ti, tj - 0.2), np.maximum(ti, tj + p.min_gap_s))
                Tadj[ar, i] = np.where(reach, new_i, ti)
            Tc = np.where(green_rows[:, None], Tadj, Tc)
        # SC in: close the field up behind the leader on the last SC lap
        sc_ending = (state == SC) & (remaining == 1) if o.neutral is None else \
            (state == SC) & (np.asarray(o.neutral)[min(lap + 1, L)] != SC)
        if sc_ending.any():
            rows = np.where(sc_ending)[0]
            ordr = np.argsort(Tc[rows], axis=1)
            lead = np.take_along_axis(Tc[rows], ordr[:, :1], 1)
            pos = np.argsort(ordr, axis=1)
            closed = lead + pos * p.restart_gap_s
            Tc[rows] = np.where(np.isfinite(Tc[rows]), closed, np.inf)

        # tyres and stops
        new_cmp = stop_cmp[ar[:, None], np.arange(C)[None, :], np.minimum(nxt, S_MAX - 1)]
        comp = np.where(pit_any, new_cmp, comp)
        age = np.where(pit_any, 0, age + 1)   # out-lap next lap is tyre life 1, as in the timing data
        n_stops += pit_any & ~free_change
        nxt = nxt + pit_any
        pit_hist[:, lap] = pit_any.sum(1)

        # 5. retirements
        if o.retire_lap is not None:
            out = alive & (lap > np.asarray(o.retire_lap)[None, :])
        else:
            out = alive & (rng.random((N, C)) < dnf[None, :])
        retired |= out
        Tc[retired] = np.inf
        laps_done = np.where(retired, laps_done, lap)
        T = Tc

    # ---------------- classification: finishers by time, retirements by laps completed
    key = np.where(np.isfinite(T), T, 1e12 - laps_done * 1e6)
    position = np.argsort(np.argsort(key, axis=1), axis=1) + 1
    return SimResult(list(p.drivers), position, T, laps_done, n_stops, neutral)
