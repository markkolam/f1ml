"""Replay check ("calibration gate"): rebuild a finished race from its own data
and compare the simulated result with what actually happened.

Per race: rank correlation of simulated median position vs actual order; share of
drivers whose actual position falls inside the simulated 10-90% range; for cars
that finished on the lead lap, the error and 80% coverage of the gap to the winner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1ml.sim.engine import simulate
from f1ml.sim.params import replay_inputs
from f1ml.store import Store


def replay_race(store: Store, season: int, round_: int, comps: dict, n_sims: int = 1000, seed: int = 0,
                strategies=None) -> tuple[dict, pd.DataFrame]:
    ri = replay_inputs(store, season, round_, comps)
    res = simulate(ri.params, strategies or ri.strategies, n_sims=n_sims, seed=seed, overrides=ri.overrides,
                   stop_jitter=0)
    pos = res.position
    T = res.time
    win = np.min(np.where(np.isfinite(T), T, np.inf), axis=1, keepdims=True)
    gap = T - win
    a = ri.actual.copy()
    a["sim_pos_med"] = np.median(pos, 0)
    a["sim_pos_p10"] = np.percentile(pos, 10, 0)
    a["sim_pos_p90"] = np.percentile(pos, 90, 0)
    with np.errstate(invalid="ignore"):
        a["sim_gap_med"] = np.nanmedian(np.where(np.isfinite(gap), gap, np.nan), 0)
        a["sim_gap_p10"] = np.nanpercentile(np.where(np.isfinite(gap), gap, np.nan), 10, 0)
        a["sim_gap_p90"] = np.nanpercentile(np.where(np.isfinite(gap), gap, np.nan), 90, 0)
    lead_lap = a["finished"] & a["gap_s"].notna() & (a["gap_s"] < 0.9 * ri.params.ref_lap_s)
    g = a[lead_lap]
    m = {
        "season": season, "round": round_, "wet": ri.wet, "n_lead_lap": int(lead_lap.sum()),
        "spearman": float(spearmanr(a["sim_pos_med"], a["finish_order"]).statistic),
        "pos_in_10_90": float(((a["finish_order"] >= a["sim_pos_p10"]) & (a["finish_order"] <= a["sim_pos_p90"])).mean()),
        "gap_mae_s": float((g["sim_gap_med"] - g["gap_s"]).abs().mean()) if len(g) else np.nan,
        "gap_in_10_90": float(((g["gap_s"] >= g["sim_gap_p10"]) & (g["gap_s"] <= g["sim_gap_p90"])).mean()) if len(g) else np.nan,
        "winner_correct": float(a.loc[a["sim_pos_med"].idxmin(), "finish_order"] == 1),
    }
    return m, a


_COMP_CACHE: dict = {}


def components_as_of(store: Store, date: pd.Timestamp) -> dict:
    """Components fitted on races strictly before `date` (cached per date)."""
    from f1ml.components import fit_components

    key = (str(store.root), pd.Timestamp(date))
    if key not in _COMP_CACHE:
        if len(_COMP_CACHE) > 8:
            _COMP_CACHE.clear()
        _COMP_CACHE[key] = fit_components(store, as_of=pd.Timestamp(date))
    return _COMP_CACHE[key]


def calibrate(store: Store, seasons: list[int], n_sims: int = 1000, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every dry race of `seasons` with components fitted only on earlier races."""
    ev = store.query("SELECT season, round, event_date FROM raw_events ORDER BY event_date")
    ev = ev[ev["season"].isin(seasons)]
    done = store.query("SELECT DISTINCT season, round FROM features_fit_params WHERE session='R'")
    ev = ev.merge(done, on=["season", "round"])
    rows, drivers = [], []
    for _, e in ev.iterrows():
        try:
            comps = components_as_of(store, e["event_date"])
            m, a = replay_race(store, int(e["season"]), int(e["round"]), comps, n_sims=n_sims)
        except Exception as exc:  # noqa: BLE001 - report and continue
            m, a = {"season": int(e["season"]), "round": int(e["round"]), "error": f"{type(exc).__name__}: {exc}"}, None
        rows.append(m)
        if a is not None:
            drivers.append(a.assign(season=int(e["season"]), round=int(e["round"])))
        if progress:
            progress(m)
    return pd.DataFrame(rows), (pd.concat(drivers, ignore_index=True) if drivers else pd.DataFrame())
