"""The simulator as a pre-race prediction model, with the same interface as the
phase-2 models (fit / predict), so `f1ml backtest` can compare it directly.

Per race: components fitted on earlier races only, RaceParams from this
weekend's qualifying, each car on a plan drawn from the best default strategies,
neutralisations drawn from the hazard model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1ml.sim.engine import simulate
from f1ml.sim.params import PreraceContext, prerace_params
from f1ml.sim.replay import components_as_of
from f1ml.sim.strategy import default_strategies


class SimRaceModel:
    tau = 1.0

    def __init__(self, n_sims: int = 2000, store=None):
        self.n_sims = n_sims
        self._store = store

    @property
    def store(self):
        if self._store is None:
            from f1ml.config import ProjectConfig
            from f1ml.store import Store

            self._store = Store(ProjectConfig().data_dir)
        return self._store

    def fit(self, df: pd.DataFrame) -> "SimRaceModel":
        return self  # everything is fitted per race, as of that race's date

    def predict(self, df: pd.DataFrame, n_sims: int | None = None) -> pd.DataFrame:
        out = []
        for (season, rnd), g in df.groupby(["season", "round"], sort=False):
            comps = components_as_of(self.store, g["date"].iloc[0])
            p = prerace_params(self.store, int(season), int(rnd), PreraceContext(comps, g))
            res = simulate(p, default_strategies(p), n_sims=self.n_sims, seed=int(season) * 100 + int(rnd))
            s = res.summary().set_index("driver")
            r = pd.DataFrame(index=g.index)
            for c in ("p_win", "p_podium", "p_top10", "exp_pos"):
                r[c] = g["driver"].map(s[c]).to_numpy()
            r["score"] = -r["exp_pos"]
            out.append(r)
        return pd.concat(out).loc[df.index]
