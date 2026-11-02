"""Walk-forward backtest: for every target race, train on all earlier races only,
predict that race, score the prediction. Nothing from the target race except
its pre-race sessions is visible to the model.

Metrics per race (then averaged):
- spearman:       rank correlation of expected position vs actual finishing order
- winner_logloss: -log P(actual winner)  (lower is better; uniform over 20 = 3.00)
- brier_win / brier_podium: mean squared error of the probabilities
- winner_hit:     the favourite won
- top3_overlap:   how many of the predicted top 3 finished top 3
- mae_pos:        mean |expected position - actual|
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1ml.models.ranker import RACE_KEY

ModelFactory = Callable[[], object]


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    per_race: pd.DataFrame

    def summary(self, by: list[str] | None = None) -> pd.DataFrame:
        by = by or ["model"]
        cols = ["spearman", "winner_logloss", "brier_win", "brier_podium", "winner_hit", "top3_overlap", "mae_pos"]
        out = self.per_race.groupby(by)[cols].mean()
        out.insert(0, "races", self.per_race.groupby(by).size())
        return out


def race_metrics(g: pd.DataFrame) -> dict:
    fo = g["finish_order"].to_numpy(float)
    won = (fo == 1).astype(float)
    pod = (fo <= 3).astype(float)
    pw = g["p_win"].to_numpy(float)
    winner = int(np.argmin(fo))
    top3_pred = set(g.nsmallest(3, "exp_pos")["driver"])
    top3_true = set(g.nsmallest(3, "finish_order")["driver"])
    return {
        "spearman": float(spearmanr(g["exp_pos"], fo).statistic),
        "winner_logloss": float(-np.log(max(pw[winner], 1e-6))),
        "brier_win": float(np.mean((pw - won) ** 2)),
        "brier_podium": float(np.mean((g["p_podium"].to_numpy(float) - pod) ** 2)),
        "winner_hit": float(np.argmax(pw) == winner),
        "top3_overlap": float(len(top3_pred & top3_true)),
        "mae_pos": float(np.mean(np.abs(g["exp_pos"].to_numpy(float) - fo))),
    }


def walk_forward(features: pd.DataFrame, models: dict[str, ModelFactory], eval_seasons: list[int],
                 min_train_races: int = 10, n_sims: int = 10000,
                 progress: Callable[[str], None] | None = None) -> BacktestResult:
    data = features.dropna(subset=["finish_order"]).copy()
    races = data[RACE_KEY + ["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    preds, rows = [], []
    for _, race in races.iterrows():
        if int(race["season"]) not in eval_seasons:
            continue
        train = data[data["date"] < race["date"]]
        if train[RACE_KEY].drop_duplicates().shape[0] < min_train_races:
            continue
        test = data[(data["season"] == race["season"]) & (data["round"] == race["round"])]
        for name, factory in models.items():
            model = factory().fit(train)
            p = model.predict(test, n_sims=n_sims)
            g = pd.concat([test[RACE_KEY + ["driver", "team", "grid_pos", "finish_order", "points"]], p], axis=1)
            g["model"] = name
            g["tau"] = model.tau
            preds.append(g)
            rows.append({"model": name, "season": int(race["season"]), "round": int(race["round"]),
                         "n_train_races": int(train[RACE_KEY].drop_duplicates().shape[0]), **race_metrics(g)})
        if progress:
            progress(f"{int(race['season'])} R{int(race['round']):02d}")
    return BacktestResult(pd.concat(preds, ignore_index=True), pd.DataFrame(rows))
