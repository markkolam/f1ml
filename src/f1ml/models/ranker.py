"""Race-outcome models that produce finishing-order probabilities.

Every model outputs a per-driver *score* (higher = better). Scores become
probabilities through a Plackett-Luce model: the race is a sequence of draws
where driver i is picked next with probability proportional to exp(tau * s_i).
`tau` is fitted by maximum likelihood on out-of-fold scores, so the ranker's
confidence is calibrated on races it did not see. P(win) is a softmax;
P(podium), P(top 10) and expected position come from Gumbel-max sampling.

Models:
- `LGBMRaceRanker`: LightGBM LambdaRank on the pre-race feature table, with
  monotone constraints where the direction is known (worse grid = worse).
- `LogPositionPL`: the baselines; score = -beta * log(position) for one column
  (grid or qualifying position), beta fitted by maximum likelihood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

RACE_KEY = ["season", "round"]

# +1: higher value -> better result; -1: higher value -> worse result.
MONOTONE = {
    "grid_pos": -1, "quali_pos": -1, "quali_gap_pct": -1, "longrun_gap_pct": -1,
    "race_pace_ewm": -1, "team_race_pace_ewm": -1, "quali_gap_ewm": -1, "finish_ewm": -1,
    "points_ewm": 1, "sprint_pos": -1, "sprint_pace_pct": -1,
}


# ============================================================================ Plackett-Luce
def _orders(df: pd.DataFrame, scores: np.ndarray) -> list[np.ndarray]:
    """Scores of each race arranged in actual finishing order."""
    tmp = df[RACE_KEY + ["finish_order"]].assign(_s=scores)
    return [g.sort_values("finish_order")["_s"].to_numpy() for _, g in tmp.groupby(RACE_KEY)]


def pl_loglik(ordered_scores: list[np.ndarray], tau: float, top_k: int | None = None) -> float:
    ll = 0.0
    for s in ordered_scores:
        z = tau * s
        k = len(z) if top_k is None else min(top_k, len(z))
        for i in range(k):
            ll += z[i] - logsumexp(z[i:])
    return ll


def fit_tau(ordered_scores: list[np.ndarray], top_k: int | None = None) -> float:
    res = minimize_scalar(lambda t: -pl_loglik(ordered_scores, t, top_k), bounds=(1e-3, 50.0), method="bounded")
    return float(res.x)


def pl_probabilities(scores: np.ndarray, tau: float, n_sims: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Per-driver P(win), P(podium), P(top10) and expected position for one race."""
    z = tau * np.asarray(scores, float)
    n = len(z)
    rng = np.random.default_rng(seed)
    g = z[None, :] + rng.gumbel(size=(n_sims, n))
    ranks = np.empty_like(g, dtype=np.int32)
    order = np.argsort(-g, axis=1)
    ranks[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1)[None, :]
    p_win = np.exp(z - logsumexp(z))
    return pd.DataFrame({
        "p_win": p_win,
        "p_podium": (ranks <= 3).mean(axis=0),
        "p_top10": (ranks <= 10).mean(axis=0),
        "exp_pos": ranks.mean(axis=0),
    })


class _PLModel:
    tau: float = 1.0

    def scores(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, df: pd.DataFrame, n_sims: int = 20000) -> pd.DataFrame:
        s = self.scores(df)
        out = []
        for i, (_, g) in enumerate(df.assign(_s=s).groupby(RACE_KEY, sort=False)):
            p = pl_probabilities(g["_s"].to_numpy(), self.tau, n_sims=n_sims, seed=i)
            p.index = g.index
            out.append(p.assign(score=g["_s"]))
        return pd.concat(out).loc[df.index]


# ============================================================================ baselines
class LogPositionPL(_PLModel):
    """score = -log(position): the 'everyone finishes where they start' baseline."""

    def __init__(self, column: str = "grid_pos", tau_top_k: int | None = None):
        self.column = column
        self.tau_top_k = tau_top_k

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        return -np.log(df[self.column].clip(lower=1).fillna(df[self.column].max()).to_numpy(float))

    def fit(self, df: pd.DataFrame) -> "LogPositionPL":
        df = df.dropna(subset=["finish_order"])
        self.tau = fit_tau(_orders(df, self.scores(df)), self.tau_top_k)
        return self


# ============================================================================ LightGBM ranker
class LGBMRaceRanker(_PLModel):
    def __init__(self, features: list[str], n_estimators: int = 300, learning_rate: float = 0.03,
                 num_leaves: int = 15, min_child_samples: int = 20, cv_folds: int = 4, seed: int = 0,
                 gain_base: float | None = None, tau_top_k: int | None = None,
                 blend_with: str | None = None):
        """gain_base: LambdaRank gain for relevance l is gain_base**l - 1 (front-weighted);
        None = linear gain (every place swap counts the same).
        tau_top_k: calibrate tau on the likelihood of the top-k order only (the front
        of the race), ignoring the noisy back of the field; None = full order.
        blend_with: a position column (e.g. "grid_pos"). Final strength =
        a * ranker_score + b * (-log position), with (a, b) fitted by Plackett-Luce
        likelihood on out-of-fold scores. The ranker is best at the overall order,
        the grid at the front; the blend lets the data decide the mix."""
        self.features = list(features)
        self.blend_with = blend_with
        self.blend_coef: tuple[float, float] | None = None
        self.gain_base = gain_base
        self.tau_top_k = tau_top_k
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1,
                           n_jobs=1)
        self.cv_folds = cv_folds
        self.seed = seed
        self.model = None

    def _make(self):
        from lightgbm import LGBMRanker

        mono = [MONOTONE.get(f, 0) for f in self.features]
        gain = [self.gain_base ** l - 1 for l in range(64)] if self.gain_base else list(range(64))
        return LGBMRanker(objective="lambdarank", label_gain=gain, monotone_constraints=mono,
                          monotone_constraints_method="advanced", **self.params)

    @staticmethod
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(RACE_KEY + ["finish_order"], kind="mergesort")

    def _fit_one(self, df: pd.DataFrame):
        df = self._prep(df)
        n_max = int(df.groupby(RACE_KEY).size().max())
        label = (n_max + 1 - df["finish_order"]).clip(lower=0).astype(int)
        groups = df.groupby(RACE_KEY, sort=False).size().to_numpy()
        m = self._make()
        m.fit(df[self.features], label, group=groups)
        return m

    def fit(self, df: pd.DataFrame) -> "LGBMRaceRanker":
        df = df.dropna(subset=["finish_order"])
        races = df[RACE_KEY].drop_duplicates().reset_index(drop=True)
        # out-of-fold scores (grouped by race) to calibrate tau honestly
        oof = pd.Series(np.nan, index=df.index)
        if len(races) >= 2 * self.cv_folds:
            fold = pd.Series(np.arange(len(races)) % self.cv_folds, index=pd.MultiIndex.from_frame(races))
            f_of_row = pd.MultiIndex.from_frame(df[RACE_KEY]).map(fold)
            for k in range(self.cv_folds):
                tr, te = df[f_of_row != k], df[f_of_row == k]
                oof.loc[te.index] = self._fit_one(tr).predict(te[self.features])
        self.model = self._fit_one(df)
        base = oof.to_numpy() if oof.notna().all() else self.model.predict(df[self.features])
        if self.blend_with:
            pos = self._pos_score(df)
            k = self.tau_top_k

            def nll(ab):
                return -pl_loglik(_orders(df, ab[0] * base + ab[1] * pos), 1.0, k)

            res = minimize(nll, x0=np.array([1.0, 1.0]), method="Nelder-Mead",
                           options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
            self.blend_coef = (float(res.x[0]), float(res.x[1]))
            self.tau = 1.0
        else:
            self.tau = fit_tau(_orders(df, base), self.tau_top_k)
        return self

    def _pos_score(self, df: pd.DataFrame) -> np.ndarray:
        col = df[self.blend_with]
        return -np.log(col.clip(lower=1).fillna(col.max()).to_numpy(float))

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        s = self.model.predict(df[self.features])
        if self.blend_coef is not None:
            a, b = self.blend_coef
            s = a * s + b * self._pos_score(df)
        return s

    def importance(self) -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)
