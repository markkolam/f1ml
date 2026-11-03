"""Tyre model: degradation and compound base pace per track x era x compound.

Inputs are the phase-1 per-race fits (deg per compound with standard errors).

1. Temperature: across races, deg rises with track temperature (Buscombe: the
   Soft went 0.10 s/lap at 52 C to 0.075 at 48 C in Budapest 2026). We fit
   deg = mu[era, compound] + gamma * (track_temp - era mean temp) by weighted
   least squares on all race x compound estimates, giving gamma in
   s/lap-per-lap per degree C.
2. Pooling: temperature-adjusted race estimates are pooled per track, then the
   tracks are shrunk toward their era x compound mean (empirical Bayes).
   Prediction for a new race at temperature T: post_mean + gamma * (T - ref_temp).

Compound labels are used because Pirelli C-numbers are not configured yet
(config/pirelli/); once they are, swap `compound` for `c_number` in the keys.
Known gaps: no cliff/graining term yet (deg is linear in age within a stint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1ml.components.data import RaceData
from f1ml.components.pooling import hierarchical
from f1ml.store import Store

DRY = ("SOFT", "MEDIUM", "HARD")


def race_compound_estimates(store: Store, rd: RaceData) -> pd.DataFrame:
    """Per race x compound: deg, its SE, base pace vs MEDIUM, laps used, mean track temp."""
    fp = store.query("""SELECT season, round, param, value, se FROM features_fit_params
                        WHERE session='R' AND (param LIKE 'deg:%' OR param LIKE 'compound_offset:%')""")
    fp[["kind", "compound"]] = fp["param"].str.split(":", n=1, expand=True)
    fp = fp.merge(rd.events[["season", "round", "track_id", "era_id"]], on=["season", "round"])
    deg = fp[fp["kind"] == "deg"].rename(columns={"value": "deg", "se": "deg_se"})
    base = fp[fp["kind"] == "compound_offset"][["season", "round", "compound", "value", "se"]]
    med = base[base["compound"] == "MEDIUM"].set_index(["season", "round"])
    key = base.set_index(["season", "round"]).index
    base = base.assign(
        base_vs_medium=base["value"] - key.map(med["value"]).to_numpy(),
        # independent-errors approximation for the SE of a difference
        base_se=np.sqrt(base["se"].fillna(0) ** 2 + key.map(med["se"]).to_numpy() ** 2),
    )[["season", "round", "compound", "base_vs_medium", "base_se"]]
    temp = rd.laps[rd.laps["is_clean"]].groupby(["season", "round"])["track_temp"].mean().rename("track_temp")
    used = rd.laps[rd.laps["is_clean_free_air"]].groupby(["season", "round", "compound"]).size().rename("n_laps")
    # Mixed-weather races (any intermediate/wet running) are left out: a drying track
    # makes slick laps faster every lap and swamps tyre wear (e.g. Spa 2025 fitted +1.1 s/lap).
    wet_share = rd.laps.groupby(["season", "round"])["is_wet_tyre"].mean().rename("wet_share")
    out = (deg[["season", "round", "track_id", "era_id", "compound", "deg", "deg_se"]]
           .merge(base, on=["season", "round", "compound"], how="left")
           .merge(temp, on=["season", "round"], how="left")
           .merge(used, on=["season", "round", "compound"], how="left")
           .merge(wet_share, on=["season", "round"], how="left"))
    return out[out["compound"].isin(DRY) & (out["n_laps"].fillna(0) >= 20) & (out["wet_share"].fillna(0) == 0)]


def fit_temperature(est: pd.DataFrame, n_iter: int = 5) -> tuple[float, float, pd.Series]:
    """WLS with track x era x compound fixed effects + gamma * (temp - era mean temp).

    Fixed effects mean gamma is learnt only from the *same* track in different
    years at different temperatures: comparing hot tracks with cold ones would mix
    temperature up with surface and layout. Returns gamma, its SE, ref temps."""
    d = est.dropna(subset=["deg", "deg_se", "track_temp"]).copy()
    ref = d.groupby("era_id")["track_temp"].mean()
    d["dt"] = d["track_temp"] - d["era_id"].map(ref)
    keys = list(zip(d["track_id"], d["era_id"], d["compound"]))
    multi = pd.Series(keys).map(pd.Series(keys).value_counts()) >= 2
    d = d[multi.to_numpy()]
    keys = [k for k, m in zip(keys, multi) if m]
    cells = sorted(set(keys))
    # need repeat visits (same track, different year) with some temperature spread; early in the
    # data window (one season only) there are none, so assume no temperature effect yet
    if len(d) < len(cells) + 5 or d["dt"].std() == 0:
        return 0.0, float("nan"), ref
    X = np.zeros((len(d), len(cells) + 1))
    idx = {c: i for i, c in enumerate(cells)}
    X[np.arange(len(d)), [idx[c] for c in keys]] = 1
    X[:, -1] = d["dt"]
    y = d["deg"].to_numpy()
    tau2 = 0.0
    for _ in range(n_iter):  # extra between-race variance re-estimated each pass
        w = 1 / (d["deg_se"].to_numpy() ** 2 + tau2)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        r = y - X @ beta
        tau2 = max(0.0, float(np.mean(r**2) - np.mean(d["deg_se"] ** 2)))
    cov = np.linalg.pinv((X * w[:, None]).T @ X)
    return float(beta[-1]), float(np.sqrt(max(cov[-1, -1], 0.0))), ref


def fit_tyre(store: Store, rd: RaceData) -> dict[str, pd.DataFrame]:
    est = race_compound_estimates(store, rd)
    gamma, gamma_se, ref = fit_temperature(est)
    est = est.assign(ref_temp=est["era_id"].map(ref))
    est["deg_adj"] = est["deg"] - gamma * (est["track_temp"] - est["ref_temp"]).fillna(0)
    deg = hierarchical(est, "deg_adj", "deg_se", ["track_id", "era_id", "compound"], ["era_id", "compound"])
    deg = deg.rename(columns={"post_mean": "deg_post", "post_sd": "deg_post_sd", "raw_mean": "deg_raw",
                              "parent_mean": "deg_era_mean"})
    base = hierarchical(est[est["compound"] != "MEDIUM"], "base_vs_medium", "base_se",
                        ["track_id", "era_id", "compound"], ["era_id", "compound"])
    base = base.rename(columns={"post_mean": "base_post", "post_sd": "base_post_sd", "raw_mean": "base_raw",
                                "parent_mean": "base_era_mean"})
    table = deg[["track_id", "era_id", "compound", "n", "deg_raw", "deg_post", "deg_post_sd", "deg_era_mean"]].merge(
        base[["track_id", "era_id", "compound", "base_raw", "base_post", "base_post_sd"]],
        on=["track_id", "era_id", "compound"], how="left")
    table["ref_temp"] = table["era_id"].map(ref)
    params = pd.DataFrame([{"param": "temp_gamma", "value": gamma, "se": gamma_se}]
                          + [{"param": f"ref_temp:{e}", "value": v, "se": np.nan} for e, v in ref.items()])
    return {"tyre_track": table, "tyre_params": params, "tyre_race_estimates": est}
