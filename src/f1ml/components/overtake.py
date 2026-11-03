"""Overtaking model ("pass ladder") and dirty-air penalty.

Buscombe's measurement (Budapest 2026): "225 times a car crossed the line
within 1.2 seconds of the car ahead under green, and 28 of them were a pass by
the next lap ... The 50 percent point is 1.09 seconds a lap." And: "first order
effect captures 80% of it, it's just pace delta"; at a new track "you look at the
layout, decide which track it resembles, and set the number"; and the field
curve is not every driver's curve ("his conversion record prices him roughly two
tenths under that").

Attempts: attacker B directly behind A at the line on lap n, gap <= 1.2 s, both
cars on green laps n and n+1, neither pitting. Pass: B ahead of A at the line on n+1.
Pace delta: A's expected lap time minus B's on lap n+1 from the phase-1 fit
(car offset + compound base + deg x tyre age), i.e. the tyre offset included.

Model (MAP logistic): logit P(pass) = (a_era + a_track) + (b_era + b_track) * delta + c_driver,
with Gaussian penalties pulling track and driver terms to zero.
Per track: 50% and 90% pass thresholds in s/lap. Per driver: the shift in s/lap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from f1ml.components.data import RaceData
from f1ml.components.pooling import hierarchical

GAP_MAX_S = 1.2


def attempts(laps: pd.DataFrame, gap_max: float = GAP_MAX_S) -> pd.DataFrame:
    L = laps[["season", "round", "track_id", "era_id", "driver", "lap_number", "time_s", "race_pos",
              "track_status", "is_in_lap", "is_out_lap", "expected_s", "compound", "tyre_life"]].copy()
    L["ok"] = (L["track_status"].astype(str) == "1") & ~L["is_in_lap"] & ~L["is_out_lap"]
    nxt = L[["season", "round", "driver", "lap_number", "race_pos", "ok", "expected_s", "tyre_life", "compound"]].copy()
    nxt["lap_number"] -= 1
    L = L.merge(nxt, on=["season", "round", "driver", "lap_number"], suffixes=("", "_next"))
    L = L[L["lap_number"] >= 2]
    L = L.sort_values(["season", "round", "lap_number", "race_pos"])
    ahead = L.groupby(["season", "round", "lap_number"]).shift(1)
    a = L.assign(**{f"a_{c}": ahead[c] for c in ["driver", "time_s", "race_pos", "ok", "ok_next", "race_pos_next",
                                                  "expected_s_next", "tyre_life_next", "compound_next"]})
    a["gap"] = a["time_s"] - a["a_time_s"]
    m = (a["gap"] <= gap_max) & a["ok"] & a["ok_next"] & a["a_ok"].eq(True) & a["a_ok_next"].eq(True) \
        & a["expected_s_next"].notna() & a["a_expected_s_next"].notna()
    a = a[m]
    return pd.DataFrame({
        "season": a["season"], "round": a["round"], "track_id": a["track_id"], "era_id": a["era_id"],
        "lap": a["lap_number"], "attacker": a["driver"], "defender": a["a_driver"], "gap": a["gap"],
        "delta": a["a_expected_s_next"] - a["expected_s_next"],
        "age_offset": a["a_tyre_life_next"] - a["tyre_life_next"],
        "same_compound": a["compound_next"] == a["a_compound_next"],
        "passed": (a["race_pos_next"] < a["a_race_pos_next"]).astype(float),
    }).reset_index(drop=True)


def _codes(s: pd.Series) -> tuple[np.ndarray, list]:
    cats = sorted(s.unique())
    return s.map({c: i for i, c in enumerate(cats)}).to_numpy(), cats


def fit_pass_model(att: pd.DataFrame, track_sd=(0.7, 0.4), driver_sd: float = 0.4) -> dict[str, pd.DataFrame]:
    """MAP logistic with era curves, pooled track deviations and pooled driver offsets."""
    era_i, eras = _codes(att["era_id"])
    att = att.assign(tk=att["track_id"] + "|" + att["era_id"])
    trk_i, trks = _codes(att["tk"])
    drv_i, drvs = _codes(att["attacker"])
    x, y = att["delta"].to_numpy(float), att["passed"].to_numpy(float)
    ne, nt, nd = len(eras), len(trks), len(drvs)

    def unpack(th):
        return th[:ne], th[ne:2 * ne], th[2 * ne:2 * ne + nt], th[2 * ne + nt:2 * ne + 2 * nt], th[2 * ne + 2 * nt:]

    def f(th):
        ae, be, at, bt, cd = unpack(th)
        z = ae[era_i] + at[trk_i] + (be[era_i] + bt[trk_i]) * x + cd[drv_i]
        p = 1 / (1 + np.exp(-z))
        ll = np.sum(y * z - np.logaddexp(0, z))
        pen = 0.5 * (np.sum(at**2) / track_sd[0]**2 + np.sum(bt**2) / track_sd[1]**2 + np.sum(cd**2) / driver_sd**2)
        r = y - p
        g = np.zeros_like(th)
        g[:ne] = np.bincount(era_i, r, ne)
        g[ne:2 * ne] = np.bincount(era_i, r * x, ne)
        g[2 * ne:2 * ne + nt] = np.bincount(trk_i, r, nt) - at / track_sd[0]**2
        g[2 * ne + nt:2 * ne + 2 * nt] = np.bincount(trk_i, r * x, nt) - bt / track_sd[1]**2
        g[2 * ne + 2 * nt:] = np.bincount(drv_i, r, nd) - cd / driver_sd**2
        return -(ll - pen), -g

    th0 = np.zeros(2 * ne + 2 * nt + nd)
    th0[:ne] = -2.5
    th0[ne:2 * ne] = 2.0
    res = minimize(f, th0, jac=True, method="L-BFGS-B")
    ae, be, at, bt, cd = unpack(res.x)
    era_tab = pd.DataFrame({"era_id": eras, "a": ae, "b": be})
    rows = []
    for k, t in enumerate(trks):
        tid, eid = t.split("|")
        e = eras.index(eid)
        a, b = ae[e] + at[k], be[e] + bt[k]
        sub = att[att["tk"] == t]
        rows.append({"track_id": tid, "era_id": eid, "a": a, "b": b, "n_attempts": len(sub),
                     "pass_rate": sub["passed"].mean(),
                     "threshold_50": -a / b if b > 0 else np.inf,
                     "threshold_90": (np.log(9) - a) / b if b > 0 else np.inf})
    trk_tab = pd.DataFrame(rows)
    b_ref = float(np.average(be, weights=np.bincount(era_i, minlength=ne)))
    drv_tab = pd.DataFrame({"driver": drvs, "logit_shift": cd,
                            "pace_equiv_s": cd / b_ref,  # +0.2 = passes as if 0.2 s/lap faster than he is
                            "n_attempts": np.bincount(drv_i, minlength=nd)})
    return {"overtake_era": era_tab, "overtake_track": trk_tab, "overtake_driver": drv_tab}


def dirty_air(laps: pd.DataFrame) -> pd.DataFrame:
    """Lap-time cost of following (s/lap): clean dirty-air laps vs clean free-air laps, per race,
    pooled per track and era. Residuals are against the phase-1 expected lap time."""
    L = laps[laps["is_clean"] & laps["expected_s"].notna() & laps["reference_offset"].notna()].copy()
    L["resid"] = L["lap_time_s"] - L["reference_offset"] - L["expected_s"]
    rows = []
    for (s, r, tid, eid), g in L.groupby(["season", "round", "track_id", "era_id"]):
        d, f = g.loc[g["is_dirty_air"], "resid"], g.loc[~g["is_dirty_air"], "resid"]
        if len(d) < 30 or len(f) < 30:
            continue
        rows.append({"season": s, "round": r, "track_id": tid, "era_id": eid, "penalty": d.mean() - f.mean(),
                     "se": np.sqrt(d.var() / len(d) + f.var() / len(f)), "n_dirty": len(d)})
    race = pd.DataFrame(rows)
    pooled = hierarchical(race, "penalty", "se", ["track_id", "era_id"], ["era_id"])
    return pooled.rename(columns={"post_mean": "penalty_post", "post_sd": "penalty_post_sd",
                                  "raw_mean": "penalty_raw", "parent_mean": "penalty_era_mean", "n": "n_races"})


def fit_overtake(rd: RaceData) -> dict[str, pd.DataFrame]:
    att = attempts(rd.laps)
    out = fit_pass_model(att)
    out["overtake_attempts"] = att
    out["dirty_air_track"] = dirty_air(rd.laps)
    return out
