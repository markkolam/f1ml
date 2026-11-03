"""Neutralisation hazard: probability that a Safety Car, VSC or red flag starts on a lap.

Buscombe: "People think Safety Cars are random. I think you can look at causality
... Flags come from three things: something external, retirements, and accidents.
Accidents come on lap one, at restarts, and when drivers go from 95 percent to
100 ... My flag model has a bump on Lap 1, a bump at every restart, a slow rise
... and the big one in the pit window where one-stoppers and two-stoppers share
the road." Mexico: "Four of the nine races here since 2016 began with a Safety
Car or VSC on lap one ... After lap 30 it's a 1.3% chance a lap."

Discrete-time hazard, one row per lap at risk (previous lap green):
logit h = b0 + b_lap1*lap1 + b_restart*(1-2 laps after a restart) + b_pit*pit_activity
          + b_late*(last 20% of race) + b_frac*race_frac + b_wet*wet + u_track + v_track*lap1
with pooled (penalised) track intercepts u and track lap-1 effects v.
Event type (SC / VSC / red) shares are estimated per era.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from f1ml.components.data import RaceData

COEFS = ["lap1", "restart", "pit_activity", "late", "race_frac", "wet"]


def neutral_type(status: str) -> str | None:
    s = str(status or "")
    if "5" in s:
        return "red"
    if "4" in s:
        return "sc"
    if "6" in s or "7" in s:
        return "vsc"
    return None


def hazard_rows(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, lap) at risk, with covariates and whether a neutralisation started."""
    out = []
    for (s, r), L in laps.groupby(["season", "round"]):
        total = int(L["lap_number"].max())
        n_cars = L["driver"].nunique()
        lead = L.sort_values("time_s").groupby("lap_number").first()       # first car to complete each lap
        status = lead["track_status"].astype(str)
        typ = status.map(neutral_type)
        pits = L[L["is_in_lap"]].groupby("lap_number").size().reindex(range(1, total + 1), fill_value=0)
        wet = L.groupby("lap_number")["is_wet_tyre"].mean().reindex(range(1, total + 1), fill_value=0)
        last_restart = -99
        prev_type = None
        for n in range(1, total + 1):
            t = typ.get(n)
            if prev_type is not None and t is None:
                last_restart = n
            if prev_type is None:  # at risk: previous lap was green (or this is lap 1)
                recent_pits = pits.loc[max(1, n - 3):max(1, n - 1)].sum() if n > 1 else 0
                out.append({"season": s, "round": r, "track_id": L["track_id"].iloc[0], "era_id": L["era_id"].iloc[0],
                            "lap": n, "total_laps": total, "event": float(t is not None), "type": t,
                            "lap1": float(n == 1), "restart": float(1 <= n - last_restart <= 2),
                            "pit_activity": recent_pits / max(n_cars, 1), "late": float(n > 0.8 * total),
                            "race_frac": n / total, "wet": float(wet.get(n - 1, 0) > 0.05 if n > 1 else wet.get(1, 0) > 0.05)})
            prev_type = t
    return pd.DataFrame(out)


def fit_hazard(rows: pd.DataFrame, track_sd: float = 0.6, lap1_sd: float = 0.8) -> dict[str, pd.DataFrame]:
    tracks = sorted(rows["track_id"].unique())
    ti = rows["track_id"].map({t: i for i, t in enumerate(tracks)}).to_numpy()
    X = rows[COEFS].to_numpy(float)
    y = rows["event"].to_numpy(float)
    l1 = rows["lap1"].to_numpy(float)
    k, nt = X.shape[1], len(tracks)

    def f(th):
        b0, b, u, v = th[0], th[1:1 + k], th[1 + k:1 + k + nt], th[1 + k + nt:]
        z = b0 + X @ b + u[ti] + v[ti] * l1
        p = 1 / (1 + np.exp(-z))
        ll = np.sum(y * z - np.logaddexp(0, z))
        pen = 0.5 * (np.sum(u**2) / track_sd**2 + np.sum(v**2) / lap1_sd**2 + np.sum(b**2) / 25.0)
        r = y - p
        g = np.concatenate([[r.sum()], X.T @ r - b / 25.0, np.bincount(ti, r, nt) - u / track_sd**2,
                            np.bincount(ti, r * l1, nt) - v / lap1_sd**2])
        return -(ll - pen), -g

    th0 = np.zeros(1 + k + 2 * nt)
    th0[0] = -4.0
    res = minimize(f, th0, jac=True, method="L-BFGS-B")
    b0, b = res.x[0], res.x[1:1 + k]
    u, v = res.x[1 + k:1 + k + nt], res.x[1 + k + nt:]
    coefs = pd.DataFrame({"param": ["intercept"] + COEFS, "value": [b0, *b]})

    def sig(z):
        return 1 / (1 + np.exp(-z))

    tab = []
    bi = dict(zip(COEFS, b))
    for i, t in enumerate(tracks):
        sub = rows[rows["track_id"] == t]
        races = sub[["season", "round"]].drop_duplicates()
        base = b0 + u[i] + bi["race_frac"] * 0.5
        tab.append({
            "track_id": t, "n_races": len(races), "u": u[i], "v": v[i],
            "obs_lap1_rate": sub.loc[sub["lap1"] == 1, "event"].mean(),
            "obs_events_per_race": sub["event"].sum() / len(races),
            "p_lap1": sig(b0 + u[i] + v[i] + bi["lap1"] + bi["race_frac"] * (1 / 50)),
            "h_green_mid": sig(base),
            "h_pit_window": sig(base + bi["pit_activity"] * 0.25),
            "h_after_restart": sig(base + bi["restart"]),
            "h_late": sig(b0 + u[i] + bi["race_frac"] * 0.9 + bi["late"]),
        })
    ev = rows[rows["event"] == 1]
    shares = ev.groupby(["era_id", "type"]).size().unstack(fill_value=0)
    shares = (shares.T / shares.sum(axis=1)).T.reset_index()
    return {"flags_coefs": coefs, "flags_track": pd.DataFrame(tab), "flags_type_share": shares}


def durations(laps: pd.DataFrame) -> pd.DataFrame:
    """Length in laps of every neutralisation period (by the leader's lap status), per era and type."""
    out = []
    for (s, r), L in laps.groupby(["season", "round"]):
        lead = L.sort_values("time_s").groupby("lap_number").first()
        typ = lead["track_status"].astype(str).map(neutral_type)
        run_t, run_n = None, 0
        for t in list(typ) + [None]:
            if t == run_t and t is not None:
                run_n += 1
                continue
            if run_t is not None:
                out.append({"season": s, "round": r, "era_id": L["era_id"].iloc[0], "type": run_t, "laps": run_n})
            run_t, run_n = t, 1
    return pd.DataFrame(out, columns=["season", "round", "era_id", "type", "laps"])


def fit_flags(rd: RaceData) -> dict[str, pd.DataFrame]:
    rows = hazard_rows(rd.laps)
    out = fit_hazard(rows)
    out["flags_rows"] = rows
    out["flags_durations"] = durations(rd.laps)
    return out
