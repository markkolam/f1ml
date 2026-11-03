"""Pit-loss model: time lost by stopping, per track x era x condition.

Buscombe: "A pit loss is the time you lose relative to yourself staying out ...
In reality the Pit Loss is not a number, it is a range." We measure each stop as

    loss = (in-lap + out-lap of the stopping car)
           - (median in+out-equivalent laps of cars that did NOT stop on those laps,
              adjusted by the car-pace difference between the stopper and that field)

Condition comes from the in-lap's status. Stops with losses outside 5-60 s
(repairs, penalties served, red flags) are dropped.

Green losses are measured. Neutralised (VSC/SC) losses are NOT measurable this
way from public timing: behind a Safety Car the queue compresses at different
rates for different cars, so lap-time sums say nothing about the time the stop
cost (fully-neutralised SC stops measure ~22 s, the same as green). They are
derived from the green loss with Buscombe's ratios instead (VSC 10.5-12 s and SC
11-13 s against 20.5-24 s green: "the field is doing the same stretch of road at
40 percent below race pace and the car in the pit lane isn't"), and labelled
`method = derived`. A position-based measurement can replace this later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1ml.components.data import RaceData
from f1ml.components.pooling import hierarchical


NEUTRAL_RATIO = {"vsc": 0.52, "sc": 0.52}  # share of the green loss; from Buscombe's published figures
NEUTRAL_RATIO_SD = 0.05


def stop_condition(status: str) -> str:
    s = str(status or "")
    if "5" in s:
        return "red"
    if "4" in s:
        return "sc"
    if "6" in s or "7" in s:
        return "vsc"
    return "green"


def measure_stops(laps: pd.DataFrame, min_ref_cars: int = 5) -> pd.DataFrame:
    """One row per pit stop with its measured loss. `laps` = race laps of one or more races."""
    out = []
    for (season, rnd), L in laps.groupby(["season", "round"]):
        L = L.set_index(["driver", "lap_number"])
        lt = L["lap_time_s"]
        pitting = set(L.index[L["is_in_lap"] | L["is_out_lap"]])
        ins = L[L["is_in_lap"]]
        for (drv, n), row in ins.iterrows():
            if (drv, n + 1) not in L.index or not L.loc[(drv, n + 1), "is_out_lap"]:
                continue
            own = lt.get((drv, n)), lt.get((drv, n + 1))
            if any(pd.isna(v) for v in own) or n == 1:
                continue
            # reference: cars that neither pitted on lap n nor n+1
            ref_rows = []
            for d2 in L.index.get_level_values(0).unique():
                if d2 == drv or (d2, n) in pitting or (d2, n + 1) in pitting:
                    continue
                a, b = lt.get((d2, n)), lt.get((d2, n + 1))
                if pd.notna(a) and pd.notna(b):
                    ref_rows.append((a + b, L.loc[(d2, n), "offset"] if "offset" in L else np.nan))
            if len(ref_rows) < min_ref_cars:
                continue
            ref = np.array(ref_rows, float)
            ref_sum = np.median(ref[:, 0])
            off_ref = np.nanmedian(ref[:, 1]) if np.isfinite(ref[:, 1]).any() else np.nan
            own_off = row.get("offset", np.nan)
            pace_adj = 2 * (own_off - off_ref) if np.isfinite(own_off) and np.isfinite(off_ref) else 0.0
            loss = own[0] + own[1] - ref_sum - pace_adj
            out.append({"season": season, "round": rnd, "driver": drv, "lap": int(n),
                        "condition": stop_condition(row["track_status"]),
                        "out_condition": stop_condition(L.loc[(drv, n + 1), "track_status"]),
                        "loss_s": float(loss), "n_ref": len(ref_rows),
                        "track_id": row["track_id"], "era_id": row["era_id"]})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df[(df["loss_s"] > 5) & (df["loss_s"] < 60) & (df["condition"] != "red")]


def fit_pitloss(rd: RaceData) -> dict[str, pd.DataFrame]:
    all_stops = measure_stops(rd.laps)
    stops = all_stops[(all_stops["condition"] == "green") & (all_stops["out_condition"] == "green")]
    # one estimate per race x condition: median loss, SE of a median ~ 1.253 * sd / sqrt(n)
    g = stops.groupby(["season", "round", "track_id", "era_id", "condition"])["loss_s"]
    race = g.agg(median="median", sd="std", n="size").reset_index()
    race["se"] = 1.253 * race["sd"].fillna(race["sd"].median()) / np.sqrt(race["n"])
    race["se"] = race["se"].clip(lower=0.3)
    pooled = hierarchical(race, "median", "se", ["track_id", "era_id", "condition"], ["era_id", "condition"])
    q = stops.groupby(["track_id", "era_id", "condition"])["loss_s"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q25", "q75"]
    pooled = pooled.merge(q.reset_index(), on=["track_id", "era_id", "condition"], how="left")
    pooled = pooled.rename(columns={"post_mean": "loss_post", "post_sd": "loss_post_sd", "raw_mean": "loss_raw",
                                    "parent_mean": "loss_era_mean", "n": "n_races"})
    n_stops = stops.groupby(["track_id", "era_id", "condition"]).size().rename("n_stops").reset_index()
    pooled = pooled.merge(n_stops, on=["track_id", "era_id", "condition"], how="left").assign(method="measured")
    derived = []
    for cond, r in NEUTRAL_RATIO.items():
        d = pooled.copy()
        d["condition"] = cond
        for c in ("loss_raw", "loss_post", "loss_era_mean", "q25", "q75"):
            d[c] = d[c] * r
        d["loss_post_sd"] = np.sqrt((pooled["loss_post_sd"] * r) ** 2 + (pooled["loss_post"] * NEUTRAL_RATIO_SD) ** 2)
        d["method"] = "derived"
        derived.append(d)
    table = pd.concat([pooled, *derived], ignore_index=True)
    return {"pitloss_track": table, "pitloss_stops": all_stops}
