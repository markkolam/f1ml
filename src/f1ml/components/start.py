"""Start model: places gained/lost on lap 1 by grid slot, grid side and track,
and the lap-1 "concertina" (nose-to-tail spread at the end of lap 1).

Buscombe: "The field concertinas to 11.0 seconds nose-to-tail by the end of lap 1,
so you join the last man's gearbox by Sector 2" (Budapest); grid side and the
lap-1 record are measured from public position data; "the launch is power-limited".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1ml.components.data import RaceData
from f1ml.components.pooling import hierarchical


def lap1_table(rd: RaceData) -> pd.DataFrame:
    l1 = rd.laps[rd.laps["lap_number"] == 1][["season", "round", "driver", "race_pos", "time_s", "track_id", "era_id"]]
    g = rd.results[["season", "round", "driver", "grid_position", "status"]]
    t = g.merge(l1, on=["season", "round", "driver"], how="left")
    t = t[t["grid_position"].fillna(0) > 0]                       # pit-lane starters handled separately
    t["completed_lap1"] = t["race_pos"].notna()
    t["gained"] = t["grid_position"] - t["race_pos"]
    t["side"] = np.where(t["grid_position"] % 2 == 1, "pole_side", "off_side")
    return t


def fit_start(rd: RaceData) -> dict[str, pd.DataFrame]:
    t = lap1_table(rd)
    t = t.merge(rd.events[["season", "round", "track_id", "era_id"]], on=["season", "round"], how="left",
                suffixes=("_lap", ""))
    done = t[t["completed_lap1"]]
    by_slot = done.groupby(["era_id", "grid_position"])["gained"].agg(["mean", "std", "size"]).reset_index()
    lap1_out = t.groupby(["era_id", "grid_position"])["completed_lap1"].apply(lambda s: 1 - s.mean()).rename(
        "p_out_lap1").reset_index()
    by_slot = by_slot.merge(lap1_out, on=["era_id", "grid_position"], how="left")

    # grid-side effect per race (pole side minus off side, slot-adjusted), pooled per track
    done = done.merge(by_slot[["era_id", "grid_position", "mean"]], on=["era_id", "grid_position"])
    done["excess"] = done["gained"] - done["mean"]
    rows = []
    for (s, r, tid, eid), g in done.groupby(["season", "round", "track_id", "era_id"]):
        a, b = g.loc[g["side"] == "pole_side", "excess"], g.loc[g["side"] == "off_side", "excess"]
        if len(a) >= 5 and len(b) >= 5:
            rows.append({"season": s, "round": r, "track_id": tid, "era_id": eid, "side_effect": a.mean() - b.mean(),
                         "se": np.sqrt(a.var() / len(a) + b.var() / len(b))})
    side = hierarchical(pd.DataFrame(rows), "side_effect", "se", ["track_id", "era_id"], ["era_id"]).rename(
        columns={"post_mean": "side_post", "post_sd": "side_post_sd", "raw_mean": "side_raw", "n": "n_races"})

    # concertina: time from first to last car across the line at the end of lap 1
    l1 = rd.laps[rd.laps["lap_number"] == 1]
    conc = l1.groupby(["season", "round", "track_id", "era_id"])["time_s"].agg(lambda x: x.max() - x.min())
    conc = conc.rename("spread_s").reset_index()
    conc_t = conc.groupby(["track_id", "era_id"])["spread_s"].agg(["median", "size"]).reset_index()
    return {"start_slot": by_slot, "start_side_track": side, "start_concertina": conc_t, "start_lap1": t}
