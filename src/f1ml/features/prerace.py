"""Pre-race feature table: one row per (race, driver), using only information
available before lights out.

Leakage rule: for race E, *history* features use races strictly before E's date;
*weekend* features use E's practice, qualifying, sprint-qualifying and sprint
sessions (all run before the Grand Prix).

Feature groups (Buscombe's lens in brackets):
- grid / qualifying: grid slot, pit-lane start, quali position, gap to pole,
  gap to teammate.
- Friday long runs: best clean long-run pace gap to the session best on the same
  compound ("half the FP2 long run ... and half qualifying").
- sprint: sprint finish and sprint race-pace offset (sprint weekends).
- form: exponentially weighted race pace (car+fuel-corrected offsets from the
  phase-1 fits), quali gap, finishing position, points; team reliability.
- track history: grid->finish rank correlation ("Saturday or Sunday track?"),
  mean places changed (overtaking difficulty proxy), share of past races with a
  Safety Car/VSC.
- era: races since the current regulation era started (rule resets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1ml.store import Store

FINISHED_STATUSES = {"Finished", "Lapped"}
DNS_STATUSES = {"Did not start", "Withdrew", "DNS"}

FEATURES = [
    # weekend
    "grid_pos", "pit_lane_start", "quali_pos", "quali_gap_pct", "teammate_quali_gap_pct",
    "longrun_gap_pct", "n_long_runs", "is_sprint_weekend", "sprint_pos", "sprint_pace_pct",
    # form
    "race_pace_ewm", "team_race_pace_ewm", "quali_gap_ewm", "finish_ewm", "points_ewm",
    "team_dnf_rate", "n_prev_races",
    # track history
    "track_grid_finish_rho", "track_mean_places_changed", "track_sc_share", "track_n_prev",
    # era
    "races_into_era",
]
TARGETS = ["finish_order", "points", "won", "podium", "top10", "dnf"]


# ---------------------------------------------------------------------------- helpers
def finish_order(res: pd.DataFrame) -> pd.Series:
    """Full classification 1..N: classified finishers by position, then retirements
    by laps completed (more laps = higher), then non-starters."""
    status = res["status"].fillna("").astype(str)
    dns = status.isin(DNS_STATUSES) | res["laps"].fillna(0).eq(0) & res["position"].isna()
    key = pd.DataFrame({
        "grp": np.where(res["position"].notna(), 0, np.where(dns, 2, 1)),
        "pos": res["position"].fillna(999),
        "laps": -res["laps"].fillna(0),
        "grid": res["grid_position"].fillna(99) if "grid_position" in res else 99,
    }, index=res.index)
    order = key.sort_values(["grp", "pos", "laps", "grid"], kind="mergesort").index
    out = pd.Series(np.arange(1, len(res) + 1), index=order)
    return out.reindex(res.index)


def best_quali_time(q: pd.DataFrame) -> pd.Series:
    return q[["q1_s", "q2_s", "q3_s"]].min(axis=1, skipna=True)


def ewm_by_key(hist: pd.DataFrame, key: str, value: str, halflife: float = 3.0) -> pd.Series:
    """Per-key exponentially weighted mean of `value` over that key's own past races (newest weight 1)."""
    h = hist.dropna(subset=[value]).sort_values("date")
    if h.empty:
        return pd.Series(dtype=float)
    k = h.groupby(key).cumcount(ascending=False)          # 0 = most recent
    w = 0.5 ** (k / halflife)
    num = (h[value] * w).groupby(h[key]).sum()
    den = w.groupby(h[key]).sum()
    return num / den


# ---------------------------------------------------------------------------- loading
def _load(store: Store) -> dict[str, pd.DataFrame]:
    q = store.query
    events = q("SELECT season, round, event_name, event_date, track_id, era_id, event_format FROM raw_events")
    results = q("""SELECT season, round, session, driver, driver_number, team, team_id, position, grid_position,
                          status, points, laps, q1_s, q2_s, q3_s FROM raw_results""")
    long_runs = q("""SELECT season, round, session, driver, compound, n_laps, mean_lap_s
                     FROM features_long_runs WHERE session IN ('FP1','FP2','FP3') AND n_laps >= 5""") \
        if store.has_table("features", "long_runs") else pd.DataFrame()
    fits = q("""SELECT season, round, session, param, value FROM features_fit_params
                WHERE param LIKE 'offset:%' OR param = 'reference_offset'""")
    neutral = q("""SELECT season, round, max(is_neutralised::INT) AS had_sc
                   FROM clean_laps WHERE session='R' GROUP BY 1, 2""")
    events["event_date"] = pd.to_datetime(events["event_date"])
    return {"events": events, "results": results, "long_runs": long_runs, "fits": fits, "neutral": neutral}


def _pace_pct(fits: pd.DataFrame, session: str) -> pd.DataFrame:
    """Car+fuel-corrected pace offset per driver as % of the field's reference lap."""
    f = fits[fits["session"] == session]
    ref = f[f["param"] == "reference_offset"].set_index(["season", "round"])["value"]
    off = f[f["param"].str.startswith("offset:")].copy()
    off["driver"] = off["param"].str.slice(len("offset:"))
    off["ref"] = off.set_index(["season", "round"]).index.map(ref)
    off["pace_pct"] = off["value"] / off["ref"] * 100
    return off[["season", "round", "driver", "pace_pct"]]


# ---------------------------------------------------------------------------- per-race history table
def race_history(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (past race, driver): outcomes and pace measures used by form features."""
    ev = d["events"]
    res = d["results"]
    R = res[res["session"] == "R"].copy()
    if R.empty:
        return pd.DataFrame()
    R["finish_order"] = pd.concat([finish_order(g) for _, g in R.groupby(["season", "round"])])
    st = R["status"].fillna("").astype(str)
    R["dns"] = st.isin(DNS_STATUSES)
    R["dnf"] = R["position"].isna() & ~R["dns"]
    R["team_key"] = R["team_id"].fillna(R["team"])
    Q = res[res["session"] == "Q"].copy()
    Q["best"] = best_quali_time(Q)
    Q["quali_gap_pct"] = (Q["best"] / Q.groupby(["season", "round"])["best"].transform("min") - 1) * 100
    R = R.merge(Q[["season", "round", "driver", "quali_gap_pct"]], on=["season", "round", "driver"], how="left")
    R = R.merge(_pace_pct(d["fits"], "R"), on=["season", "round", "driver"], how="left")
    R = R.merge(ev[["season", "round", "event_date", "track_id", "era_id"]], on=["season", "round"], how="left")
    return R.rename(columns={"event_date": "date"})


# ---------------------------------------------------------------------------- main builder
def build_features(store: Store, include_upcoming: bool = True) -> pd.DataFrame:
    """Feature + target table for every race with weekend data (targets NaN if not yet raced)."""
    d = _load(store)
    ev = d["events"].sort_values("event_date").reset_index(drop=True)
    res = d["results"]
    hist_all = race_history(d)
    pace_s = _pace_pct(d["fits"], "S")
    neutral = d["neutral"]
    ev["races_into_era"] = ev.groupby("era_id").cumcount()

    rows = []
    for _, e in ev.iterrows():
        s, r, date = int(e["season"]), int(e["round"]), e["event_date"]
        wk = res[(res["season"] == s) & (res["round"] == r)]
        R = wk[wk["session"] == "R"]
        Q = wk[wk["session"] == "Q"]
        base = R if len(R) else Q
        if base.empty or (R.empty and not include_upcoming):
            continue
        df = base[["driver", "driver_number", "team", "team_id"]].copy()
        df["team_key"] = df["team_id"].fillna(df["team"])
        df.insert(0, "round", r)
        df.insert(0, "season", s)
        df["date"] = date
        df["track_id"] = e["track_id"]
        df["era_id"] = e["era_id"]
        n = len(df)

        # ---- qualifying
        q = Q.assign(best=best_quali_time(Q)).set_index("driver")
        pole = q["best"].min()
        df["quali_pos"] = df["driver"].map(q["position"]).fillna(n)
        df["quali_gap_pct"] = df["driver"].map((q["best"] / pole - 1) * 100)
        df["quali_gap_pct"] = df["quali_gap_pct"].fillna(df["quali_gap_pct"].max() + 1.0 if df["quali_gap_pct"].notna().any() else np.nan)
        tm_other = df.groupby("team_key")["quali_gap_pct"].transform(
            lambda x: x.iloc[::-1].values if len(x) == 2 else np.full(len(x), np.nan))
        df["teammate_quali_gap_pct"] = df["quali_gap_pct"] - tm_other

        # ---- grid (pit-lane start = 0 in the timing data)
        if len(R):
            g = R.set_index("driver")["grid_position"]
            df["grid_pos"] = df["driver"].map(g)
        else:
            df["grid_pos"] = np.nan
        df["pit_lane_start"] = df["grid_pos"].eq(0).astype(float)
        df.loc[df["grid_pos"].eq(0), "grid_pos"] = n + 1
        df["grid_pos"] = df["grid_pos"].fillna(df["quali_pos"])

        # ---- Friday long runs: gap to the best run on the same compound in the same session
        lr = d["long_runs"]
        lr = lr[(lr["season"] == s) & (lr["round"] == r)] if len(lr) else lr
        if len(lr):
            lr = lr[~lr["compound"].str.contains("UNKNOWN|TEST", na=True)]
            best = lr.groupby(["session", "compound"])["mean_lap_s"].transform("min")
            lr = lr.assign(gap=(lr["mean_lap_s"] / best - 1) * 100)
            df["longrun_gap_pct"] = df["driver"].map(lr.groupby("driver")["gap"].min())
            df["n_long_runs"] = df["driver"].map(lr.groupby("driver").size()).fillna(0)
        else:
            df["longrun_gap_pct"] = np.nan
            df["n_long_runs"] = 0.0

        # ---- sprint
        S = wk[wk["session"] == "S"]
        df["is_sprint_weekend"] = float(len(S) > 0 or "sprint" in str(e["event_format"]))
        if len(S):
            S = S.assign(fo=finish_order(S))
            df["sprint_pos"] = df["driver"].map(S.set_index("driver")["fo"])
            ps = pace_s[(pace_s["season"] == s) & (pace_s["round"] == r)].set_index("driver")["pace_pct"]
            df["sprint_pace_pct"] = df["driver"].map(ps)
        else:
            df["sprint_pos"] = np.nan
            df["sprint_pace_pct"] = np.nan

        # ---- form (strictly earlier races)
        h = hist_all[hist_all["date"] < date] if len(hist_all) else hist_all
        if len(h):
            df["race_pace_ewm"] = df["driver"].map(ewm_by_key(h, "driver", "pace_pct"))
            team_race = h.groupby(["season", "round", "team_key", "date"], as_index=False)["pace_pct"].mean()
            df["team_race_pace_ewm"] = df["team_key"].map(ewm_by_key(team_race, "team_key", "pace_pct"))
            df["quali_gap_ewm"] = df["driver"].map(ewm_by_key(h, "driver", "quali_gap_pct"))
            df["finish_ewm"] = df["driver"].map(ewm_by_key(h, "driver", "finish_order"))
            df["points_ewm"] = df["driver"].map(ewm_by_key(h, "driver", "points"))
            recent = h[h["date"] >= h["date"].drop_duplicates().nlargest(10).min()]
            df["team_dnf_rate"] = df["team_key"].map(recent.groupby("team_key")["dnf"].mean())
            df["n_prev_races"] = df["driver"].map(h.groupby("driver").size()).fillna(0)

            # ---- track history
            th = h[h["track_id"] == e["track_id"]]
            if len(th):
                fin = th[~th["dnf"] & ~th["dns"] & th["grid_position"].gt(0)]
                rhos = [spearmanr(g["grid_position"], g["finish_order"]).statistic
                        for _, g in fin.groupby(["season", "round"]) if len(g) > 5]
                df["track_grid_finish_rho"] = float(np.nanmean(rhos)) if rhos else np.nan
                df["track_mean_places_changed"] = float((fin["grid_position"] - fin["finish_order"]).abs().mean())
                races = th[["season", "round"]].drop_duplicates()
                sc = races.merge(neutral, on=["season", "round"], how="left")["had_sc"]
                df["track_sc_share"] = float(sc.mean()) if sc.notna().any() else np.nan
                df["track_n_prev"] = float(len(races))
        for c in ["race_pace_ewm", "team_race_pace_ewm", "quali_gap_ewm", "finish_ewm", "points_ewm",
                  "team_dnf_rate", "n_prev_races", "track_grid_finish_rho", "track_mean_places_changed",
                  "track_sc_share", "track_n_prev"]:
            if c not in df:
                df[c] = np.nan
        df["n_prev_races"] = df["n_prev_races"].fillna(0)
        df["track_n_prev"] = df["track_n_prev"].fillna(0)
        df["races_into_era"] = float(e["races_into_era"])

        # ---- targets
        if len(R):
            Rt = R.assign(fo=finish_order(R)).set_index("driver")
            df["finish_order"] = df["driver"].map(Rt["fo"])
            df["points"] = df["driver"].map(Rt["points"]).fillna(0)
            st = df["driver"].map(Rt["status"]).fillna("")
            df["dnf"] = (df["driver"].map(Rt["position"]).isna() & ~st.isin(DNS_STATUSES)).astype(float)
        else:
            df["finish_order"] = np.nan
            df["points"] = np.nan
            df["dnf"] = np.nan
        df["won"] = (df["finish_order"] == 1).astype(float).where(df["finish_order"].notna())
        df["podium"] = (df["finish_order"] <= 3).astype(float).where(df["finish_order"].notna())
        df["top10"] = (df["finish_order"] <= 10).astype(float).where(df["finish_order"].notna())
        rows.append(df)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out
