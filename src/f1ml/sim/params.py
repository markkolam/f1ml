"""Build simulator inputs (RaceParams) from stored data and the phase-3 components.

Two modes:

- `prerace_params`: only what is known before lights out. Components fitted on
  earlier races (`fit_components(as_of=race date)`), car pace from the pace
  tracker blended with this weekend's qualifying gap ("half the FP2 long run
  ... and half qualifying" in Buscombe's words; here half tracker, half quali),
  tyre/pit/overtake/hazard numbers for this track and era.
- `replay_inputs`: a finished race rebuilt from its own data: that race's fitted
  pace and tyre model, the strategies actually run, the actual neutralisations,
  retirements and lap-1 order. Buscombe's standard: "the model must reproduce the
  real race to the second before it earns an opinion about a fake one."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1ml.components.flags import neutral_type
from f1ml.sim.engine import GREEN, RED, SC, VSC, Overrides, RaceParams, Strategy
from f1ml.store import Store

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
TYPE_CODE = {"vsc": VSC, "sc": SC, "red": RED}
QUALI_WEIGHT = 0.5


def _row(df: pd.DataFrame, **eq) -> pd.Series | None:
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        m &= (df[k] == v).to_numpy()
    return df.loc[m].iloc[0] if m.any() else None


def _era(df: pd.DataFrame, era: str) -> str:
    """Era to read from a component table: `era` if it has data, else the latest earlier era
    (a rule reset starts from the previous era's knowledge), else the latest era available."""
    eras = sorted(set(df["era_id"].dropna())) if "era_id" in df else []
    if era in eras or not eras:
        return era
    earlier = [e for e in eras if e < era]
    return earlier[-1] if earlier else eras[-1]


def _track_row(df: pd.DataFrame, track: str, era: str, **eq) -> pd.Series | None:
    """Track-specific row for this era, else for the fallback era."""
    r = _row(df, track_id=track, era_id=era, **eq)
    fb = _era(df, era)
    return r if r is not None or fb == era else _row(df, track_id=track, era_id=fb, **eq)


def _event(store: Store, season: int, round_: int) -> pd.Series:
    ev = store.query(f"SELECT * FROM raw_events WHERE season={season} AND round={round_}")
    if ev.empty:
        raise LookupError(f"no event {season} R{round_}")
    e = ev.iloc[0].copy()
    e["event_date"] = pd.Timestamp(e["event_date"])
    return e


def _track_history(store: Store, track_id: str, before: pd.Timestamp) -> pd.DataFrame:
    """Per past race at this track: reference lap, fuel effect, fit sigma, laps, pole time."""
    q = store.query(f"""
        WITH ev AS (SELECT season, round, track_id, era_id, event_date FROM raw_events),
        fp AS (SELECT season, round,
                      max(CASE WHEN param='reference_offset' THEN value END) ref,
                      max(CASE WHEN param='fuel_effect' THEN value END) fuel,
                      max(CASE WHEN param='sigma' THEN value END) sigma
               FROM features_fit_params WHERE session='R' GROUP BY 1, 2),
        lp AS (SELECT season, round, max(lap_number) laps FROM clean_laps WHERE session='R' GROUP BY 1, 2),
        qp AS (SELECT season, round, min(least(coalesce(q1_s, 1e9), coalesce(q2_s, 1e9), coalesce(q3_s, 1e9))) pole
               FROM raw_results WHERE session='Q' GROUP BY 1, 2)
        SELECT ev.*, fp.ref, fp.fuel, fp.sigma, lp.laps, qp.pole FROM ev
        JOIN fp USING (season, round) LEFT JOIN lp USING (season, round) LEFT JOIN qp USING (season, round)
        WHERE ev.event_date < TIMESTAMP '{before}'""")
    q["is_track"] = q["track_id"] == track_id
    return q


@dataclass
class PreraceContext:
    """Everything `prerace_params` needs, so a backtest can reuse fitted components."""
    comps: dict[str, pd.DataFrame]
    features: pd.DataFrame          # pre-race feature rows for this race (one per driver)


def prerace_params(store: Store, season: int, round_: int, ctx: PreraceContext,
                   laps: int | None = None) -> RaceParams:
    e = _event(store, season, round_)
    track, era, date = e["track_id"], e["era_id"], e["event_date"]
    c = ctx.comps
    f = ctx.features.sort_values("grid_pos").reset_index(drop=True)
    C = len(f)

    hist = _track_history(store, track, date)
    th = hist[hist["is_track"]]
    era_h = hist[hist["era_id"] == era] if (hist["era_id"] == era).any() else hist
    if len(th) and th["ref"].notna().any():
        ref = float(th["ref"].median())
    else:  # new venue: scale this weekend's pole by the usual race-lap / pole ratio
        ratio = float((hist["ref"] / hist["pole"]).median())
        q = store.query(f"""SELECT min(least(coalesce(q1_s,1e9), coalesce(q2_s,1e9), coalesce(q3_s,1e9))) pole
                            FROM raw_results WHERE season={season} AND round={round_} AND session='Q'""")
        ref = float(q["pole"].iloc[0]) * ratio
    fuel = float(th["fuel"].median()) if len(th) else float(era_h["fuel"].median())
    sigma = float(th["sigma"].median()) if len(th) else float(era_h["sigma"].median())
    if laps is None:
        laps = int(th["laps"].median()) if len(th) and th["laps"].notna().any() else int(round(305_000 / (ref * 55)))

    # ---- car pace: pace tracker (team + driver) blended with this weekend's qualifying gap
    tn = c["pace_team_now"].set_index("team")
    dn = c["pace_driver_now"].set_index("driver")
    team_m = f["team"].map(tn["state_mean"])
    team_sd = f["team"].map(tn["state_sd"])
    drv_m = f["driver"].map(dn["state_mean"]).fillna(0.0)
    drv_sd = f["driver"].map(dn["state_sd"]).fillna(0.2)
    quali = f["quali_gap_pct"] - f["quali_gap_pct"].median()
    tracker = team_m + drv_m
    pace_pct = np.where(tracker.notna(), (1 - QUALI_WEIGHT) * tracker + QUALI_WEIGHT * quali.fillna(tracker), quali)
    pace_pct = pd.Series(pace_pct).fillna(np.nanmax(pace_pct) if np.isfinite(pace_pct).any() else 0.0)
    sd_pct = np.sqrt((team_sd.fillna(0.5) ** 2 + drv_sd ** 2)) * (1 - QUALI_WEIGHT) + 0.08
    pace_s = pace_pct.to_numpy() / 100 * ref
    pace_sd_s = sd_pct.to_numpy() / 100 * ref

    # ---- tyres: pooled per track x era x compound, else era means; temperature-adjusted
    tt = c["tyre_track"]
    gamma = float(c["tyre_params"].set_index("param").loc["temp_gamma", "value"])
    tp = c["tyre_params"].set_index("param")["value"]
    ref_temp = float(tp.get(f"ref_temp:{era}", tp.get(f"ref_temp:{_era(c['tyre_track'], era)}", np.nan)))
    w = store.query(f"""SELECT avg(track_temp) t FROM raw_weather
                        WHERE season={season} AND round={round_} AND session IN ('Q', 'FP2', 'FP3')""")
    race_temp = float(w["t"].iloc[0]) if len(w) and pd.notna(w["t"].iloc[0]) else ref_temp
    dt = (race_temp - ref_temp) if np.isfinite(ref_temp) else 0.0
    base, deg = [], []
    for comp in COMPOUNDS:
        r = _track_row(tt, track, era, compound=comp)
        e_rows = tt[(tt["era_id"] == _era(tt, era)) & (tt["compound"] == comp)]
        d = r["deg_post"] if r is not None else (e_rows["deg_era_mean"].iloc[0] if len(e_rows) else 0.06)
        if comp == "MEDIUM":
            b = 0.0
        elif r is not None and pd.notna(r["base_post"]):
            b = r["base_post"]
        else:
            b = e_rows["base_raw"].median() if len(e_rows) else {"SOFT": -0.4, "HARD": 0.4}[comp]
        deg.append(max(float(d) + gamma * dt, 0.0))
        base.append(float(b))

    # ---- pit loss
    pl = c["pitloss_track"]
    loss = {}
    for cond, code in (("green", GREEN), ("vsc", VSC), ("sc", SC)):
        r = _track_row(pl, track, era, condition=cond)
        e_rows = pl[(pl["era_id"] == _era(pl, era)) & (pl["condition"] == cond)]
        loss[code] = float(r["loss_post"]) if r is not None else float(e_rows["loss_era_mean"].iloc[0])

    # ---- overtaking + dirty air
    ot = c["overtake_track"]
    r = _track_row(ot, track, era)
    er = _row(c["overtake_era"], era_id=_era(c["overtake_era"], era))
    a, b = (float(r["a"]), float(r["b"])) if r is not None else (float(er["a"]), float(er["b"]))
    shift = f["driver"].map(c["overtake_driver"].set_index("driver")["logit_shift"]).fillna(0.0).to_numpy()
    da = c["dirty_air_track"]
    r = _track_row(da, track, era)
    dirty = float(r["penalty_post"]) if r is not None else \
        float(da.loc[da["era_id"] == _era(da, era), "penalty_era_mean"].iloc[0])

    # ---- neutralisation hazard, mix and durations
    co = c["flags_coefs"].set_index("param")["value"]
    ft = _row(c["flags_track"], track_id=track)
    u, v = (float(ft["u"]), float(ft["v"])) if ft is not None else (0.0, 0.0)
    sh = _row(c["flags_type_share"], era_id=_era(c["flags_type_share"], era))
    share = {VSC: float(sh.get("vsc", 0.4)), SC: float(sh.get("sc", 0.55)), RED: float(sh.get("red", 0.05))} \
        if sh is not None else {VSC: 0.4, SC: 0.55, RED: 0.05}
    du = c["flags_durations"]
    du = du[du["era_id"] == _era(du, era)]
    durs = {TYPE_CODE[t]: list(g["laps"].astype(int)) or [3] for t, g in du.groupby("type")}
    for k, default in ((VSC, [2]), (SC, [4]), (RED, [2])):
        durs.setdefault(k, default)

    # ---- start
    ss = c["start_slot"]
    ss = ss[ss["era_id"] == _era(ss, era)]
    slots = ss.groupby("grid_position")[["mean", "std", "p_out_lap1"]].mean()
    idx = np.arange(1, C + 1)
    gm = slots["mean"].reindex(idx).fillna(0.0).to_numpy()
    gs = slots["std"].reindex(idx).fillna(2.0).to_numpy()
    po = slots["p_out_lap1"].reindex(idx).fillna(0.02).to_numpy()
    cc = c["start_concertina"]
    r = _track_row(cc, track, era)
    conc = float(r["median"]) if r is not None else float(cc.loc[cc["era_id"] == _era(cc, era), "median"].median())

    dnf_rate = f["team_dnf_rate"].fillna(0.06).clip(0.01, 0.3).to_numpy()
    grid = f["grid_pos"].to_numpy(float)
    return RaceParams(
        drivers=list(f["driver"]), grid=grid, pace_s=pace_s, pace_sd_s=pace_sd_s, compounds=list(COMPOUNDS),
        base_s=np.array(base), deg_s=np.array(deg), laps=laps, ref_lap_s=ref, fuel_s_per_lap=fuel,
        lap_noise_s=sigma, concertina_s=conc if np.isfinite(conc) else 12.0, pit_loss_s=loss,
        pass_a=a, pass_b=b, driver_pass_shift=shift, dirty_air_s=dirty,
        hz_intercept=float(co["intercept"]) + u, hz_lap1=float(co["lap1"]) + v, hz_restart=float(co["restart"]),
        hz_pit=float(co["pit_activity"]), hz_late=float(co["late"]), hz_frac=float(co["race_frac"]),
        type_share=share, durations=durs, start_gain_mean=gm, start_gain_sd=gs, p_out_lap1=po,
        dnf_per_lap=dnf_rate / laps,
    )


# ============================================================================ replay
@dataclass
class ReplayInputs:
    params: RaceParams
    strategies: list[Strategy]
    overrides: Overrides
    actual: pd.DataFrame          # driver, finish_order, gap_s (to winner, lead-lap finishers), finished
    wet: bool


def actual_strategies(laps: pd.DataFrame, drivers: list[str]) -> list[Strategy] | None:
    out = []
    for d in drivers:
        L = laps[laps["driver"] == d].sort_values("lap_number")
        if L.empty:
            out.append(Strategy(COMPOUNDS.index("MEDIUM")))
            continue
        comps = L["compound"].astype(str).tolist()
        if any(cp not in COMPOUNDS for cp in comps if cp != "UNKNOWN"):
            return None  # wet tyres used: not a dry replay
        comps = [cp if cp in COMPOUNDS else "MEDIUM" for cp in comps]
        stops = []
        lapn = L["lap_number"].tolist()
        inl = L["is_in_lap"].tolist()
        for k in range(len(lapn) - 1):
            if inl[k] and comps[k + 1]:
                stops.append((int(lapn[k]), COMPOUNDS.index(comps[k + 1])))
        out.append(Strategy(COMPOUNDS.index(comps[0]), stops))
    return out


def replay_inputs(store: Store, season: int, round_: int, comps: dict[str, pd.DataFrame]) -> ReplayInputs:
    e = _event(store, season, round_)
    track, era = e["track_id"], e["era_id"]
    laps = store.query(f"""SELECT driver, lap_number, time_s, compound, is_in_lap, track_status
                           FROM clean_laps WHERE season={season} AND round={round_} AND session='R'""")
    res = store.query(f"""SELECT driver, position, grid_position, status, laps FROM raw_results
                          WHERE season={season} AND round={round_} AND session='R'""")
    fp = store.query(f"""SELECT param, value FROM features_fit_params
                         WHERE season={season} AND round={round_} AND session='R'""").set_index("param")["value"]
    drivers = [d for d in res.sort_values("grid_position")["driver"] if d in set(laps["driver"])]
    C, L = len(drivers), int(laps["lap_number"].max())

    strategies = actual_strategies(laps, drivers)
    wet = strategies is None
    if wet:
        strategies = [Strategy(1) for _ in drivers]

    # this race's own fit: pace offsets, compound base and deg, fuel, reference lap, noise
    off = np.array([fp.get(f"offset:{d}", np.nan) for d in drivers], float)
    off = np.where(np.isfinite(off), off, np.nanmax(off))
    ref = float(fp["reference_offset"])
    base = np.array([fp.get(f"compound_offset:{cp}", np.nan) for cp in COMPOUNDS], float)
    med = base[1] if np.isfinite(base[1]) else np.nanmin(base)
    base = np.where(np.isfinite(base), base - med, 0.0)
    deg = np.array([fp.get(f"deg:{cp}", np.nan) for cp in COMPOUNDS], float)
    deg = np.where(np.isfinite(deg), deg, np.nanmean(deg))

    # actual neutralisation schedule (leader's lap status), retirements and lap-1 times
    lead = laps.sort_values("time_s").groupby("lap_number").first()
    sched = np.zeros(L + 1, np.int8)
    for n, st in lead["track_status"].astype(str).items():
        t = neutral_type(st)
        sched[int(n)] = TYPE_CODE[t] if t else GREEN
    done = laps.groupby("driver")["lap_number"].max()
    finished = res.set_index("driver")["position"].notna()
    retire = np.array([np.inf if finished.get(d, False) else float(done.get(d, 0)) for d in drivers])
    l1 = laps[laps["lap_number"] == 1].set_index("driver")["time_s"]
    lap1 = np.array([l1.get(d, np.inf) for d in drivers], float)
    lap1 = lap1 - np.nanmin(lap1[np.isfinite(lap1)]) + ref + 6.0

    # shared behaviour from the components (as of this race)
    ot = _track_row(comps["overtake_track"], track, era)
    er = _row(comps["overtake_era"], era_id=_era(comps["overtake_era"], era))
    a, b = (float(ot["a"]), float(ot["b"])) if ot is not None else (float(er["a"]), float(er["b"]))
    shift = pd.Series(drivers).map(comps["overtake_driver"].set_index("driver")["logit_shift"]).fillna(0).to_numpy()
    da = comps["dirty_air_track"]
    r = _track_row(da, track, era)
    dirty = float(r["penalty_post"]) if r is not None else \
        float(da.loc[da["era_id"] == _era(da, era), "penalty_era_mean"].iloc[0])
    pl = comps["pitloss_track"]
    loss = {}
    for cond, code in (("green", GREEN), ("vsc", VSC), ("sc", SC)):
        r = _track_row(pl, track, era, condition=cond)
        loss[code] = float(r["loss_post"]) if r is not None else \
            float(pl.loc[(pl["era_id"] == _era(pl, era)) & (pl["condition"] == cond), "loss_era_mean"].iloc[0])

    p = RaceParams(
        drivers=drivers, grid=res.set_index("driver").loc[drivers, "grid_position"].replace(0, C + 1).to_numpy(float),
        pace_s=off, pace_sd_s=np.zeros(C), compounds=list(COMPOUNDS), base_s=base, deg_s=deg, laps=L,
        ref_lap_s=ref, fuel_s_per_lap=float(fp["fuel_effect"]), lap_noise_s=float(fp["sigma"]),
        pit_loss_s=loss, pass_a=a, pass_b=b, driver_pass_shift=shift, dirty_air_s=dirty,
    )
    # actual result: order, and gap to the winner for cars that finished on the lead lap
    last = laps[laps["lap_number"] == L].set_index("driver")["time_s"]
    win_t = last.min()
    from f1ml.features.prerace import finish_order
    res = res.assign(finish_order=finish_order(res))
    act = pd.DataFrame({"driver": drivers})
    act["finish_order"] = act["driver"].map(res.set_index("driver")["finish_order"])
    act["gap_s"] = act["driver"].map(last - win_t)
    act["finished"] = act["driver"].map(finished).fillna(False)
    ov = Overrides(neutral=sched, retire_lap=retire, lap1_time=lap1, no_reactive=True)
    return ReplayInputs(p, strategies, ov, act, wet)
