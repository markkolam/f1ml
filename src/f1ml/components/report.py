"""Markdown report for the component models, with Buscombe's published figures alongside."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from f1ml.eval.report import _md


def _get(df: pd.DataFrame, col: str, **eq):
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        m &= (df[k] == v).to_numpy()
    s = df.loc[m, col]
    return float(s.iloc[0]) if len(s) else np.nan


def benchmark_table(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    E26 = "2026_active_aero"
    tyre, pit, ot, dd = t["tyre_track"], t["pitloss_track"], t["overtake_track"], t["dirty_air_track"]
    att, flags, conc, drv = t["overtake_attempts"], t["flags_track"], t["start_concertina"], t["overtake_driver"]
    bud = att[(att["track_id"] == "hungaroring") & (att["season"] == 2026)]
    rows = [
        ("Monza 2026 race deg (s/lap)", "0.03", _get(tyre, "deg_raw", track_id="monza", era_id=E26, compound="MEDIUM")),
        ("Hungary 2026 green pit loss (s)", "21.9 measured", _get(pit, "loss_post", track_id="hungaroring", era_id=E26, condition="green")),
        ("Austin 2026 green pit loss (s)", "20.5-21.5", _get(pit, "loss_post", track_id="americas", era_id=E26, condition="green")),
        ("Mexico 2026 green pit loss (s)", "20.5", _get(pit, "loss_post", track_id="rodriguez", era_id=E26, condition="green")),
        ("Madrid 2026 green pit loss (s)", "~24 (pre-race estimate)", _get(pit, "loss_post", track_id="madring", era_id=E26, condition="green")),
        ("Budapest 2026 pass attempts (<=1.2 s, green)", "225", float(len(bud))),
        ("Budapest 2026 passes by next lap", "28", float(bud["passed"].sum())),
        ("Budapest 2026 50% pass threshold (s/lap)", "1.09", _get(ot, "threshold_50", track_id="hungaroring", era_id=E26)),
        ("Austin 2026 50% pass threshold (s/lap)", "0.5-0.7", _get(ot, "threshold_50", track_id="americas", era_id=E26)),
        ("Singapore 50% pass threshold (s/lap)", "2.5", _get(ot, "threshold_50", track_id="marina_bay", era_id=E26)),
        ("Monaco 50% pass threshold (s/lap)", "4.5", _get(ot, "threshold_50", track_id="monaco", era_id=E26)),
        ("Verstappen pass-curve shift (s/lap)", "~0.2 better than field", _get(drv, "pace_equiv_s", driver="VER")),
        ("Dirty-air cost, 2026 era mean (s/lap)", "~0.3 (Norris, Austin)", float(dd.loc[dd["era_id"] == E26, "penalty_era_mean"].iloc[0])),
        ("Mexico late-race neutralisation hazard (per lap)", "1.3%", _get(flags, "h_late", track_id="rodriguez") * 100),
        ("Budapest 2026 lap-1 concertina (s)", "11.0", _get(conc, "median", track_id="hungaroring", era_id=E26)),
    ]
    return pd.DataFrame(rows, columns=["quantity", "buscombe", "f1ml"]).set_index("quantity")


def write_components_report(t: dict[str, pd.DataFrame], path: Path) -> None:
    E26 = "2026_active_aero"
    tyre = t["tyre_track"]
    tyre26 = tyre[tyre["era_id"] == E26].pivot_table(index="track_id", columns="compound", values="deg_post").round(3)
    pit = t["pitloss_track"]
    pit26 = pit[(pit["era_id"] == E26)].pivot_table(index="track_id", columns="condition", values="loss_post").round(1)
    ot = t["overtake_track"]
    ot26 = ot[ot["era_id"] == E26].set_index("track_id")[["n_attempts", "pass_rate", "threshold_50", "threshold_90"]]
    ot26 = ot26.sort_values("threshold_50").round(2)
    fl = t["flags_track"].set_index("track_id")[["n_races", "obs_events_per_race", "p_lap1", "h_green_mid",
                                                 "h_pit_window", "h_late"]].round(3)
    params = t["tyre_params"].round(4).set_index("param")
    coefs = t["flags_coefs"].round(3).set_index("param")
    pace = t["pace_team_now"]
    pace26 = pace[pace["era_id"] == E26].sort_values("state_mean").set_index("team")[["state_mean", "state_sd"]].round(3)
    drv = t["overtake_driver"]
    drv = drv[drv["n_attempts"] >= 150].sort_values("pace_equiv_s", ascending=False).set_index("driver").round(3)
    parts = [
        "# Component models (phase 3)",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from 2023-2026 race data (OpenF1).",
        "## Against Ruth Buscombe's published figures", _md(benchmark_table(t).round(3)),
        "Notes: our pass thresholds use the expected (race-average) pace gap, which is smoother than a lap-by-lap "
        "gap, so they sit ~0.3-0.6 s above hers; track ordering agrees. Green pit losses run ~1-3 s above her "
        "figures: we count the whole in-lap and out-lap against cars that stayed out (cold-tyre out-lap "
        "included). VSC/SC pit losses are derived from the measured green loss with her ~0.52 ratio (not "
        "measurable from lap sums behind a Safety Car). The restart effect on neutralisation hazard came out "
        "negative in this data (no post-restart bump), unlike her flag model.",
        "## Tyre: 2026 deg by track (s/lap per lap of tyre age, temperature-adjusted, pooled)", _md(tyre26),
        "Tyre parameters (temp_gamma = extra deg per lap of age per degree C of track temperature):", _md(params),
        "## Pit loss, 2026 (s)", _md(pit26),
        "## Overtaking, 2026: pass thresholds (s/lap of pace advantage)", _md(ot26),
        "Driver pass-curve shifts (s/lap equivalent; positive = converts better than the field):", _md(drv.head(10)),
        "## Neutralisation hazard", "Coefficients (logit scale):", _md(coefs),
        "Per track (probabilities per lap at risk):", _md(fl),
        "Event mix by era:", _md(t["flags_type_share"].set_index("era_id").round(2)),
        "## Car pace now (2026, % of lap vs field median; negative = faster)", _md(pace26),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
