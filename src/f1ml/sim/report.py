"""Markdown report for the simulator replay check."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1ml.eval.report import _md

NOTES = {
    "spearman": "rank correlation of simulated median position vs actual order",
    "pos_in_10_90": "share of drivers whose actual position is inside the simulated 10-90% range (ideal ~0.8)",
    "gap_mae_s": "mean |simulated median gap - actual gap| to the winner, lead-lap finishers (s)",
    "gap_in_10_90": "share of lead-lap finishers whose actual gap is inside the simulated 10-90% range (ideal ~0.8)",
    "winner_correct": "simulated favourite (best median position) actually won",
}


def write_sim_report(per_race: pd.DataFrame, per_driver: pd.DataFrame, path: Path) -> None:
    cols = list(NOTES)
    ok = per_race[per_race.get("error").isna()] if "error" in per_race else per_race
    dry = ok[~ok["wet"].astype(bool)]
    by_season = dry.groupby("season")[cols].mean().round(3)
    by_season.insert(0, "races", dry.groupby("season").size())
    overall = dry[cols].mean().round(3).to_frame("dry races").T
    overall.insert(0, "races", len(dry))
    worst = dry.sort_values("gap_mae_s", ascending=False).head(8)[["season", "round", *cols]].round(3)
    errors = per_race[per_race["error"].notna()][["season", "round", "error"]] if "error" in per_race else None
    parts = [
        "# Race simulator: replay check (phase 4)",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Each finished dry race is rebuilt from its own "
        "fitted pace and tyre model, with the strategies, Safety Cars/VSCs, retirements and lap-1 order that "
        "actually happened; overtaking, dirty air and pit loss come from components fitted only on earlier races. "
        "Buscombe's standard: the model must reproduce the real race before it earns an opinion about a fake one.",
        f"Wet or mixed-condition races skipped: {int(ok['wet'].astype(bool).sum())}.",
        "## Overall", _md(overall), "## By season", _md(by_season),
        "## Metrics", "\n".join(f"- **{k}**: {v}" for k, v in NOTES.items()),
        "## Largest gap errors", _md(worst.set_index(["season", "round"])),
    ]
    if errors is not None and len(errors):
        parts += ["## Races that could not be replayed", _md(errors.set_index(["season", "round"]))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
