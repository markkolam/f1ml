"""Strategy briefing: one driver, one race, in Buscombe's terms."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1ml.eval.report import _md


def write_briefing(path: Path, title: str, context: dict, evaluation, crossover_df: pd.DataFrame,
                   robustness: pd.DataFrame | None = None, sweep_summary: pd.DataFrame | None = None,
                   rivals: pd.DataFrame | None = None, team: pd.DataFrame | None = None) -> None:
    t = evaluation.table
    top = t.iloc[0]
    n = int(t["votes"].sum())
    verdict = ("clear call" if top["vote_share"] >= 0.6 else
               "lean" if top["vote_share"] >= 0.35 else "close: the driver is the casting vote")
    parts = [
        f"# Strategy briefing: {title}",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.",
        "\n".join(f"- **{k}**: {v}" for k, v in context.items()),
        "## Plans",
        f"Best plan for **{evaluation.driver}** on **{evaluation.objective}**: **{top['plan']}**, best in "
        f"{int(top['votes'])} of {n} simulated races ({top['vote_share']:.0%}): {verdict}. "
        "`vs_best` is the paired difference from the best plan (same simulated races), with its standard error.",
        _md(t.round(3).set_index("plan")),
        "## When does another stop pay?",
        _md(crossover_df.round(3)) if len(crossover_df) else "No change in the best stop count across the deg range tried.",
    ]
    if robustness is not None:
        parts += ["## If the tyre model is wrong",
                  "Regret = objective lost against the best plan in that tyre world (0 = best there).",
                  _md(robustness.round(3))]
    if sweep_summary is not None:
        parts += ["## Safety Car on every lap",
                  "Share of Safety Car laps on which each plan was best, and its objective averaged over them.",
                  _md(sweep_summary.round(3))]
    if rivals is not None:
        parts += ["## Rivals' best responses", _md(rivals.round(3).set_index("driver"))]
    if team is not None:
        parts += ["## Teammate split (top pairs)", _md(team.head(8).round(3))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
