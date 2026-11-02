"""Markdown report for a backtest run."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from f1ml.eval.backtest import BacktestResult

METRIC_NOTES = {
    "spearman": "rank correlation, predicted vs actual order (higher better)",
    "winner_logloss": "-log P(actual winner) (lower better; uniform guess over 20 cars = 3.00)",
    "brier_win": "Brier score of P(win) (lower better)",
    "brier_podium": "Brier score of P(podium) (lower better)",
    "winner_hit": "share of races the favourite won",
    "top3_overlap": "predicted top-3 drivers who finished top-3 (0-3)",
    "mae_pos": "mean |expected position - actual| (lower better)",
}


def _md(df: pd.DataFrame) -> str:
    d = df.reset_index()
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(res: BacktestResult, path, importance: pd.Series | None = None, notes: str = "") -> None:
    parts = [
        "# Backtest report",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Walk-forward: each race is predicted by a "
        "model trained only on earlier races (plus that weekend's practice, qualifying and sprint).",
        notes,
        "## Overall", _md(res.summary().round(3)),
        "## By season", _md(res.summary(["model", "season"]).round(3)),
        "## Metrics", "\n".join(f"- **{k}**: {v}" for k, v in METRIC_NOTES.items()),
    ]
    if importance is not None:
        imp = (importance / importance.sum() * 100).round(1).rename("gain_%").to_frame()
        parts += ["## Feature importance (final model, % of total gain)", _md(imp)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(p for p in parts if p) + "\n", encoding="utf-8")
