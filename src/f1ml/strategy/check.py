"""History check: what the pre-race strategy layer would have recommended to each
race's eventual winner, against what they actually ran.

Not a test of "the right answer" (the winner's plan was not necessarily optimal,
and strategies are contingent on the race), but a sanity check that the
recommended number of stops and compound order look like real F1 strategy.
"""

from __future__ import annotations

import pandas as pd

from f1ml.features import build_features
from f1ml.sim.params import PreraceContext, actual_strategies, prerace_params
from f1ml.sim.replay import components_as_of
from f1ml.sim.strategy import default_strategies, plan_text
from f1ml.store import Store
from f1ml.strategy.evaluate import evaluate_plans, strategy_candidates


def _seq(plan: str) -> str:
    return plan.split("@")[0]


def strategy_check(store: Store, seasons: list[int], n_sims: int = 600, progress=None) -> pd.DataFrame:
    feats = build_features(store, include_upcoming=False)
    feats = feats[feats["season"].isin(seasons)]
    rows = []
    for (season, rnd), g in feats.groupby(["season", "round"], sort=False):
        winner = g.loc[g["finish_order"] == 1, "driver"]
        if winner.empty:
            continue
        w = winner.iloc[0]
        try:
            laps = store.query(f"""SELECT driver, lap_number, compound, is_in_lap FROM clean_laps
                                   WHERE season={season} AND round={rnd} AND session='R' AND driver='{w}'""")
            act = actual_strategies(laps, [w])
            if act is None:
                rows.append({"season": season, "round": rnd, "winner": w, "skipped": "wet"})
                continue
            p = prerace_params(store, int(season), int(rnd), PreraceContext(components_as_of(store, g["date"].iloc[0]), g))
            act_plan = plan_text(act[0], p.compounds)
            ev = evaluate_plans(p, default_strategies(p), w, strategy_candidates(p), "points", n_sims=n_sims,
                                seed=int(season) * 100 + int(rnd))
            t = ev.table
            seqs = [_seq(x) for x in t["plan"]]
            rank = seqs.index(_seq(act_plan)) + 1 if _seq(act_plan) in seqs else None
            rows.append({
                "season": season, "round": rnd, "winner": w, "actual": act_plan, "recommended": t["plan"].iloc[0],
                "vote_share": t["vote_share"].iloc[0], "actual_stops": len(act[0].stops),
                "recommended_stops": int(t["stops"].iloc[0]),
                "same_stops": len(act[0].stops) == int(t["stops"].iloc[0]),
                "same_start_compound": act_plan[0] == t["plan"].iloc[0][0],
                "same_sequence": _seq(act_plan) == _seq(t["plan"].iloc[0]),
                "actual_sequence_rank": rank, "n_candidates": len(t),
            })
        except Exception as exc:  # noqa: BLE001 - report and continue
            rows.append({"season": season, "round": rnd, "winner": w, "skipped": f"{type(exc).__name__}: {exc}"})
        if progress:
            progress(rows[-1])
    return pd.DataFrame(rows)
