"""Named model configurations used by `f1ml backtest` and `f1ml predict`.

All models calibrate their probability scale on the top-3 finishing order
(tau_top_k=3): the full order is dominated by noise at the back of the field
(DNFs, lapped traffic), which made win probabilities far too flat.
Chosen from the 2024-2026 walk-forward comparison (see reports/backtest.md).
"""

from f1ml.features import FEATURES
from f1ml.models.ranker import LGBMRaceRanker, LogPositionPL

MODEL_ZOO = {
    "grid": lambda: LogPositionPL("grid_pos", tau_top_k=3),
    "quali": lambda: LogPositionPL("quali_pos", tau_top_k=3),
    # default: LightGBM ranker blended with the grid, blend weights fitted on out-of-fold races
    "lgbm": lambda: LGBMRaceRanker(FEATURES, tau_top_k=3, blend_with="grid_pos"),
    "lgbm_pure": lambda: LGBMRaceRanker(FEATURES, tau_top_k=3),
    # phase-4 race simulator (components fitted as of each race; slower: ~10 s per race)
    "sim": lambda: __import__("f1ml.sim.model", fromlist=["SimRaceModel"]).SimRaceModel(n_sims=2000),
}
