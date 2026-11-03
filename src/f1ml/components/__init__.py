"""Phase 3 component models: tyre, pit loss, overtaking + dirty air, neutralisation
hazard, start, car/driver pace. Each is fitted from the stored race data and can
be restricted to races before a date (`as_of`) for leakage-free use later.
"""

from __future__ import annotations

import pandas as pd

from f1ml.components.data import RaceData, load_race_data
from f1ml.components.flags import fit_flags
from f1ml.components.overtake import fit_overtake
from f1ml.components.pace import fit_pace
from f1ml.components.pitloss import fit_pitloss
from f1ml.components.start import fit_start
from f1ml.components.tyre import fit_tyre
from f1ml.store import Store

# tables written by `f1ml components` (large per-event tables are kept for inspection too)
COMPONENTS = ("tyre", "pitloss", "overtake", "flags", "start", "pace")


def fit_components(store: Store, as_of: pd.Timestamp | None = None,
                   only: tuple[str, ...] = COMPONENTS) -> dict[str, pd.DataFrame]:
    rd = load_race_data(store, as_of=as_of)
    out: dict[str, pd.DataFrame] = {}
    if "tyre" in only:
        out.update(fit_tyre(store, rd))
    if "pitloss" in only:
        out.update(fit_pitloss(rd))
    if "overtake" in only:
        out.update(fit_overtake(rd))
    if "flags" in only:
        out.update(fit_flags(rd))
    if "start" in only:
        out.update(fit_start(rd))
    if "pace" in only:
        out.update(fit_pace(rd))
    return out


__all__ = ["COMPONENTS", "RaceData", "fit_components", "load_race_data"]
