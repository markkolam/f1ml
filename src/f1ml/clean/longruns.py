"""Long-run extraction (practice race simulations and race stints).

Buscombe: "best clean stint of five laps or more". Practice fuel loads are not
published, so the raw slope understates deg by the (unknown) fuel effect; we
report both the raw slope and a fuel-adjusted deg using the configured prior
fuel effect (phase 3 treats fuel load as a latent variable instead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _longest_consecutive(lap_numbers: np.ndarray) -> np.ndarray:
    """Boolean mask of the longest run of consecutive lap numbers (input sorted)."""
    if len(lap_numbers) == 0:
        return np.zeros(0, bool)
    breaks = np.concatenate([[True], np.diff(lap_numbers) != 1])
    run_id = np.cumsum(breaks)
    counts = np.bincount(run_id)
    best = int(np.argmax(counts))
    return run_id == best


def extract_long_runs(laps: pd.DataFrame, *, min_laps: int = 5, fuel_effect_prior: float = 0.045) -> pd.DataFrame:
    rows = []
    clean = laps[laps["is_clean"] & laps["stint"].notna()]
    for (driver, stint), g in clean.groupby(["driver", "stint"], sort=True):
        g = g.sort_values("lap_number")
        mask = _longest_consecutive(g["lap_number"].to_numpy(int))
        run = g[mask]
        if len(run) < min_laps:
            continue
        x = run["tyre_life"].to_numpy(float)
        y = run["lap_time_s"].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else np.nan
        rows.append({
            "driver": driver,
            "team": run["team"].iloc[0] if "team" in run else pd.NA,
            "stint": float(stint),
            "compound": run["compound"].iloc[0],
            "c_number": run["c_number"].iloc[0] if "c_number" in run else pd.NA,
            "n_laps": len(run),
            "first_lap": int(run["lap_number"].iloc[0]),
            "start_tyre_life": float(x[0]),
            "mean_lap_s": float(y.mean()),
            "median_lap_s": float(np.median(y)),
            "slope_s_per_lap": slope,
            "deg_fuel_adj_s_per_lap": slope + fuel_effect_prior if np.isfinite(slope) else np.nan,
            "mean_track_temp": float(run["track_temp"].mean()) if "track_temp" in run else np.nan,
        })
    cols = ["driver", "team", "stint", "compound", "c_number", "n_laps", "first_lap", "start_tyre_life",
            "mean_lap_s", "median_lap_s", "slope_s_per_lap", "deg_fuel_adj_s_per_lap", "mean_track_temp"]
    return pd.DataFrame(rows, columns=cols)
