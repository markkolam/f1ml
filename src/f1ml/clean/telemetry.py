"""Telemetry features. "The window" (Buscombe): the share of lap *time* spent in
the speed band where the tyre, not power or the wing, limits the car
(about 70-150 km/h). Austin 16%, Singapore 22%, Monaco 29%, Monza 6%; about 70%
on a Safety Car restart lap.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def window_stats(time_s: np.ndarray, speed_kph: np.ndarray, throttle: np.ndarray | None = None,
                 low: float = 70.0, high: float = 150.0) -> dict[str, float]:
    """Time-weighted fractions over one lap of telemetry samples."""
    t = np.asarray(time_s, float)
    v = np.asarray(speed_kph, float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < 3:
        return {"window_frac": np.nan, "full_throttle_frac": np.nan, "min_speed_kph": np.nan}
    dt = np.diff(t)
    vm = v[:-1]
    total = dt.sum()
    in_window = ((vm >= low) & (vm <= high)) * dt
    res = {"window_frac": float(in_window.sum() / total) if total > 0 else np.nan,
           "min_speed_kph": float(v.min())}
    if throttle is not None:
        th = np.asarray(throttle, float)[ok][:-1]
        res["full_throttle_frac"] = float(((th >= 99) * dt).sum() / total) if total > 0 else np.nan
    else:
        res["full_throttle_frac"] = np.nan
    return res


def lap_window_table(ff1_laps, low: float = 70.0, high: float = 150.0) -> pd.DataFrame:
    """Per-lap window stats from a loaded FastF1 Laps object (session loaded with telemetry)."""
    rows = []
    for _, lap in ff1_laps.iterlaps():
        if pd.isna(lap.get("LapTime")):
            continue
        try:
            car = lap.get_car_data()
        except Exception as exc:  # missing telemetry for a lap is common; skip it
            log.debug("no car data for %s lap %s: %s", lap.get("Driver"), lap.get("LapNumber"), exc)
            continue
        if car is None or len(car) == 0:
            continue
        stats = window_stats(car["Time"].dt.total_seconds().to_numpy(), car["Speed"].to_numpy(),
                             car["Throttle"].to_numpy() if "Throttle" in car else None, low, high)
        rows.append({"driver": lap["Driver"], "lap_number": int(lap["LapNumber"]), **stats})
    return pd.DataFrame(rows, columns=["driver", "lap_number", "window_frac", "full_throttle_frac", "min_speed_kph"])
