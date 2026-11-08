"""Per-race joint fit of fuel(+track evolution), tyre degradation and car offsets.

    lap_time = offset[driver] + base[compound] + base_fuel[compound] * f - fuel_effect * (lap - 1)
               + (deg[compound] + deg_fuel * f) * tyre_life + e

with f the fuel fraction: 1 on lap 1, 0 on the last lap.

`base[compound]` is each compound's pace when new *on an empty tank*, relative
to the most-used compound (e.g. "the Hard is 0.6 s slower"). Without it, a
compound's pace difference leaks into its deg estimate.

`base_fuel[compound]` is how much that difference changes from an empty to a
full tank. Buscombe: "On full tanks, deg and fuel cancel, and the three
compounds collapse into one tyre ... The compound is worth nothing at lap 5 and
everything at lap 50." If she is right, the Soft's base_fuel is positive (its
advantage shrinks on heavy fuel). `deg_fuel` lets wear depend on fuel load too
(more load, more energy into the tread; graining "on high fuel").

fitted on clean, free-air, dry race laps by robust (Huber-IRLS) least squares,
with weak Gaussian priors on fuel_effect and deg (Buscombe's typical ranges:
fuel 0.03-0.065 s/lap, deg 0.02-0.10 s/lap). Tyre life resets at each stop
while race lap keeps counting, which is what separates deg from fuel burn.

Fuel burn and track evolution both scale with race lap, so `fuel_effect`
is their combined effect. This is the phase-1 baseline used for corrected
lap times; phase 3 replaces it with the hierarchical Bayesian tyre model
(temperature, graining, cliff, driver mode).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1ml.clean.tagging import DRY_COMPOUNDS


@dataclass
class FitResult:
    params: pd.DataFrame          # long format: param, value, se, n_laps, method
    fuel_effect: float
    offsets: dict[str, float]     # driver -> offset relative to field median
    deg: dict[str, float]
    compound_offsets: dict[str, float]
    sigma: float
    n_laps: int
    compound_fuel: dict[str, float] | None = None   # change in compound offset from empty to full tank
    deg_fuel: float = 0.0                           # change in deg (s/lap per lap) from empty to full
    race_laps: int | None = None


def _solve(X, y, w, prior_rows, sigma):
    sw = np.sqrt(w)
    A = X * sw[:, None]
    b = y * sw
    if prior_rows:
        P = np.zeros((len(prior_rows), X.shape[1]))
        t = np.zeros(len(prior_rows))
        for i, (j, mean, sd) in enumerate(prior_rows):
            scale = sigma / sd
            P[i, j] = scale
            t[i] = mean * scale
        A = np.vstack([A, P])
        b = np.concatenate([b, t])
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return beta, A


def fit_race(
    laps: pd.DataFrame,
    *,
    fuel_prior: tuple[float, float] = (0.045, 0.03),
    deg_prior: tuple[float, float] = (0.05, 0.15),
    huber_k: float = 1.5,
    min_laps: int = 60,
    iterations: int = 8,
    compound_prior_sd: float = 1.0,
    fuel_interaction: bool = True,
    compound_fuel_prior_sd: float = 0.5,
    deg_fuel_prior_sd: float = 0.05,
    race_laps: int | None = None,
) -> FitResult | None:
    dry = laps["compound"].isin(DRY_COMPOUNDS) & laps["tyre_life"].notna()
    method = "clean_free_air"
    data = laps[laps["is_clean_free_air"] & dry]
    if len(data) < min_laps:
        method = "clean_incl_dirty_air"
        data = laps[laps["is_clean"] & dry]
    if len(data) < min_laps:
        return None

    drivers = sorted(data["driver"].astype(str).unique())
    compounds = sorted(data["compound"].astype(str).unique())
    ref_compound = data["compound"].astype(str).value_counts().idxmax()
    others = [c for c in compounds if c != ref_compound]
    n, kd, kc = len(data), len(drivers), len(compounds)
    ko = kd + 1 + kc  # first compound-offset column
    kf = ko + len(others)  # first compound x fuel column (then one deg x fuel column)
    n_fuel = (len(others) + 1) if fuel_interaction else 0
    X = np.zeros((n, kf + n_fuel))
    N = int(race_laps or laps["lap_number"].max())
    f = 1.0 - (data["lap_number"].to_numpy(float) - 1.0) / max(N - 1, 1)
    d_idx = data["driver"].astype(str).map({d: i for i, d in enumerate(drivers)}).to_numpy()
    X[np.arange(n), d_idx] = 1.0
    X[:, kd] = -(data["lap_number"].to_numpy(float) - 1.0)
    c_idx = data["compound"].astype(str).map({c: i for i, c in enumerate(compounds)}).to_numpy()
    X[np.arange(n), kd + 1 + c_idx] = data["tyre_life"].to_numpy(float)
    comp = data["compound"].astype(str).to_numpy()
    for j, c in enumerate(others):
        X[:, ko + j] = (comp == c).astype(float)
        if fuel_interaction:
            X[:, kf + j] = (comp == c) * f
    if fuel_interaction:
        X[:, kf + len(others)] = data["tyre_life"].to_numpy(float) * f
    y = data["lap_time_s"].to_numpy(float)

    priors = [(kd, *fuel_prior)] + [(kd + 1 + i, *deg_prior) for i in range(kc)]
    priors += [(ko + j, 0.0, compound_prior_sd) for j in range(len(others))]
    if fuel_interaction:
        priors += [(kf + j, 0.0, compound_fuel_prior_sd) for j in range(len(others))]
        priors += [(kf + len(others), 0.0, deg_fuel_prior_sd)]
    w = np.ones(n)
    sigma = max(np.std(y - np.median(y)), 0.3)
    for _ in range(iterations):
        beta, A = _solve(X, y, w, priors, sigma)
        r = y - X @ beta
        sigma = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-3)
        a = np.abs(r) / sigma
        w = np.where(a <= huber_k, 1.0, huber_k / np.maximum(a, 1e-12))

    try:
        cov = np.linalg.inv(A.T @ A) * sigma**2
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)

    raw_offsets = beta[:kd]
    ref = float(np.median(raw_offsets))
    offsets = {d: float(raw_offsets[i] - ref) for i, d in enumerate(drivers)}
    deg = {c: float(beta[kd + 1 + i]) for i, c in enumerate(compounds)}
    fuel = float(beta[kd])
    comp_off = {ref_compound: 0.0, **{c: float(beta[ko + j]) for j, c in enumerate(others)}}
    comp_fuel = {ref_compound: 0.0, **{c: float(beta[kf + j]) for j, c in enumerate(others)}} \
        if fuel_interaction else {c: 0.0 for c in compounds}
    deg_fuel = float(beta[kf + len(others)]) if fuel_interaction else 0.0

    rows = [("fuel_effect", fuel, se[kd]), ("sigma", sigma, np.nan), ("reference_offset", ref, np.nan)]
    rows += [(f"deg:{c}", deg[c], se[kd + 1 + i]) for i, c in enumerate(compounds)]
    rows += [(f"compound_offset:{ref_compound}", 0.0, 0.0)]
    rows += [(f"compound_offset:{c}", comp_off[c], se[ko + j]) for j, c in enumerate(others)]
    if fuel_interaction:
        rows += [(f"compound_fuel:{ref_compound}", 0.0, 0.0)]
        rows += [(f"compound_fuel:{c}", comp_fuel[c], se[kf + j]) for j, c in enumerate(others)]
        rows += [("deg_fuel", deg_fuel, se[kf + len(others)]), ("race_laps", float(N), np.nan)]
    rows += [(f"offset:{d}", offsets[d], se[i]) for i, d in enumerate(drivers)]
    params = pd.DataFrame(rows, columns=["param", "value", "se"])
    params["n_laps"] = n
    params["method"] = method
    return FitResult(params=params, fuel_effect=fuel, offsets=offsets, deg=deg,
                     compound_offsets=comp_off, sigma=float(sigma), n_laps=n,
                     compound_fuel=comp_fuel, deg_fuel=deg_fuel, race_laps=N)


def apply_corrections(laps: pd.DataFrame, fit: FitResult | None, total_laps: int | None) -> pd.DataFrame:
    """Add fuel-corrected (to end-of-race fuel) and car+fuel-corrected lap times."""
    out = laps.copy()
    if fit is None:
        out["lap_time_fuel_corr_s"] = np.nan
        out["lap_time_car_fuel_corr_s"] = np.nan
        return out
    n_total = total_laps or int(out["lap_number"].max())
    out["lap_time_fuel_corr_s"] = out["lap_time_s"] - fit.fuel_effect * (n_total - out["lap_number"].astype(float))
    off = out["driver"].astype(str).map(fit.offsets).astype(float)
    out["lap_time_car_fuel_corr_s"] = out["lap_time_fuel_corr_s"] - off
    return out
