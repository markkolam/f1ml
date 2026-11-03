"""Empirical-Bayes partial pooling (normal-normal random effects).

Used everywhere a quantity is estimated per race and we want per-track values
that borrow strength from the rest of the calendar: a track raced once is
pulled toward its era mean, a track raced four times mostly speaks for itself.

`dl_pool`  : DerSimonian-Laird random-effects mean, its SE and between-group variance tau^2.
`shrink`   : posterior mean/SD of each group toward a parent mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dl_pool(y, se) -> tuple[float, float, float]:
    """Random-effects pooled mean, SE of that mean, and between-estimate variance tau^2."""
    y = np.asarray(y, float)
    v = np.asarray(se, float) ** 2
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    if len(y) == 0:
        return np.nan, np.nan, np.nan
    if len(y) == 1:
        return float(y[0]), float(np.sqrt(v[0])), 0.0
    w = 1 / v
    mu_fe = np.sum(w * y) / w.sum()
    q = np.sum(w * (y - mu_fe) ** 2)
    c = w.sum() - np.sum(w**2) / w.sum()
    tau2 = max(0.0, (q - (len(y) - 1)) / c) if c > 0 else 0.0
    w_re = 1 / (v + tau2)
    mu = np.sum(w_re * y) / w_re.sum()
    return float(mu), float(np.sqrt(1 / w_re.sum())), float(tau2)


def shrink(group_mean, group_var, parent_mean, parent_tau2) -> tuple[np.ndarray, np.ndarray]:
    """Posterior of each group's true value: precision-weighted blend of its own estimate and the parent."""
    gm = np.asarray(group_mean, float)
    gv = np.asarray(group_var, float)
    if not np.isfinite(parent_tau2) or parent_tau2 <= 0:
        return np.full_like(gm, parent_mean), np.full_like(gm, np.sqrt(np.nanmean(gv)) if len(gv) else np.nan)
    prec = 1 / gv + 1 / parent_tau2
    post = (gm / gv + parent_mean / parent_tau2) / prec
    return post, np.sqrt(1 / prec)


def within_variance(df: pd.DataFrame, value: str, se: str, group: list[str]) -> float:
    """Typical extra variance between repeat estimates of the same group (e.g. the same
    track in different years) beyond their own SEs, from groups with >= 2 estimates."""
    t2 = [dl_pool(g[value], g[se])[2] for _, g in df.groupby(group) if g[value].notna().sum() >= 2]
    return float(np.median(t2)) if t2 else 0.0  # median: robust to the odd freak race


def hierarchical(df: pd.DataFrame, value: str, se: str, group: list[str], parent: list[str],
                 min_se: float = 1e-4, add_within: bool = True) -> pd.DataFrame:
    """Two-level pooling: estimates -> group (e.g. track) -> parent (e.g. era x compound).

    A single race's SE only covers lap-to-lap noise; the same track also varies
    year to year (weather, surface, setups). With `add_within`, that variance,
    learnt from groups seen more than once, is added to every estimate's SE so a
    group seen once is shrunk appropriately. Returns one row per group with the raw
    pooled estimate, the parent mean and the shrunk posterior mean / SD.
    """
    d = df.dropna(subset=[value, se]).copy()
    d[se] = d[se].clip(lower=min_se)
    if add_within:
        omega2 = within_variance(d, value, se, group)
        d[se] = np.sqrt(d[se] ** 2 + omega2)
    rows = []
    for gkey, g in d.groupby(group):
        m, s, t2 = dl_pool(g[value], g[se])
        rows.append({**dict(zip(group, gkey if isinstance(gkey, tuple) else (gkey,))),
                     "raw_mean": m, "raw_se": s, "within_tau2": t2, "n": len(g)})
    G = pd.DataFrame(rows)
    out = []
    for pkey, p in G.groupby(parent):
        mu, mu_se, tau2 = dl_pool(p["raw_mean"], p["raw_se"])
        post, post_sd = shrink(p["raw_mean"], p["raw_se"] ** 2, mu, tau2)
        out.append(p.assign(parent_mean=mu, parent_tau2=tau2, post_mean=post, post_sd=post_sd))
    return pd.concat(out, ignore_index=True) if out else G
