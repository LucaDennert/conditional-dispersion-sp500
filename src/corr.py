"""Implied and realized index correlation.

Both measures use the same "average correlation" formula: the single correlation rho that,
applied to every pair of constituents, reproduces the index variance from constituent
volatilities and weights:

    sigma_I^2 = sum_i w_i^2 sigma_i^2 + rho * sum_{i != j} w_i w_j sigma_i sigma_j

    rho = (sigma_I^2 - sum_i w_i^2 sigma_i^2) / ((sum_i w_i sigma_i)^2 - sum_i w_i^2 sigma_i^2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def average_correlation(sigma_index: float, sigma: np.ndarray, w: np.ndarray) -> float:
    """Average pairwise correlation implied by an index vol and constituent vols/weights.

    Weights are renormalized to sum to one over the names passed in.
    """
    sigma = np.asarray(sigma, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(sigma) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2 or not np.isfinite(sigma_index):
        return np.nan
    s, ww = sigma[ok], w[ok] / w[ok].sum()
    own = np.sum(ww**2 * s**2)
    cross = np.sum(ww * s) ** 2 - own
    return (sigma_index**2 - own) / cross if cross > 0 else np.nan


def implied_correlation_panel(
    iv_index: pd.Series,
    iv_members: pd.DataFrame,
    weights: pd.DataFrame,
    min_coverage: float = 0.90,
) -> pd.DataFrame:
    """IC for each date.

    iv_index:   date -> index implied vol
    iv_members: long frame with columns [date, secid, iv]
    weights:    long frame with columns [date, secid, weight] (weights as of the date's
                previous month end, summing to ~1 over all members)
    Returns a frame indexed by date with columns [ic, coverage, n].
    A date is NaN when members with valid IV cover less than `min_coverage` of index weight.
    """
    m = iv_members.merge(weights, on=["date", "secid"], how="right")
    out = []
    for d, g in m.groupby("date", sort=True):
        valid = g["iv"].notna() & (g["iv"] > 0)
        cov = g.loc[valid, "weight"].sum() / g["weight"].sum()
        si = iv_index.get(d, np.nan)
        ic = average_correlation(si, g.loc[valid, "iv"].values, g.loc[valid, "weight"].values) if cov >= min_coverage else np.nan
        out.append((d, ic, cov, int(valid.sum())))
    return pd.DataFrame(out, columns=["date", "ic", "coverage", "n"]).set_index("date")


def realized_correlation(
    ret_index: pd.Series,
    ret_members: pd.DataFrame,
    weights: pd.Series,
    min_obs_frac: float = 0.8,
) -> float:
    """Average realized correlation over one window.

    ret_index:   daily log returns of the index over the window
    ret_members: dates x secid daily log returns over the same window
    weights:     secid -> weight (fixed over the window)
    Names with fewer than `min_obs_frac` valid returns are dropped (weights renormalized).
    """
    n = len(ret_index)
    keep = ret_members.notna().sum() >= min_obs_frac * n
    r = ret_members.loc[:, keep]
    w = weights.reindex(r.columns).fillna(0.0)
    if (w > 0).sum() < 2:
        return np.nan
    sig = r.std(ddof=1).values * np.sqrt(TRADING_DAYS)
    sig_i = ret_index.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return average_correlation(sig_i, sig, w.values)


def fisher_z(x):
    return np.arctanh(np.clip(x, -0.999, 0.999))


def inv_fisher_z(z):
    return np.tanh(z)
