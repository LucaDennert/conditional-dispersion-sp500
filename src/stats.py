"""Small statistics toolkit (numpy only): OLS with Newey-West errors, Sharpe, drawdown,
block-bootstrap test of a Sharpe-ratio difference, deflated Sharpe ratio."""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt, log, e

import numpy as np


@dataclass
class OLSResult:
    params: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    r2: float
    rmse: float
    resid: np.ndarray
    names: list

    def summary(self) -> str:
        rows = [f"{n:>12s} {p: .4f}  (se {s:.4f}, t {t: .2f})" for n, p, s, t in zip(self.names, self.params, self.se, self.tstat)]
        return "\n".join(rows + [f"R2 = {self.r2:.3f}   RMSE = {self.rmse:.4f}"])


def ols_newey_west(y: np.ndarray, X: np.ndarray, lags: int, names=None, add_const=True) -> OLSResult:
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    # Newey-West (Bartlett kernel) long-run covariance of X'u
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        wl = 1 - l / (lags + 1)
        G = Xu[l:].T @ Xu[:-l]
        S += wl * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (u @ u) / np.sum((y - y.mean()) ** 2)
    names = (["const"] if add_const else []) + list(names or [f"x{i}" for i in range(k - add_const)])
    return OLSResult(b, se, b / se, r2, float(np.sqrt(np.mean(u**2))), u, names)


def sharpe(monthly: np.ndarray) -> float:
    m = np.asarray(monthly, float)
    m = m[np.isfinite(m)]
    return float(np.mean(m) / np.std(m, ddof=1) * np.sqrt(12)) if len(m) > 2 and np.std(m) > 0 else np.nan


def max_drawdown(monthly: np.ndarray) -> float:
    """Max drawdown of the cumulative P&L (additive, in P&L units), returned as a positive number."""
    c = np.cumsum(np.nan_to_num(np.asarray(monthly, float)))
    peak = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float(np.max(peak - c))


def block_bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, block: int = 3, draws: int = 10_000, seed: int = 0):
    """One-sided p-value for H0: Sharpe(a) <= Sharpe(b), circular block bootstrap on paired months.

    Returns (observed difference, p-value)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    obs = sharpe(a) - sharpe(b)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    diffs = np.empty(draws)
    for k in range(draws):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        diffs[k] = sharpe(a[idx]) - sharpe(b[idx])
    # center the bootstrap distribution on zero (null of no difference)
    p = float(np.mean(diffs - obs >= obs))
    return obs, p


def _norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _norm_ppf(p):
    # Acklam's rational approximation, accurate to ~1e-9
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    if p < 0.02425:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -_norm_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated_sharpe(monthly: np.ndarray, n_trials: int, sr_trials_var: float) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): probability that the true
    (non-annualized) Sharpe exceeds the maximum expected under n_trials independent tries."""
    m = np.asarray(monthly, float)
    m = m[np.isfinite(m)]
    T = len(m)
    sr = np.mean(m) / np.std(m, ddof=1)
    z = (m - m.mean()) / m.std(ddof=0)
    skew, kurt = np.mean(z**3), np.mean(z**4)
    g = 0.5772156649
    sr0 = sqrt(sr_trials_var) * ((1 - g) * _norm_ppf(1 - 1 / n_trials) + g * _norm_ppf(1 - 1 / (n_trials * e)))
    return _norm_cdf((sr - sr0) * sqrt(T - 1) / sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))
