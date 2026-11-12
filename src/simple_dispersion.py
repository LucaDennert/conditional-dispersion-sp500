"""Descriptive analysis: the plain (unconditional) dispersion trade, 1996-2025.

Not part of the pre-registered kill test. Every month, open a vega-neutral dispersion tranche:
short a 3-month ATM-forward straddle on SPX, long 3-month ATM-forward straddles on the members
(top 50 by market cap at the previous month end; all members as robustness), vega allocated by
index weight. Each straddle is delta-hedged daily at its inception implied vol and held to expiry.

P&L of each straddle is computed exactly on the realized price path:
    entry price   Black-Scholes at the market implied vol (IvyDB 91-day ATM-forward IV)
    hedge         daily Black-Scholes delta, short shares, dividends paid, cash financed at r
    exit          actual payoff |S_T - K| at expiry (63 trading days)
P&L is expressed in vol points per unit of initial vega, so legs are comparable and the trade
is vega-neutral by construction:  dispersion = sum_i w_i x_i - x_SPX.
Costs: half bid/ask spread paid at entry, in vol points (desk assumptions, see COSTS).

Usage: python -m src.simple_dispersion DATA_DIR RESULTS_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.signals import entry_dates

H = 63                      # trading days to expiry
T = H / 252.0               # option life in years (trading-day clock)
DT = 1 / 252.0
COSTS = {"index": 0.15, "single": 0.50}   # half-spread at entry, vol points
TOP_N = 50


def bs_straddle(S, K, tau, r, q, sig):
    """Value and delta of a straddle; arrays broadcast. tau may be 0 (returns payoff)."""
    tau = np.maximum(tau, 1e-10)
    sq = sig * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * tau) / sq
    d2 = d1 - sq
    dfq, dfr = np.exp(-q * tau), np.exp(-r * tau)
    call = S * dfq * ndtr(d1) - K * dfr * ndtr(d2)
    put = K * dfr * ndtr(-d2) - S * dfq * ndtr(-d1)
    delta = dfq * (2 * ndtr(d1) - 1)
    return call + put, delta


def straddle_vega(S, K, tau, r, q, sig):
    sq = sig * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * tau) / sq
    return 2 * S * np.exp(-q * tau) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(tau)


def hedged_pnl(retx: np.ndarray, ret: np.ndarray, sig: np.ndarray, r: float, q: np.ndarray):
    """Delta-hedged long straddle P&L per unit vega (vol points), held to expiry.

    retx, ret: names x H price and total returns over the option life.
    sig, q: per-name implied vol and dividend yield. Spot normalized to 1 at inception.
    """
    n = retx.shape[0]
    S = np.ones((n, H + 1))
    S[:, 1:] = np.cumprod(1 + retx, axis=1)
    K = np.exp((r - q) * T)                       # ATM-forward strike
    tau = T - np.arange(H + 1) * DT
    V, D = bs_straddle(S, K[:, None], tau[None, :], r, q[:, None], sig[:, None])
    V[:, -1] = np.abs(S[:, -1] - K)               # actual payoff at expiry
    hedge = -(D[:, :-1] * S[:, :-1] * ret).sum(axis=1)                   # short delta shares, total return
    financing = -(r * (V[:, :-1] - D[:, :-1] * S[:, :-1]) * DT).sum(axis=1)
    pnl = V[:, -1] - V[:, 0] + hedge + financing
    vega = straddle_vega(1.0, K, T, r, q, sig) / 100.0                    # per vol point
    approx = (np.sqrt(252) * np.std(np.log1p(retx), axis=1, ddof=1) - sig) * 100   # RV - IV, vol points
    return pnl / vega, approx


def rate_91(z: pd.DataFrame, d: pd.Timestamp) -> float:
    row = z.loc[:d].iloc[-1].dropna()
    return float(np.interp(91, row.index.values.astype(float), row.values))


def build(data_dir: str | Path, top_n: int | None = TOP_N) -> pd.DataFrame:
    data = Path(data_dir)
    members = pd.read_pickle(data / "members_daily.pkl")
    weights = pd.read_pickle(data / "weights_monthly.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    ivm = pd.read_pickle(data / "iv_members.pkl")
    z = pd.read_pickle(data / "zero_curve.pkl")

    dates = spx.index
    pos = {d: i for i, d in enumerate(dates)}
    Rx = members.pivot_table(index="date", columns="permno", values="retx").reindex(dates)
    Rt = members.pivot_table(index="date", columns="permno", values="ret").reindex(dates)
    div = (Rt - Rx).rolling(252, min_periods=200).sum()            # trailing dividend yield
    iv91 = ivm.pivot_table(index="date", columns="secid", values="iv91")
    spx_rx = spx["spx"].pct_change()

    rows = []
    for t in entry_dates(spx):
        i = pos[t]
        if i + H >= len(dates):
            break
        w = weights[weights["month"] == t.to_period("M") - 1].dropna(subset=["secid"])
        if w.empty or t not in iv91.index:
            continue
        w = w.assign(iv=iv91.loc[t].reindex(w["secid"]).values).dropna(subset=["iv"])
        w = w[w["iv"] > 0].sort_values("weight", ascending=False)
        if top_n:
            w = w.head(top_n)
        win = slice(i + 1, i + H + 1)
        rx = Rx.iloc[win].reindex(columns=w["permno"]).values.T
        rt = Rt.iloc[win].reindex(columns=w["permno"]).values.T
        ok = np.isfinite(rx).all(axis=1) & np.isfinite(rt).all(axis=1)     # full path required
        w, rx, rt = w[ok], rx[ok], rt[ok]
        r = rate_91(z, t)
        q = np.nan_to_num(div.iloc[i].reindex(w["permno"]).values, nan=0.0).clip(0, 0.15)
        x_m, a_m = hedged_pnl(rx, rt, w["iv"].values, r, q)
        ww = w["weight"].values / w["weight"].sum()

        # index leg: q implied by the 91-day forward
        s0, f91, iv_i = spx["spx"].iat[i], spx["fwd91"].iat[i], spx["iv91"].iat[i]
        q_i = r - np.log(f91 / s0) / (91 / 365)
        rx_i = spx_rx.iloc[win].values[None, :]
        rt_i = rx_i + q_i * DT                                               # dividends accrue at q
        x_i, a_i = hedged_pnl(rx_i, rt_i, np.array([iv_i]), r, np.array([q_i]))

        long_leg = float(ww @ x_m)
        short_leg = -float(x_i[0])
        rows.append({
            "date": t, "exit": dates[i + H], "n_names": int(ok.sum()), "coverage": float(w["weight"].sum()),
            "long_leg": long_leg, "short_leg": short_leg, "gross": long_leg + short_leg,
            "net": long_leg + short_leg - COSTS["single"] - COSTS["index"],
            "approx_gross": float(ww @ a_m) - float(a_i[0]),
            "iv_spx": iv_i, "iv_members_avg": float(ww @ w["iv"].values),
        })
    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    data_dir, res_dir = sys.argv[1], Path(sys.argv[2])
    res_dir.mkdir(parents=True, exist_ok=True)
    for name, n in [("top50", TOP_N), ("all", None)]:
        df = build(data_dir, n)
        df.to_pickle(Path(data_dir) / f"simple_dispersion_{name}.pkl")
        print(name, len(df), "tranches")
        print(df[["long_leg", "short_leg", "gross", "net", "approx_gross"]].describe().round(2).to_string())
