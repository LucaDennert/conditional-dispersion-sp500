"""Monthly signal panel (PREREGISTRATION.md, sections 4-5).

For every entry date t (first trading day of each month) computes:
  ic        implied correlation from 91-day ATM-forward IVs (SPX and all members)
  coverage  share of index weight with a valid member IV
  rc21, rc63, rc252   trailing realized correlation (windows ending on t)
  rc_fwd    realized correlation over the 53 trading days after t (forecast target)
  iv_spx91, rv_spx21, vp, vix
Weights are CRSP market-cap weights at the previous month end.

Usage: python -m src.signals DATA_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.corr import TRADING_DAYS, average_correlation

HOLD = 53            # trading days, forecast horizon
WINDOWS = (21, 63, 252)
MIN_COVERAGE = 0.90
MIN_OBS_FRAC = 0.8


def entry_dates(spx: pd.DataFrame) -> pd.DatetimeIndex:
    d = spx.index.to_series()
    return pd.DatetimeIndex(d.groupby(d.dt.to_period("M")).min().values)


def _rc_window(R: np.ndarray, r_idx: np.ndarray, w: np.ndarray) -> float:
    """R: T x N member log returns (NaN allowed), r_idx: T index returns, w: N weights."""
    T = len(r_idx)
    if T < 10:
        return np.nan
    ok = (np.isfinite(R).sum(axis=0) >= MIN_OBS_FRAC * T) & (w > 0)
    if ok.sum() < 2:
        return np.nan
    sig = np.nanstd(R[:, ok], axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
    sig_i = np.nanstd(r_idx, ddof=1) * np.sqrt(TRADING_DAYS)
    return average_correlation(sig_i, sig, w[ok])


def build_signals(data_dir: str | Path) -> pd.DataFrame:
    data = Path(data_dir)
    members = pd.read_pickle(data / "members_daily.pkl")
    weights = pd.read_pickle(data / "weights_monthly.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    ivm_path = data / "iv_members.pkl"
    ivm = pd.read_pickle(ivm_path) if ivm_path.exists() else None

    dates = spx.index
    R = members.pivot_table(index="date", columns="permno", values="logret").reindex(dates)
    permnos = R.columns.values
    Rv = R.values
    ridx = spx["spx_logret"].values
    pos = {d: i for i, d in enumerate(dates)}

    if ivm is not None:
        iv91 = ivm.pivot_table(index="date", columns="secid", values="iv91")

    rows = []
    for t in entry_dates(spx):
        i = pos[t]
        prev_month = (t.to_period("M") - 1)
        w = weights[weights["month"] == prev_month]
        if w.empty:
            continue
        wv = pd.Series(w["weight"].values, index=w["permno"].values).reindex(permnos).fillna(0.0).values
        rec = {"date": t, "n_members": len(w)}
        for L in WINDOWS:
            rec[f"rc{L}"] = _rc_window(Rv[max(0, i - L + 1): i + 1], ridx[max(0, i - L + 1): i + 1], wv) if i + 1 >= L else np.nan
        rec["rc_fwd"] = _rc_window(Rv[i + 1: i + 1 + HOLD], ridx[i + 1: i + 1 + HOLD], wv) if i + HOLD < len(dates) else np.nan
        rec["iv_spx91"] = spx["iv91"].iat[i]
        rec["rv_spx21"] = np.std(ridx[i - 20: i + 1], ddof=1) * np.sqrt(TRADING_DAYS) if i >= 20 else np.nan
        rec["vix"] = spx["vix"].iat[i]
        if ivm is not None and t in iv91.index:
            s = iv91.loc[t]
            ws = w.dropna(subset=["secid"]).set_index("secid")["weight"]
            ivs = s.reindex(ws.index)
            valid = ivs.notna() & (ivs > 0)
            cov = ws[valid].sum() / w["weight"].sum()
            rec["coverage"] = cov
            rec["ic"] = average_correlation(rec["iv_spx91"], ivs[valid].values, ws[valid].values) if cov >= MIN_COVERAGE else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("date")
    out["vp"] = out["iv_spx91"] - out["rv_spx21"]
    return out


if __name__ == "__main__":
    sig = build_signals(sys.argv[1])
    sig.to_pickle(Path(sys.argv[1]) / "signals_monthly.pkl")
    print(sig.describe().T.round(3).to_string())
