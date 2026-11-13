"""Part 1: the correlation risk premium on the S&P 500, 1996-2025 (descriptive).

On the first trading day of each month t:
  IC_h   implied correlation from h-day ATM-forward IVs (SPX and all members), h in {30, 91}
  RC_h   realized correlation over the following 21 (for 30d) or 63 (for 91d) trading days
  CRP_h  = IC_h - RC_h   (ex-post correlation risk premium)
Weights are CRSP market-cap weights at the previous month end.

Usage: python -m src.premium DATA_DIR RESULTS_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.corr import average_correlation
from src.signals import _rc_window, entry_dates
from src.stats import ols_newey_west

HORIZONS = {30: 21, 91: 63}       # IV maturity (calendar days) -> realized window (trading days)
PERIODS = {"1996-2007": ("1996", "2007"), "2008-2009": ("2008", "2009"),
           "2010-2019": ("2010", "2019"), "2020-2025": ("2020", "2025"), "Full sample": ("1996", "2025")}


def build(data_dir: Path) -> pd.DataFrame:
    members = pd.read_pickle(data_dir / "members_daily.pkl")
    weights = pd.read_pickle(data_dir / "weights_monthly.pkl")
    spx = pd.read_pickle(data_dir / "spx_daily.pkl")
    ivm = pd.read_pickle(data_dir / "iv_members.pkl")
    dates = spx.index
    pos = {d: i for i, d in enumerate(dates)}
    R = members.pivot_table(index="date", columns="permno", values="logret").reindex(dates)
    permnos, Rv, ridx = R.columns.values, R.values, spx["spx_logret"].values
    ivp = {h: ivm.pivot_table(index="date", columns="secid", values=f"iv{h}") for h in HORIZONS}

    rows = []
    for t in entry_dates(spx):
        i = pos[t]
        w = weights[weights["month"] == t.to_period("M") - 1]
        if w.empty:
            continue
        wv = pd.Series(w["weight"].values, index=w["permno"].values).reindex(permnos).fillna(0.0).values
        ws = w.dropna(subset=["secid"]).set_index("secid")["weight"]
        rec = {"date": t, "vix": spx["vix"].iat[i]}
        for h, L in HORIZONS.items():
            ivs = ivp[h].loc[t].reindex(ws.index) if t in ivp[h].index else pd.Series(dtype=float)
            ok = ivs.notna() & (ivs > 0)
            cov = ws[ok].sum() / w["weight"].sum() if len(ivs) else 0.0
            rec[f"ic{h}"] = average_correlation(spx[f"iv{h}"].iat[i], ivs[ok].values, ws[ok].values) if cov >= 0.9 else np.nan
            rec[f"rc{h}"] = _rc_window(Rv[i + 1:i + 1 + L], ridx[i + 1:i + 1 + L], wv) if i + L < len(dates) else np.nan
            rec[f"crp{h}"] = rec[f"ic{h}"] - rec[f"rc{h}"]
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


def nw_mean(x: pd.Series, lags: int):
    r = ols_newey_west(x.values, np.empty((len(x), 0)), lags, names=[])
    return r.params[0], r.tstat[0]


def main(data_dir: str, res_dir: str) -> None:
    data, res = Path(data_dir), Path(res_dir)
    res.mkdir(parents=True, exist_ok=True)
    df = build(data)
    df.to_pickle(data / "premium_monthly.pkl")
    lines = []

    rows = []
    for name, (a, b) in PERIODS.items():
        for h, lags in [(30, 1), (91, 3)]:
            x = df.loc[a:b, [f"ic{h}", f"rc{h}", f"crp{h}"]].dropna()
            m, t = nw_mean(x[f"crp{h}"], lags)
            rows.append({"period": name, "maturity": f"{h}d", "months": len(x), "IC": x[f"ic{h}"].mean(),
                         "RC (after)": x[f"rc{h}"].mean(), "premium": m, "t-stat (NW)": t,
                         "share > 0": (x[f"crp{h}"] > 0).mean()})
    tab = pd.DataFrame(rows).set_index(["period", "maturity"])
    tab.round(3).to_csv(res / "premium_by_period.csv")
    lines.append("## Correlation risk premium by period (IC minus subsequently realized correlation)\n" + tab.round(3).to_string())

    # by volatility regime (VIX terciles over the full sample)
    q = pd.qcut(df["vix"], 3, labels=["low VIX", "mid VIX", "high VIX"])
    reg = df.groupby(q, observed=True)[["vix", "ic91", "rc91", "crp91", "crp30"]].mean()
    lines.append("## By VIX tercile (full sample)\n" + reg.round(3).to_string())
    reg.round(3).to_csv(res / "premium_by_vix.csv")

    # does the premium predict itself? premium vs IC level
    d = df[["ic91", "crp91"]].dropna()
    r = ols_newey_west(d["crp91"].values, d[["ic91"]].values, 3, names=["ic91"])
    lines.append("## CRP(3M) on IC(3M) level\n" + r.summary())

    (res / "premium_report.txt").write_text("\n\n".join(lines) + "\n")
    print("\n\n".join(lines))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(df.index, df["ic91"], label="Implied correlation, 3M (at t)")
    axes[0].plot(df.index, df["rc91"], label="Realized correlation, next 3M")
    axes[0].set_ylabel("Correlation")
    axes[0].legend(frameon=False)
    axes[0].set_title("S&P 500: implied vs subsequently realized correlation (monthly)")
    axes[1].bar(df.index, df["crp91"], width=25, color=np.where(df["crp91"] >= 0, "#3b6ea8", "#c0504d"))
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[1].set_ylabel("Premium (IC − RC)")
    fig.tight_layout()
    fig.savefig(res / "fig_correlation_premium.png", dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:3])
