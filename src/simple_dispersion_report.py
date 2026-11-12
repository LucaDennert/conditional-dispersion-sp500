"""Statistics, factor tests and figures for the plain dispersion trade (descriptive, not pre-registered).

Usage: python -m src.simple_dispersion_report DATA_DIR RAW_DIR RESULTS_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.simple_dispersion import H, COSTS
from src.stats import ols_newey_west

NW = 2          # tranches overlap for two months
PERIODS = {"1996-2007": ("1996", "2007"), "2008-2009": ("2008", "2009"),
           "2010-2019": ("2010", "2019"), "2020-2025": ("2020", "2025"), "Full sample": ("1996", "2025")}


def sharpe_nonoverlap(x: pd.Series) -> float:
    """Average annualized Sharpe of the three non-overlapping quarterly sub-series."""
    out = []
    for k in range(3):
        s = x.iloc[k::3].dropna()
        if len(s) > 4 and s.std() > 0:
            out.append(s.mean() / s.std() * 2)      # 4 tranches a year -> sqrt(4)
    return float(np.mean(out)) if out else np.nan


def mean_t(x: pd.Series):
    r = ols_newey_west(x.values, np.empty((len(x), 0)), NW, names=[])
    return r.params[0], r.tstat[0]


def summary_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for name, (a, b) in PERIODS.items():
        x = df.loc[a:b, col].dropna()
        m, t = mean_t(x)
        rows.append({"period": name, "tranches": len(x), "mean (vol pts)": m, "t-stat (NW)": t,
                     "Sharpe (ann.)": sharpe_nonoverlap(x), "hit rate": (x > 0).mean(),
                     "worst": x.min(), "best": x.max(), "skew": x.skew()})
    return pd.DataFrame(rows).set_index("period")


def window_factors(df: pd.DataFrame, ff: pd.DataFrame, spx: pd.DataFrame) -> pd.DataFrame:
    """Sum of daily factor returns (in %) over each tranche's life; change in VIX."""
    out = []
    for t, row in df.iterrows():
        w = ff.loc[(ff.index > t) & (ff.index <= row["exit"])]
        rec = (w.sum() * 100).to_dict()
        rec["dVIX"] = spx["vix"].asof(row["exit"]) - spx["vix"].asof(t)
        out.append(rec)
    return pd.DataFrame(out, index=df.index)


def model_free_var_strike(vs: pd.DataFrame, fwd: float, r: float, T: float = 91 / 365) -> float:
    """Variance-swap strike from one date's 91-day smile (delta grid), VIX-style integral.

    Smile interpolated in log-moneyness, extrapolated flat, OTM option prices integrated over
    a fine strike grid: K_var = (2 e^{rT} / T) * integral Q(K) / K^2 dK.
    """
    vs = vs[vs["impl_strike"] > 0]
    otm = vs[((vs["cp_flag"] == "P") & (vs["impl_strike"] <= fwd)) | ((vs["cp_flag"] == "C") & (vs["impl_strike"] > fwd))]
    otm = otm.sort_values("impl_strike")
    if len(otm) < 6:
        return np.nan
    k_pts = np.log(otm["impl_strike"].values / fwd)
    iv_pts = otm["impl_volatility"].values
    k = np.linspace(-2.0, 1.0, 3001)
    iv = np.interp(k, k_pts, iv_pts)
    K = fwd * np.exp(k)
    sq = iv * np.sqrt(T)
    d1 = (np.log(fwd / K) + 0.5 * sq**2) / sq
    d2 = d1 - sq
    df = np.exp(-r * T)
    call = df * (fwd * ndtr(d1) - K * ndtr(d2))
    put = df * (K * ndtr(-d2) - fwd * ndtr(-d1))
    Q = np.where(K < fwd, put, call)
    return float(2 * np.exp(r * T) / T * np.trapezoid(Q / K**2, K))


def main(data_dir: str, raw_dir: str, res_dir: str) -> None:
    data, raw, res = Path(data_dir), Path(raw_dir), Path(res_dir)
    res.mkdir(parents=True, exist_ok=True)
    top = pd.read_pickle(data / "simple_dispersion_top50.pkl")
    allm = pd.read_pickle(data / "simple_dispersion_all.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    z = pd.read_pickle(data / "zero_curve.pkl")
    ff = pd.read_csv(raw / "ff5_umd_daily.csv", parse_dates=["date"]).set_index("date")
    ff = ff[["mktrf", "smb", "hml", "rmw", "cma", "umd"]]
    lines = []

    # 1. performance
    for label, df, col in [("TOP-50 basket, net of costs", top, "net"), ("TOP-50 basket, gross", top, "gross"),
                           ("All members, net of costs", allm, "net"),
                           ("Short SPX straddle leg alone (gross)", top, "short_leg"),
                           ("Long single-stock leg alone (gross)", top, "long_leg")]:
        tab = summary_table(df, col)
        lines.append(f"## {label}\n{tab.round(2).to_string()}")
        if label.startswith("TOP-50 basket, net"):
            tab.round(3).to_csv(res / "simple_dispersion_performance.csv")
    # cost sensitivity
    cs = []
    for mult in [0, 0.5, 1, 2, 3]:
        x = top["gross"] - mult * (COSTS["single"] + COSTS["index"])
        cs.append({"cost multiple": mult, "mean": x.mean(), "Sharpe": sharpe_nonoverlap(x)})
    lines.append("## Cost sensitivity (TOP-50; 1x = 0.50 single + 0.15 index vol pts half-spread)\n" + pd.DataFrame(cs).round(2).to_string(index=False))
    lines.append(f"## Exact delta-hedged P&L vs 'vega x (RV-IV)' approximation: corr {top['gross'].corr(top['approx_gross']):.3f}, "
                 f"mean gap {(top['gross'] - top['approx_gross']).mean():.2f} vol pts")

    # 2. factor regressions
    F = window_factors(top, ff, spx)
    y = top["net"]
    specs = {"CAPM": ["mktrf"], "FF5 + momentum": ["mktrf", "smb", "hml", "rmw", "cma", "umd"],
             "FF5 + momentum + dVIX": ["mktrf", "smb", "hml", "rmw", "cma", "umd", "dVIX"],
             "Short SPX vol only": ["short_leg"],
             "All": ["mktrf", "smb", "hml", "rmw", "cma", "umd", "dVIX", "short_leg"]}
    X_all = F.join(top[["short_leg"]])
    reg_rows = []
    for name, cols in specs.items():
        d = pd.concat([y, X_all[cols]], axis=1).dropna()
        r = ols_newey_west(d.iloc[:, 0].values, d[cols].values, NW, names=cols)
        lines.append(f"## Factor regression: {name} (dependent: net P&L per tranche, vol pts; factors summed over the tranche, %)\n{r.summary()}")
        reg_rows.append({"model": name, "alpha (vol pts/tranche)": r.params[0], "t(alpha)": r.tstat[0], "R2": r.r2,
                         **{f"beta {c}": b for c, b in zip(cols, r.params[1:])}, **{f"t {c}": t for c, t in zip(cols, r.tstat[1:])}})
    pd.DataFrame(reg_rows).round(3).to_csv(res / "simple_dispersion_factors.csv", index=False)

    # 3. index skew premium: short SPX variance swap at model-free strike vs at ATM vol
    vs = pd.read_csv(raw / "optionm_vsurfd_spx_91d.csv.gz", parse_dates=["date"])
    vs_by = dict(tuple(vs.groupby("date")))
    rec = []
    pos = {d: i for i, d in enumerate(spx.index)}
    for t in top.index:
        if t not in vs_by:
            continue
        i = pos[t]
        row = z.loc[:t].iloc[-1].dropna()
        r = float(np.interp(91, row.index.values.astype(float), row.values))
        kvar = model_free_var_strike(vs_by[t], spx["fwd91"].iat[i], r)
        rv2 = np.var(spx["spx_logret"].iloc[i + 1:i + H + 1], ddof=0) * 252
        atm = spx["iv91"].iat[i]
        rec.append({"date": t, "vol_strike_mf": np.sqrt(kvar), "atm": atm,
                    "short_var_mf": (kvar - rv2) / (2 * np.sqrt(kvar)) * 100,
                    "short_var_atm": (atm**2 - rv2) / (2 * atm) * 100})
    sk = pd.DataFrame(rec).set_index("date")
    sk.to_pickle(data / "spx_var_strike.pkl")
    lines.append("## SPX 3M variance-swap strike (model-free) vs ATM vol, vol pts\n"
                 f"mean model-free strike {100 * sk['vol_strike_mf'].mean():.2f}, mean ATM {100 * sk['atm'].mean():.2f}, "
                 f"mean gap {100 * (sk['vol_strike_mf'] - sk['atm']).mean():.2f}\n"
                 f"short SPX var swap P&L per unit vega: at model-free strike {sk['short_var_mf'].mean():.2f}, "
                 f"at ATM strike {sk['short_var_atm'].mean():.2f} vol pts per tranche")

    (res / "simple_dispersion_report.txt").write_text("\n\n".join(lines) + "\n")
    print("\n\n".join(lines))

    # 4. figures
    ex = top.set_index("exit")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, lab in [("net", "Dispersion (net)"), ("short_leg", "Short SPX straddles (gross)"), ("long_leg", "Long single-stock straddles (gross)")]:
        ax.plot(ex.index, ex[col].cumsum() / 3, label=lab)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Monthly ladder of 3-month vega-neutral dispersion tranches (each 1/3 of vega)")
    ax.set_ylabel("Cumulative P&L, vol points per unit vega")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(res / "fig_simple_dispersion_cumulative.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(top["short_leg"], top["net"], s=10, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Short SPX straddle P&L (vol pts)")
    ax.set_ylabel("Dispersion net P&L (vol pts)")
    ax.set_title("Dispersion vs short index vol, per tranche")
    fig.tight_layout()
    fig.savefig(res / "fig_dispersion_vs_short_vol.png", dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:4])
