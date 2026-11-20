"""The basic dispersion trade on listed contracts: returns, costs, risks, tails and stress episodes.

Main specification: every month, short the SPX straddle and buy straddles on the 20 largest members
(vega-neutral, vega split by index weight), delta-hedged daily, held to expiry. Alternatives: top-50
basket, exit 10 days before expiry, correlation-weighted long leg (x sqrt(IC)).
Run: python -m src.basic_trade DATA_DIR RAW_DIR RESULTS_DIR
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ladder import ladder_daily, to_monthly
from src.stats import block_bootstrap_ci, max_drawdown, ols_newey_west, sharpe

PERIODS = {"1996-2000": ("1996", "2000"), "2001-2007": ("2001", "2007"), "2008-2010": ("2008", "2010"),
           "2011-2019": ("2011", "2019"), "2020-2025": ("2020", "2025"), "Full sample": ("1996", "2025")}
STRESS = {
    "LTCM / Russia (Aug-Oct 1998)": ("1998-08-01", "1998-10-31"),
    "9/11 (Sep 2001)": ("2001-09-01", "2001-10-31"),
    "2002 bear market (Jul-Oct 2002)": ("2002-07-01", "2002-10-31"),
    "Lehman (Sep-Nov 2008)": ("2008-09-01", "2008-11-30"),
    "Flash crash (May 2010)": ("2010-05-01", "2010-06-30"),
    "US downgrade (Aug 2011)": ("2011-08-01", "2011-09-30"),
    "China deval. (Aug 2015)": ("2015-08-01", "2015-09-30"),
    "Volmageddon (Feb 2018)": ("2018-02-01", "2018-03-31"),
    "Covid (Feb-Mar 2020)": ("2020-02-15", "2020-04-15"),
    "Yen carry unwind (Aug 2024)": ("2024-07-15", "2024-08-31"),
}


def tranche_table(T, sig, **kw):
    rows = []
    for t in T.entries():
        g = T.daily(t, cost_mult=0, legs=True, **kw)
        n = T.daily(t, legs=True, **kw)
        E = T.E[t]
        top = kw.get("top", 50)
        keep = (E["rank"] >= 0) & (E["rank"] < top)
        w = np.where(keep, E["w"], 0.0)
        w = w / w.sum() * kw.get("long_mult", 1.0)
        hs0 = np.nan_to_num(E["hs"][:, 0])
        entry_spread = float(w @ hs0 + hs0[0])
        rows.append({"date": t, "exit": g.index[-1], "gross": g.pnl.sum(), "net": n.pnl.sum(),
                     "long_g": g.long.sum(), "short_g": g.short.sum(), "long_n": n.long.sum(), "short_n": n.short.sum(),
                     "entry_spread": entry_spread, "mtm_min": n.pnl.cumsum().min()})
    r = pd.DataFrame(rows).set_index("date")
    r["other_costs"] = r.gross - r.net - r.entry_spread     # exit spread (if any) + hedging
    return r


def sharpe_tranche(x):
    """Annualized Sharpe from overlapping 3-month tranches: average over the 3 non-overlapping sub-series."""
    out = []
    for k in range(3):
        s = np.asarray(x)[k::3]
        if len(s) > 4 and s.std() > 0:
            out.append(s.mean() / s.std() * 2)
    return float(np.mean(out)) if out else np.nan


def perf_row(x, label):
    x = pd.Series(x).dropna()
    lo, hi = block_bootstrap_ci(x.values, np.mean, block=3)
    slo, shi = block_bootstrap_ci(x.values, sharpe_tranche, block=3)
    return {"": label, "tranches": len(x), "mean": x.mean(), "ci_lo": lo, "ci_hi": hi,
            "sharpe": sharpe_tranche(x.values), "sh_lo": slo, "sh_hi": shi, "hit": (x > 0).mean(),
            "worst": x.min(), "best": x.max()}


def tail_row(x, label, monthly):
    x = pd.Series(x).dropna()
    q5, q1 = x.quantile(0.05), x.quantile(0.01)
    return {"": label, "mean": x.mean(), "sd": x.std(), "skew": x.skew(), "kurt": x.kurt(),
            "VaR5": -q5, "CVaR5": -x[x <= q5].mean(), "CVaR1": -x[x <= q1].mean(), "worst": x.min(),
            "worst_month": monthly.min(), "maxDD": max_drawdown(monthly.values),
            "mean/CVaR5": x.mean() / -x[x <= q5].mean()}


def main(data: Path, raw: Path, res: Path):
    T = pickle.load(open(data / "tranches_expiry.pkl", "rb"))
    T10 = pickle.load(open(data / "tranches_obj.pkl", "rb"))
    sig = pd.read_pickle(data / "signals_monthly.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    cal = spx.index
    sqrt_ic = lambda t: float(np.sqrt(max(sig.ic.get(t, np.nan), 0.0)))
    ff = pd.read_csv(raw / "ff5_umd_daily.csv", parse_dates=["date"]).set_index("date")[["mktrf", "smb", "hml", "rmw", "cma", "umd"]]

    base = tranche_table(T, sig, top=20)
    base.to_pickle(data / "basic_top20_expiry.pkl")
    variants = {
        "Top-20, held to expiry (main)": base,
        "Top-50, held to expiry": tranche_table(T, sig, top=50),
        "Top-20, exit 10 days before expiry": tranche_table(T10, sig, top=20),
        "Top-50, exit 10 days before expiry": tranche_table(T10, sig, top=50),
    }
    rows_sq = []
    for t in T.entries():
        n = T.daily(t, top=20, long_mult=sqrt_ic(t), legs=True)
        g = T.daily(t, top=20, long_mult=sqrt_ic(t), cost_mult=0, legs=True)
        rows_sq.append({"date": t, "gross": g.pnl.sum(), "net": n.pnl.sum(), "long_n": n.long.sum(), "short_n": n.short.sum()})
    sq = pd.DataFrame(rows_sq).set_index("date")
    variants["Top-20, held to expiry, long leg x sqrt(IC)"] = sq
    lines = []

    # 1. performance by period, gross and net, with CIs
    perf = []
    for name, (a, b) in PERIODS.items():
        x = base.loc[a:b]
        r1 = perf_row(x.gross, name); r2 = perf_row(x.net, name)
        perf.append({"period": name, "tranches": r1["tranches"],
                     "gross": r1["mean"], "gross_lo": r1["ci_lo"], "gross_hi": r1["ci_hi"],
                     "net": r2["mean"], "net_lo": r2["ci_lo"], "net_hi": r2["ci_hi"],
                     "sharpe_net": r2["sharpe"], "sh_lo": r2["sh_lo"], "sh_hi": r2["sh_hi"],
                     "hit_net": r2["hit"], "worst_net": r2["worst"],
                     "short_leg_g": x.short_g.mean(), "long_leg_g": x.long_g.mean(),
                     "entry_spread": x.entry_spread.mean(), "hedging": x.other_costs.mean()})
    perf = pd.DataFrame(perf).set_index("period")
    lines.append("## Main trade (top-20, held to expiry): per tranche, vol pts per unit vega; 95% block-bootstrap CIs\n" + perf.round(2).to_string())

    # 2. implementation comparison (full sample)
    comp = []
    for name, df in variants.items():
        r = perf_row(df.net, name)
        comp.append({"variant": name, "gross": df.gross.mean(), "costs": (df.gross - df.net).mean(), "net": df.net.mean(),
                     "net_lo": r["ci_lo"], "net_hi": r["ci_hi"], "sharpe_net": r["sharpe"],
                     "net_1996_2010": df.loc[:"2010", "net"].mean(), "net_2011_2025": df.loc["2011":, "net"].mean()})
    comp = pd.DataFrame(comp).set_index("variant")
    lines.append("## Implementation choices (full sample, per tranche)\n" + comp.round(2).to_string())

    # 3. tails: short SPX straddle alone, long basket alone, dispersion, sqrt(IC) dispersion (net)
    D, _, _ = ladder_daily(T, None, "1996-01-01", "2025-08-29", cal, top=20)
    Dq, _, _ = ladder_daily(T, None, "1996-01-01", "2025-08-29", cal, top=20, long_mult=sqrt_ic)
    M, Mq = to_monthly(D), to_monthly(Dq)
    tails = pd.DataFrame([
        tail_row(base.short_n, "Short SPX straddle alone", M["short"]),
        tail_row(base.long_n, "Long top-20 straddles alone", M["long"]),
        tail_row(base.net, "Dispersion (main)", M["pnl"]),
        tail_row(sq.net, "Dispersion, long leg x sqrt(IC)", Mq["pnl"]),
    ]).set_index("")
    lines.append("## Tail risk, net, per tranche (vol pts per unit vega); worst month and max drawdown of the ladder (per unit vega budget)\n"
                 + tails.round(2).to_string())
    worst = base.nsmallest(8, "net")[["net", "short_n", "long_n", "mtm_min"]]
    worst["VIX at entry"] = sig.vix.reindex(worst.index)
    lines.append("## Worst tranches (main trade)\n" + worst.round(2).to_string())

    # 4. stress episodes (ladder, net, cumulative over the window)
    st = []
    for name, (a, b) in STRESS.items():
        w = D.loc[a:b]; wq = Dq.loc[a:b]
        st.append({"episode": name, "short SPX": w["short"].sum(), "long basket": w["long"].sum(),
                   "dispersion": w["pnl"].sum(), "dispersion sqrt(IC)": wq["pnl"].sum(),
                   "VIX peak": spx.loc[a:b, "vix"].max()})
    st = pd.DataFrame(st).set_index("episode")
    lines.append("## Stress episodes: cumulative ladder P&L in the window (vol pts per unit vega budget)\n" + st.round(2).to_string())

    # 5. factor regressions (net tranche P&L on factors summed over the tranche life)
    fac = []
    for t, row in base.iterrows():
        w = ff.loc[(ff.index > t) & (ff.index <= row["exit"])]
        rec = (w.sum() * 100).to_dict()
        rec["dVIX"] = spx["vix"].asof(row["exit"]) - spx["vix"].asof(t)
        fac.append(rec)
    F = pd.DataFrame(fac, index=base.index)
    specs = {"CAPM": ["mktrf"], "FF5 + momentum": ["mktrf", "smb", "hml", "rmw", "cma", "umd"],
             "FF5 + momentum + dVIX": ["mktrf", "smb", "hml", "rmw", "cma", "umd", "dVIX"]}
    freg = []
    for name, cols in specs.items():
        X = F[cols]; ok = X.notna().all(axis=1) & base.net.notna()
        o = ols_newey_west(base.net[ok].values, X[ok].values, 2, names=cols)
        freg.append({"model": name, "alpha": o.params[0], "t_alpha": o.tstat[0], "beta_mkt": o.params[1],
                     "t_mkt": o.tstat[1], "R2": o.r2})
    o = ols_newey_west(base.net.values, base[["short_n"]].values, 2, names=["short"])
    freg.append({"model": "Short SPX straddle only", "alpha": o.params[0], "t_alpha": o.tstat[0],
                 "beta_mkt": np.nan, "t_mkt": np.nan, "R2": o.r2})
    freg = pd.DataFrame(freg).set_index("model")
    lines.append("## Factor regressions, net tranche P&L (NW 2 lags)\n" + freg.round(3).to_string())

    # monthly ladder stats
    lad = {}
    for name, m in [("main", M["pnl"]), ("sqrt_ic", Mq["pnl"]), ("short_spx", M["short"])]:
        lo, hi = block_bootstrap_ci(m.values, sharpe, block=3)
        lad[name] = {"mean_month": round(float(m.mean()), 3), "sharpe": round(sharpe(m.values), 3),
                     "sharpe_ci": [round(lo, 2), round(hi, 2)], "maxDD": round(max_drawdown(m.values), 2)}
    lines.append("## Monthly ladder (per unit vega budget)\n" + json.dumps(lad, indent=1))

    txt = "\n\n".join(lines)
    (res / "basic_trade_report.txt").write_text(txt + "\n")
    print(txt)
    pickle.dump({"perf": perf, "comp": comp, "tails": tails, "worst": worst, "stress": st, "freg": freg,
                 "ladder": lad, "M": M, "Mq": Mq, "D": D, "sq": sq, "variants": variants},
                open(data / "basic_trade_out.pkl", "wb"))

    # figures
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(M.index.to_timestamp(), M["pnl"].cumsum(), color="C0", lw=2, label="Dispersion, net")
    ax.plot(M.index.to_timestamp(), M["short"].cumsum(), color="C3", lw=1, label="Short SPX straddle leg, net")
    ax.plot(M.index.to_timestamp(), M["long"].cumsum(), color="C2", lw=1, label="Long top-20 straddle leg, net")
    ax.plot(Mq.index.to_timestamp(), Mq["pnl"].cumsum(), color="C1", lw=1.5, ls="--", label="Dispersion, long leg x sqrt(IC), net")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative P&L, vol pts per unit of vega budget")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(res / "fig_basic_cumulative.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-25, 25, 51)
    ax.hist(base.short_n.clip(-25, 25), bins=bins, alpha=0.55, color="C3", label="Short SPX straddle alone")
    ax.hist(base.net.clip(-25, 25), bins=bins, alpha=0.55, color="C0", label="Dispersion")
    ax.set_xlabel("P&L per tranche, net (vol pts per unit vega); clipped at +/-25")
    ax.set_ylabel("Number of tranches")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(res / "fig_basic_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    yy = np.arange(len(st))
    ax.barh(yy - 0.2, st["short SPX"], height=0.4, color="C3", label="Short SPX straddle alone")
    ax.barh(yy + 0.2, st["dispersion"], height=0.4, color="C0", label="Dispersion")
    ax.set_yticks(yy); ax.set_yticklabels(st.index, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Cumulative P&L over the episode, vol pts per unit vega budget")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(res / "fig_stress.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
