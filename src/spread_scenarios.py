"""Investor vs desk: results under different execution costs.

Scenario f = fraction of the quoted half-spread paid on each option trade:
  f = 1    price-taker (investor): buys at the ask, sells at the bid
  f = 0.5  execution halfway to mid (e.g. working orders, some internalization)
  f = 0.25 market-making desk crossing little of the spread
  f = 0    execution at mid (gross of option spreads)
Delta-hedging costs are always paid. For the proposed strategy the cost hurdle is recomputed with
the same f, so a cheaper trader also has a lower hurdle.
Run: python -m src.spread_scenarios DATA_DIR RESULTS_DIR
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
from src.richness import entry_costs
from src.stats import block_bootstrap_ci, max_drawdown, sharpe

FS = [1.0, 0.5, 0.25, 0.0]


def tranche_series(T, f, **kw):
    return pd.Series({t: T.daily(t, spread_mult=f, **kw).sum() for t in T.entries()})


def main(data: Path, res: Path):
    T = pickle.load(open(data / "tranches_expiry.pkl", "rb"))
    T10 = pickle.load(open(data / "tranches_obj.pkl", "rb"))
    sig = pd.read_pickle(data / "signals_monthly.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    fc = pd.read_pickle(data / "richness_signal.pkl")
    tc = pd.read_pickle(data / "tranche_costs.pkl")
    cal = spx.index
    sq = lambda t: float(np.sqrt(max(sig.ic.get(t, np.nan), 0.0)))
    one = lambda t: 1.0

    rows = []
    specs = {
        "Vega-neutral, top-20, to expiry": (T, dict(top=20)),
        "Correlation-weighted, top-20, to expiry": (T, dict(top=20, long_mult="sq")),
        "Vega-neutral, top-20, exit 10 days early": (T10, dict(top=20)),
    }
    for name, (TT, kw) in specs.items():
        for f in FS:
            vals = {}
            for t in TT.entries():
                k = dict(kw)
                if k.get("long_mult") == "sq":
                    k["long_mult"] = sq(t)
                vals[t] = TT.daily(t, spread_mult=f, **k).sum()
            x = pd.Series(vals)
            lo, hi = block_bootstrap_ci(x.values, np.mean, block=3)
            k2 = dict(kw); 
            if k2.get("long_mult") == "sq":
                k2["long_mult"] = sq
            D, X, _ = ladder_daily(TT, None, "1996-01-01", "2025-08-29", cal, spread_mult=f, **k2)
            m = to_monthly(D)["pnl"]
            rows.append({"strategy": name, "f": f, "mean": x.mean(), "ci_lo": lo, "ci_hi": hi,
                         "mean_2011_2025": x.loc["2011":].mean(), "sharpe": sharpe(m.values), "maxDD": max_drawdown(m.values)})
    # short SPX straddle alone
    for f in FS:
        x = pd.Series({t: T.daily(t, top=20, spread_mult=f, legs=True)["short"].sum() for t in T.entries()})
        lo, hi = block_bootstrap_ci(x.values, np.mean, block=3)
        D, X, _ = ladder_daily(T, None, "1996-01-01", "2025-08-29", cal, top=20, spread_mult=f)
        m = to_monthly(D)["short"]
        rows.append({"strategy": "Short SPX straddle alone", "f": f, "mean": x.mean(), "ci_lo": lo, "ci_hi": hi,
                     "mean_2011_2025": x.loc["2011":].mean(), "sharpe": sharpe(m.values), "maxDD": max_drawdown(m.values)})
    tab = pd.DataFrame(rows).set_index(["strategy", "f"])

    # proposed strategy with scenario-consistent hurdle (2000-2025 walk-forward window)
    conv = (2 * tc.sigma_I / tc.cross).reindex(fc.index)
    ent = entry_costs(T, 20, sq).reindex(fc.index)
    hedge = fc["exp_hedge_sqrt(IC)"]
    prow = []
    a, b = fc.index[0], "2025-08-29"
    for f in FS:
        c = (f * ent + hedge) / 100 * conv
        size = ((fc.richness - c) / fc.rmse).clip(lower=0, upper=1)
        D, X, n = ladder_daily(T, size, a, b, cal, top=20, long_mult=sq, spread_mult=f)
        Db, Xb, _ = ladder_daily(T, None, a, b, cal, top=20, long_mult=sq, spread_mult=f)
        m, mb = to_monthly(D)["pnl"], to_monthly(Db)["pnl"]
        lo, hi = block_bootstrap_ci(m.values, sharpe, block=3)
        prow.append({"f": f, "avg_vega": float(X.mean()), "mean_month": float(m.mean()), "per_unit_vega": float(m.mean() / X.mean()),
                     "sharpe": sharpe(m.values), "sh_lo": lo, "sh_hi": hi, "maxDD": max_drawdown(m.values),
                     "always_mean_month": float(mb.mean()), "always_sharpe": sharpe(mb.values), "always_maxDD": max_drawdown(mb.values),
                     "share_traded": float((size > 0).mean()), "hurdle_mean": float(c.mean())})
    prop = pd.DataFrame(prow).set_index("f")
    txt = ("## Per-tranche net P&L (vol pts per unit vega) by fraction f of the half-spread paid; full sample 1996-2025; ladder Sharpe and max DD\n"
           + tab.round(3).to_string()
           + "\n\n## Proposed strategy (2000-2025) with hurdle recomputed for each f\n" + prop.round(3).to_string())
    (res / "spread_scenarios_report.txt").write_text(txt + "\n")
    pickle.dump({"tab": tab, "prop": prop}, open(data / "spread_scenarios_out.pkl", "wb"))
    print(txt)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in [("Short SPX straddle alone", "C3"), ("Correlation-weighted, top-20, to expiry", "C0"),
                    ("Vega-neutral, top-20, to expiry", "C2"), ("Vega-neutral, top-20, exit 10 days early", "C7")]:
        d = tab.loc[name]
        ax.plot([1 - f for f in d.index], d["mean"], marker="o", color=c, label=name)
        ax.fill_between([1 - f for f in d.index], d["ci_lo"], d["ci_hi"], color=c, alpha=0.12)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks([0, 0.5, 0.75, 1.0]); ax.set_xticklabels(["Investor\n(pays full spread)", "Pays half", "Pays a quarter", "Desk at mid"], fontsize=8)
    ax.set_ylabel("Net P&L per tranche (vol pts per unit vega)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(res / "fig_spread_scenarios.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
