"""Figures and descriptive decomposition for the listed-contract results (after the kill test).
Run: python -m src.kill_test_figures DATA_DIR RESULTS_DIR
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stats import ols_newey_west


def main(data: Path, res: Path):
    T = pickle.load(open(data / "tranches_obj.pkl", "rb"))
    out = pickle.load(open(data / "kill_test_out.pkl", "rb"))
    sig = pd.read_pickle(data / "signals_monthly.pkl")
    tc = pd.read_pickle(data / "tranche_costs.pkl")
    rows = []
    for t in T.entries():
        g = T.daily(t, cost_mult=0, legs=True)
        n = T.daily(t, legs=True)
        rows.append({"date": t, "gross": g.pnl.sum(), "net": n.pnl.sum(), "g_long": g.long.sum(),
                     "g_short": g.short.sum(), "n_long": n.long.sum(), "n_short": n.short.sum()})
    r = pd.DataFrame(rows).set_index("date").join(tc[["cost_rt"]]).join(sig[["cp", "vix", "ic"]])
    r["costs_total"] = r.gross - r.net
    r["hedge_costs"] = r.costs_total - r.cost_rt
    per = pd.cut(r.index.year, [1995, 2000, 2007, 2010, 2019, 2025],
                 labels=["1996-2000", "2001-2007", "2008-2010", "2011-2019", "2020-2025"])
    tab = r.groupby(per, observed=True)[["gross", "g_long", "g_short", "cost_rt", "hedge_costs", "net"]].mean()
    tab.loc["Full sample"] = r[["gross", "g_long", "g_short", "cost_rt", "hedge_costs", "net"]].mean()
    tab["tranches"] = list(r.groupby(per, observed=True).size()) + [len(r)]
    lines = ["## Unconditional listed-contract tranches: decomposition, vol pts per tranche\n" + tab.round(2).to_string()]
    # does the correlation premium predict tranche P&L?
    for lab, a, b in [("estimation", "1996", "2010"), ("test", "2011", "2025")]:
        x = r.loc[a:b].dropna(subset=["cp", "net"])
        ols = ols_newey_west(x.net.values, x[["cp"]].values, 3, names=["cp"])
        ols_g = ols_newey_west(x.gross.values, x[["cp"]].values, 3, names=["cp"])
        lines.append(f"## Tranche P&L on CP, {lab} ({len(x)} tranches)\nnet:\n{ols.summary()}\ngross:\n{ols_g.summary()}")
    txt = "\n\n".join(lines)
    (res / "listed_decomposition.txt").write_text(txt + "\n")
    print(txt)
    r.to_pickle(data / "tranche_decomposition.pkl")

    # figure 1: cumulative monthly P&L, both samples
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, sample in zip(axes, ["estimation", "test"]):
        bench = out["monthly"][(sample, "bench_unscaled")]
        for name, lab in [("Primary (k=0.5, VIX filter)", "Part 3 primary rule"),
                          ("P4 ridge (final trade)", "Part 4 ridge"), ("P4 gradient boosting", "Part 4 GBM")]:
            m = out["monthly"][(sample, name)]
            ax.plot(m.index.to_timestamp(), m.cumsum(), label=lab)
        rep = out["report"][sample]
        e = rep.loc["Primary (k=0.5, VIX filter)", "avg_vega"] / rep.loc["Unconditional (every month), unscaled", "avg_vega"]
        ax.plot(bench.index.to_timestamp(), (bench * e).cumsum(), color="grey", ls="--", label="Unconditional, vega-matched to primary")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{sample.capitalize()} sample")
        ax.set_ylabel("Cumulative P&L, vol pts per unit of vega budget")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "fig_killtest_cumulative.png", dpi=150)

    # figure 2: CP vs tranche net P&L
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lab, a, b, c in [("1996-2010", "1996", "2010", "C0"), ("2011-2025", "2011", "2025", "C3")]:
        x = r.loc[a:b]
        ax.scatter(x.cp, x.net, s=12, alpha=0.6, color=c, label=lab)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0.0937 + 0.5 * 0.0883, color="black", ls=":", lw=1, label="Primary threshold c + 0.5 RMSE")
    ax.set_xlabel("Correlation premium CP at entry (IC minus HAR forecast)")
    ax.set_ylabel("Tranche P&L net of costs (vol pts)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "fig_cp_vs_pnl.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
