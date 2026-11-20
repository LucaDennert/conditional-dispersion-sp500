"""Synthetic short correlation swap on the S&P 500.

Each month: sell a 3-month correlation swap struck at K = IC(3M, from ATM options) - h, where h is the
dealer margin (correlation points). Payoff per unit notional = K - RC, with RC the average correlation
realized over the next 63 trading days (same formula as IC, realized vols, cap weights, all members).
Conditional rule: sell only when walk-forward richness (IC - HAR forecast, src.richness) exceeds h.
Run: python -m src.corr_swap DATA_DIR RESULTS_DIR
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

from src.stats import block_bootstrap_ci, ols_newey_west

MARGINS = [0.0, 0.02, 0.04, 0.06]
PERIODS = {"1996-2007": ("1996", "2007"), "2008-2009": ("2008", "2009"), "2010-2019": ("2010", "2019"),
           "2020-2025": ("2020", "2025"), "Full sample": ("1996", "2025")}


def sharpe_q(x):
    """Annualized Sharpe of 3-month payoffs: average over the three non-overlapping sub-series."""
    x = np.asarray(x, float)
    out = [x[k::3].mean() / x[k::3].std() * 2 for k in range(3) if len(x[k::3]) > 4 and x[k::3].std() > 0]
    return float(np.mean(out)) if out else np.nan


def stats(x):
    x = pd.Series(x).dropna()
    lo, hi = block_bootstrap_ci(x.values, np.mean, block=3)
    q5 = x.quantile(0.05)
    return {"months": len(x), "mean": x.mean(), "ci_lo": lo, "ci_hi": hi, "loss_share": (x < 0).mean(),
            "CVaR5": -x[x <= q5].mean(), "worst": x.min(), "sharpe": sharpe_q(x.values)}


def main(data: Path, res: Path):
    p = pd.read_pickle(data / "premium_monthly.pkl")
    fc = pd.read_pickle(data / "richness_signal.pkl")
    tc = pd.read_pickle(data / "tranche_costs.pkl")
    prem = p.crp91.dropna()                          # IC - RC, h = 0
    lines = []

    # A. unconditional, by period
    A = []
    for name, (a, b) in PERIODS.items():
        x = prem.loc[a:b]
        s0 = stats(x); s4 = stats(x - 0.04)
        A.append({"period": name, "months": s0["months"], "break_even_margin": s0["mean"], "ci_lo": s0["ci_lo"],
                  "ci_hi": s0["ci_hi"], "loss_share_h0": s0["loss_share"], "CVaR5_h0": s0["CVaR5"], "worst_h0": s0["worst"],
                  "sharpe_h0": s0["sharpe"], "sharpe_h4": s4["sharpe"]})
    A = pd.DataFrame(A).set_index("period")
    lines.append("## A. Short 3M correlation swap at IC - h, payoff K - RC in correlation points; unconditional\n" + A.round(3).to_string())

    # B. always vs richness rule, by margin (walk-forward window)
    j = fc[["richness"]].join(prem.rename("prem")).dropna()
    B = []
    for h in MARGINS:
        for lab, x in [("always", j.prem - h), ("richness > h", (j.prem - h)[j.richness > h])]:
            s = stats(x)
            B.append({"margin_h": h, "rule": lab, **s, "share_traded": len(x) / len(j)})
    B = pd.DataFrame(B).set_index(["margin_h", "rule"])
    lines.append(f"## B. Always vs richness rule, {j.index[0].date()} to {j.index[-1].date()}\n" + B.round(3).to_string())

    # C. regimes at entry (h = 0)
    jj = j.join(p[["vix", "ic91"]])
    C = {}
    for lab, key in [("VIX tercile", pd.qcut(jj.vix, 3, labels=["low", "mid", "high"])),
                     ("IC tercile", pd.qcut(jj.ic91, 3, labels=["low", "mid", "high"])),
                     ("Richness quintile", pd.qcut(jj.richness, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]))]:
        g = jj.groupby(key, observed=True).prem
        C[lab] = pd.DataFrame({"mean": g.mean(), "loss_share": g.apply(lambda s: (s < 0).mean()),
                               "worst": g.min(), "n": g.size()})
    for lab, df in C.items():
        lines.append(f"## C. By {lab} at entry (h = 0)\n" + df.round(3).to_string())
    o = ols_newey_west(j.prem.values, j[["richness"]].values, 4, names=["richness"])
    lines.append("## Payoff on richness (NW 4 lags)\n" + o.summary())
    worst = prem.nsmallest(6).to_frame("payoff").join(p[["ic91", "rc91", "vix"]])
    lines.append("## Worst entries\n" + worst.round(3).to_string())
    conv = float((tc.cross / (2 * tc.sigma_I)).median())
    lines.append(f"## Conversion: 1 correlation point ~ {conv * 100 * 0.01:.3f} vol points on the index (median C/(2 sigma_I) = {conv:.3f})")

    txt = "\n\n".join(lines)
    (res / "corr_swap_report.txt").write_text(txt + "\n")
    pickle.dump({"A": A, "B": B, "C": C, "worst": worst, "ols": (o.params, o.tstat, o.r2), "conv": conv},
                open(data / "corr_swap_out.pkl", "wb"))
    print(txt)

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for h, c in zip(MARGINS, ["C0", "C2", "C1", "C3"]):
        x = (prem - h) / 3          # ladder: each monthly swap one-third of the notional budget
        ax.plot(x.index, x.cumsum() * 100, color=c, label=f"Always, margin {int(h * 100)} pts")
    xr = ((j.prem - 0.04) / 3).where(j.richness > 0.04, 0.0)
    ax.plot(xr.index, xr.cumsum() * 100, color="black", ls="--", label="Richness > margin, margin 4 pts")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative P&L, correlation points\nper unit of notional budget")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    d = C["Richness quintile"]
    ax.bar(range(5), d["mean"] * 100, color="C0", label="Average payoff")
    ax.bar(range(5), d["worst"] * 100, color="C3", alpha=0.6, label="Worst payoff")
    ax.set_xticks(range(5)); ax.set_xticklabels(["Q1\n(cheap)", "Q2", "Q3", "Q4", "Q5\n(rich)"], fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Correlation points (margin 0)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(res / "fig_corr_swap.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
