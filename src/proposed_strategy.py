"""Profile of the proposed strategy: correlation-weighted top-20 dispersion, held to expiry,
sized by richness net of the entry-day cost hurdle (src.richness). Figure + summary.
Run: python -m src.proposed_strategy DATA_DIR RESULTS_DIR
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

from src.basic_trade import STRESS
from src.ladder import ladder_daily, to_monthly
from src.stats import block_bootstrap_ci, max_drawdown, sharpe


def main(data: Path, res: Path):
    T = pickle.load(open(data / "tranches_expiry.pkl", "rb"))
    sig = pd.read_pickle(data / "signals_monthly.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    fc = pd.read_pickle(data / "richness_signal.pkl")
    sq = lambda t: float(np.sqrt(max(sig.ic.get(t, np.nan), 0.0)))
    size = ((fc.richness - fc["c_sqrt(IC)"]) / fc.rmse).clip(lower=0, upper=1)
    a, b = fc.index[0], "2025-08-29"
    cal = spx.index
    D, X, n = ladder_daily(T, size, a, b, cal, top=20, long_mult=sq)
    Dg, _, _ = ladder_daily(T, size, a, b, cal, top=20, long_mult=sq, cost_mult=0)
    Db, Xb, nb = ladder_daily(T, None, a, b, cal, top=20, long_mult=sq)
    Ds, _, _ = ladder_daily(T, None, a, b, cal, top=20)
    out = {}
    for lab, m, x in [("proposed", to_monthly(D)["pnl"], X), ("always_cw", to_monthly(Db)["pnl"], Xb),
                      ("short_spx", to_monthly(Ds)["short"], Xb)]:
        lo, hi = block_bootstrap_ci(m.values, sharpe, block=3)
        mlo, mhi = block_bootstrap_ci(m.values, np.mean, block=3)
        out[lab] = {"mean_month": round(float(m.mean()), 3), "mean_ci": [round(mlo, 3), round(mhi, 3)],
                    "per_unit_vega_deployed": round(float(m.mean() / x.mean()), 3), "avg_vega": round(float(x.mean()), 3),
                    "sharpe": round(sharpe(m.values), 2), "sharpe_ci": [round(lo, 2), round(hi, 2)],
                    "maxDD": round(max_drawdown(m.values), 2), "worst_month": round(float(m.min()), 2),
                    "skew": round(float(pd.Series(m.values).skew()), 2)}
    out["proposed"]["gross_mean_month"] = round(float(to_monthly(Dg)["pnl"].mean()), 3)
    out["proposed"]["share_months_traded"] = round(float((size > 0).mean()), 2)
    out["proposed"]["share_full_size"] = round(float((size >= 1).mean()), 2)
    out["proposed"]["avg_size_when_traded"] = round(float(size[size > 0].mean()), 2)
    st = {}
    for name, (x0, x1) in STRESS.items():
        if pd.Timestamp(x1) < pd.Timestamp(a):
            continue
        st[name] = {"proposed": round(float(D.loc[x0:x1, "pnl"].sum()), 2), "always_cw": round(float(Db.loc[x0:x1, "pnl"].sum()), 2),
                    "short_spx": round(float(Ds.loc[x0:x1, "short"].sum()), 2)}
    out["stress"] = st
    (res / "proposed_strategy.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))

    M, Mb = to_monthly(D)["pnl"], to_monthly(Db)["pnl"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(M.index.to_timestamp(), M.cumsum(), color="C0", lw=2, label="Proposed: richness-sized, correlation-weighted, top-20")
    ax.plot(Mb.index.to_timestamp(), Mb.cumsum(), color="grey", ls="--", label="Correlation-weighted, every month (full size)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative net P&L\n(vol pts per unit vega budget)")
    ax.legend(frameon=False, fontsize=8)
    ax2.fill_between(size.index, size, step="post", color="C0", alpha=0.4)
    ax2.set_ylabel("Size of new\ntranche (0-1)")
    ax2.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(res / "fig_proposed.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
