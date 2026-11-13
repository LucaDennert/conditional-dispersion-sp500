"""Combine engine legs into dispersion tranches (daily P&L per unit of tranche vega).

tranche = sum_i w_i * long_straddle_i  -  long_straddle_SPX,  all legs net of their own costs.
Options: basket size (top-50 or top-30 by weight), long-leg multiplier (1 = vega-neutral,
sqrt(IC) for variant 4), cost multiplier, optional stop day (whole tranche closed at that close).
"""
from pathlib import Path

import numpy as np
import pandas as pd

SPX_SECID = 108105
BP = {"index": 0.5e-4, "stock": 1e-4}


class Tranches:
    def __init__(self, data: Path):
        legs = pd.read_pickle(data / "engine_legs.pkl")
        weights = pd.read_pickle(data / "weights_monthly.pkl").dropna(subset=["secid"])
        self.cal = pd.read_pickle(data / "spx_daily.pkl").index
        self.E = {}
        for t0, g in legs.groupby("entry"):
            w = weights[weights["month"] == t0.to_period("M") - 1].sort_values("weight", ascending=False)
            rank = {int(s): k for k, s in enumerate(w.secid.values)}
            wt = dict(zip(w.secid.astype(int), w.weight))
            names = [s for s in g.secid.unique() if s != SPX_SECID and rank.get(s, 999) < 50]
            if SPX_SECID not in set(g.secid) or len(names) < 2:
                continue
            n = int(g.sched_exit.max()) + 1                  # legs can have different expiries
            i0 = self.cal.get_loc(t0)
            dates = self.cal[i0: i0 + n]
            def mat(col, sids):
                p = g.pivot(index="secid", columns="day", values=col).reindex(index=sids)
                return p.reindex(columns=range(n)).values.astype(float)
            sids = [SPX_SECID] + names
            self.E[t0] = {
                "dates": pd.DatetimeIndex(dates), "sids": sids,
                "rank": np.array([-1] + [rank[s] for s in names]),
                "w": np.array([0.0] + [wt[s] for s in names]),
                "mid": np.nan_to_num(mat("mid", sids)), "hc": np.nan_to_num(mat("hc", sids)),
                "hs": mat("hs", sids), "hpos": np.nan_to_num(mat("hpos", sids)),
                "exit": g.groupby("secid").exit_day.first().reindex(sids).values.astype(int),
            }

    def entries(self):
        return sorted(self.E)

    def daily(self, t0, top=50, long_mult=1.0, cost_mult=1.0, stop=None, legs=False):
        """Daily P&L of one tranche (index = trading dates). stop = day index to close at, or None."""
        E = self.E[t0]
        m, hc, hs, hpos, ex = E["mid"].copy(), E["hc"].copy(), E["hs"], E["hpos"], E["exit"].copy()
        nd = m.shape[1]
        bp = np.array([BP["index"]] + [BP["stock"]] * (len(E["sids"]) - 1))
        if stop is not None:
            for j in range(len(ex)):
                if ex[j] > stop:
                    hc[j, stop] = abs(hpos[j, stop - 1]) * bp[j] if stop >= 1 else 0.0
                    m[j, stop + 1:] = 0.0
                    hc[j, stop + 1:] = 0.0
                    ex[j] = stop
        cost = np.zeros_like(m)
        for j in range(len(ex)):
            cost[j, 0] += np.nan_to_num(hs[j, 0])
            if ex[j] < nd:
                cost[j, ex[j]] += np.nan_to_num(hs[j, ex[j]], nan=np.nan_to_num(hs[j, 0]))
        c_all = cost_mult * (hc + cost)
        net = m - c_all
        keep = (E["rank"] >= 0) & (E["rank"] < top)
        w = np.where(keep, E["w"], 0.0)
        w = w / w.sum() * long_mult
        long_leg = w @ net
        short_leg = -m[0] - c_all[0]                      # short straddle pays its own costs
        out = pd.DataFrame({"long": long_leg, "short": short_leg}, index=E["dates"][:nd])
        out["pnl"] = out["long"] + out["short"]
        return out if legs else out["pnl"]

    def total(self, t0, **kw):
        return float(self.daily(t0, **kw).sum())
