"""Pre-registered cost threshold c (section 6): average round-trip cost of a tranche,
in correlation units, estimated on the estimation sample (entries up to 2010-12-31).

Round-trip cost of a tranche, in vol points per unit of tranche vega (one unit on each leg):
    cost_t = h_SPX + sum_i w_i h_i
    h_x    = [half-spread of the straddle at entry + half-spread at exit] / straddle vega at entry
using the listed contracts actually selected (src.select_contracts) and their quotes on the exit
date (10 trading days before expiry; last quote within 3 trading days if missing).
w_i are index weights renormalized over the top-50 names with a selected contract.

Conversion to correlation units. With sigma_I^2 = own + rho * cross, a change d rho in the
correlation priced into the index moves index implied vol by d sigma_I = cross * d rho / (2 sigma_I).
The vega-neutral tranche earns, per unit of vega, the index-vol mispricing in vol points; so the
correlation premium needed to pay cost_t is
    c_t = (cost_t / 100) * 2 sigma_I,t / cross_t ,
with sigma_I the 91-day SPX ATM IV and cross = (sum w sigma)^2 - sum w^2 sigma^2 over all members
(the same quantities that define IC). c = mean of c_t over estimation-sample entry dates.

Run: python -m src.cost_corr DATA_DIR RESULTS_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.signals import entry_dates

EST_END = pd.Timestamp("2010-12-31")
EXIT_BUFFER = 10
CARRY = 3
SPX_SECID = 108105


def tranche_costs(data: Path) -> pd.DataFrame:
    ch = pd.read_csv(data / "contracts_chosen_v2.csv.gz", parse_dates=["date", "exdate"])
    ch["optionid"] = ch.optionid.astype("int64")
    paths = pd.read_pickle(data / "listed_paths.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    weights = pd.read_pickle(data / "weights_monthly.pkl")
    ivm = pd.read_pickle(data / "iv_members.pkl")
    cal = spx.index
    q = paths.set_index(["optionid", "date"])[["best_bid", "best_offer"]].sort_index()
    by_opt = {k: g.droplevel(0) for k, g in q.groupby(level=0)}

    def exit_half_spread(oid, exdate):
        k = cal.searchsorted(exdate, side="left")          # first trading day >= expiry
        ex = cal[max(k - EXIT_BUFFER, 0)]
        g = by_opt.get(oid)
        if g is None:
            return np.nan
        g = g[(g.index <= ex) & (g.index >= cal[max(k - EXIT_BUFFER - CARRY, 0)])]
        g = g[(g.best_bid > 0) & (g.best_offer > g.best_bid)]
        return np.nan if g.empty else float((g.best_offer.iloc[-1] - g.best_bid.iloc[-1]) / 2)

    rows = []
    iv91 = ivm.pivot_table(index="date", columns="secid", values="iv91")
    for t in entry_dates(spx):
        c = ch[ch.date == t]
        if c.empty:
            continue
        legs = []
        for sid, g in c.groupby("secid"):
            if len(g) != 2:
                continue
            ent = float(((g.best_offer - g.best_bid) / 2).sum())
            ext = sum(exit_half_spread(o, e) for o, e in zip(g.optionid, g.exdate))
            vega = float(g.vega.sum()) / 100.0
            if vega <= 0.005 or not np.isfinite(ext):
                ext = np.nan if not np.isfinite(ext) else ext
            legs.append((sid, ent, ext, vega))
        L = pd.DataFrame(legs, columns=["secid", "ent", "ext", "vega"]).set_index("secid")
        L = L[L.vega > 0.005]
        L["h_ent"] = L.ent / L.vega
        L["h_rt"] = (L.ent + L.ext) / L.vega
        if SPX_SECID not in L.index:
            continue
        w = weights[weights["month"] == t.to_period("M") - 1].dropna(subset=["secid"])
        top = w.sort_values("weight", ascending=False).head(50)
        mem = L.drop(SPX_SECID, errors="ignore").join(top.set_index("secid")["weight"], how="inner")
        mem = mem[np.isfinite(mem.h_rt)]
        ww = mem.weight / mem.weight.sum()
        # IC ingredients over all members
        s = iv91.loc[t].reindex(w.secid).values if t in iv91.index else np.full(len(w), np.nan)
        ok = np.isfinite(s) & (s > 0)
        wa = w.weight.values[ok] / w.weight.values[ok].sum()
        own = np.sum(wa**2 * s[ok]**2)
        cross = np.sum(wa * s[ok])**2 - own
        sig_i = spx["iv91"].get(t, np.nan)
        cost = L.loc[SPX_SECID, "h_rt"] + float(ww @ mem.h_rt)
        rows.append({"date": t, "h_spx_rt": L.loc[SPX_SECID, "h_rt"], "h_basket_rt": float(ww @ mem.h_rt),
                     "h_spx_entry": L.loc[SPX_SECID, "h_ent"], "h_basket_entry": float(ww @ mem.h_ent),
                     "cost_rt": cost, "n_names": len(mem), "sigma_I": sig_i, "cross": cross,
                     "c_t": cost / 100 * 2 * sig_i / cross})
    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    data, res = Path(sys.argv[1]), Path(sys.argv[2])
    df = tranche_costs(data)
    df.to_pickle(data / "tranche_costs.pkl")
    est = df[df.index <= EST_END]
    c = float(est.c_t.mean())
    out = {"c": round(c, 4), "c_median": round(float(est.c_t.median()), 4),
           "n_entries": int(est.c_t.notna().sum()),
           "mean_cost_rt_volpts": round(float(est.cost_rt.mean()), 3),
           "mean_h_spx_rt": round(float(est.h_spx_rt.mean()), 3),
           "mean_h_basket_rt": round(float(est.h_basket_rt.mean()), 3),
           "sample": "entries 1996-01 to 2010-12 (estimation sample)"}
    (res / "cost_threshold_frozen.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    per = pd.cut(df.index.year, [1995, 2000, 2007, 2010, 2019, 2025])
    print(df.groupby(per, observed=True)[["h_spx_rt", "h_basket_rt", "cost_rt", "c_t", "n_names"]].mean().round(3))
