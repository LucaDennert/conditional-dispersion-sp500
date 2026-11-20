"""Richness strategy: sell correlation when the index is rich relative to our forecast, net of costs.

At each entry date t, using only information available at t:
  forecast   HAR regression of 63-day forward realized correlation on RC21, RC63, RC252 and IC (Fisher z),
             re-estimated every month on the entries whose 63-day target window has already ended
             (walk-forward, expanding window, at least 36 observations)
  richness   R_t = IC_t - forecast_t   (correlation points)
  cost       c_t = [entry half-spreads of the SPX and basket straddles observed at t
                    + average hedging cost of the tranches already closed] in vol points,
             converted to correlation units with c = cost/100 * 2 sigma_I / C
  size       min(max(R_t - c_t, 0) / RMSE_t, 1) x one-third of the vega budget, RMSE_t = in-sample
             forecast error at t (full size when the excess richness is one forecast error)
Binary version: size 1 whenever R_t > c_t.
Run: python -m src.richness DATA_DIR RESULTS_DIR
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

from src.corr import fisher_z, inv_fisher_z
from src.ladder import ladder_daily, to_monthly
from src.stats import block_bootstrap_ci, block_bootstrap_sharpe_diff, deflated_sharpe, max_drawdown, sharpe

H = 63
MIN_OBS = 36
COLS = ["rc21", "rc63", "rc252", "ic"]
N_TRIALS = 20     # every strategy/variant evaluated in the project so far, incl. the pre-registered ones


def walk_forward_forecast(sig, prem, cal):
    d = sig[COLS].join(prem[["rc91"]]).dropna(subset=COLS)
    pos = {x: i for i, x in enumerate(cal)}
    out = []
    for t in d.index:
        i = pos[t]
        known = [s for s in d.index if s < t and pos[s] + H <= i and np.isfinite(d.at[s, "rc91"])]
        if len(known) < MIN_OBS:
            continue
        tr = d.loc[known]
        X = np.column_stack([np.ones(len(tr)), fisher_z(tr[COLS].values)])
        y = fisher_z(tr.rc91.values)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        fit = inv_fisher_z(X @ b)
        rmse = float(np.sqrt(np.mean((tr.rc91.values - fit) ** 2)))
        x0 = np.concatenate([[1.0], fisher_z(d.loc[t, COLS].values.astype(float))])
        out.append({"date": t, "rc_hat": float(inv_fisher_z(x0 @ b)), "rmse": rmse, "n_train": len(tr)})
    return pd.DataFrame(out).set_index("date")


def frozen_forecast(sig, res):
    fz = json.load(open(res / "forecast_frozen.json"))["primary"]
    p = np.array(fz["params"]) if "params" in fz else None
    return p


def entry_costs(T, top, long_mult_fn):
    """Entry half-spread cost of the tranche (vol pts per unit vega), observed at t."""
    out = {}
    for t in T.entries():
        E = T.E[t]
        keep = (E["rank"] >= 0) & (E["rank"] < top)
        w = np.where(keep, E["w"], 0.0)
        w = w / w.sum() * long_mult_fn(t)
        hs0 = np.nan_to_num(E["hs"][:, 0])
        out[t] = float(w @ hs0 + hs0[0])
    return pd.Series(out)


def evaluate(D, X, Db, Xb, label):
    M, Mb = to_monthly(D)["pnl"], to_monthly(Db)["pnl"]
    e, eb = float(X.mean()), float(Xb.mean())
    Mb_s = Mb * (e / eb) if eb > 0 else Mb * 0
    lo, hi = block_bootstrap_ci(M.values, sharpe, block=3)
    mlo, mhi = block_bootstrap_ci(M.values, np.mean, block=3)
    d, p = block_bootstrap_sharpe_diff(M.values, Mb_s.values, block=3, draws=10_000, seed=0)
    return {"strategy": label, "avg_vega": e, "mean_month": float(M.mean()), "mean_lo": mlo, "mean_hi": mhi,
            "sharpe": sharpe(M.values), "sh_lo": lo, "sh_hi": hi, "maxDD": max_drawdown(M.values),
            "worst_month": float(M.min()), "sharpe_diff": d, "p": p,
            "dd_ratio": max_drawdown(M.values) / max_drawdown(Mb_s.values) if max_drawdown(Mb_s.values) > 0 else np.nan}


def main(data: Path, res: Path):
    T = pickle.load(open(data / "tranches_expiry.pkl", "rb"))
    sig = pd.read_pickle(data / "signals_monthly.pkl")
    prem = pd.read_pickle(data / "premium_monthly.pkl")
    tc = pd.read_pickle(data / "tranche_costs.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    base = pd.read_pickle(data / "basic_top20_expiry.pkl")
    cal = spx.index
    sqrt_ic = lambda t: float(np.sqrt(max(sig.ic.get(t, np.nan), 0.0)))
    one = lambda t: 1.0

    fc = walk_forward_forecast(sig, prem, cal)
    fc["ic"] = sig.ic.reindex(fc.index)
    fc["richness"] = fc.ic - fc.rc_hat
    # realized forecast error, for reporting only
    fc["rc_realized"] = prem.rc91.reindex(fc.index)

    conv = (2 * tc.sigma_I / tc.cross).reindex(fc.index)

    strategies = {}
    for leg, fn in [("vega-neutral", one), ("sqrt(IC)", sqrt_ic)]:
        # expected hedging cost: mean over tranches already closed at t (same leg weighting)
        ent = entry_costs(T, 20, fn)
        hist = []
        for t in T.entries():
            g = T.daily(t, top=20, long_mult=fn(t), cost_mult=0)
            n = T.daily(t, top=20, long_mult=fn(t))
            hist.append((t, g.index[-1], float(g.sum() - n.sum()) - ent[t]))
        hist = pd.DataFrame(hist, columns=["date", "exit", "hedge"]).set_index("date")
        exp_hedge = pd.Series({t: (hist[hist.exit < t].hedge.mean() if (hist.exit < t).sum() >= 12 else 0.2)
                               for t in fc.index})
        fc[f"exp_hedge_{leg}"] = exp_hedge
        cost = ent.reindex(fc.index) + exp_hedge
        c = cost / 100 * conv
        fc[f"c_{leg}"] = c
        excess = fc.richness - c
        strategies[f"Richness, continuous, {leg}"] = ((excess / fc.rmse).clip(lower=0, upper=1), fn)
        strategies[f"Richness, binary, {leg}"] = ((excess > 0).astype(float), fn)
    fc.to_pickle(data / "richness_signal.pkl")

    start = fc.index[0]
    windows = {"Walk-forward, full": (start, "2025-08-29"), "1999-2010": (start, "2010-12-31"),
               "2011-2025": ("2011-01-01", "2025-08-29")}
    rows, monthly = [], {}
    for wname, (a, b) in windows.items():
        for leg, fn in [("vega-neutral", one), ("sqrt(IC)", sqrt_ic)]:
            Db, Xb, nb = ladder_daily(T, None, a, b, cal, top=20, long_mult=fn)
            Dbg, _, _ = ladder_daily(T, None, a, b, cal, top=20, long_mult=fn, cost_mult=0)
            r = evaluate(Db, Xb, Db, Xb, f"Unconditional, {leg}")
            r.update({"window": wname, "tranches": nb, "mean_month_gross": float(to_monthly(Dbg)["pnl"].mean())})
            rows.append(r); monthly[(wname, f"Unconditional, {leg}")] = to_monthly(Db)["pnl"]
            for sname, (size, f2) in strategies.items():
                if f2 is not fn:
                    continue
                D, X, n = ladder_daily(T, size, a, b, cal, top=20, long_mult=fn)
                Dg, _, _ = ladder_daily(T, size, a, b, cal, top=20, long_mult=fn, cost_mult=0)
                r = evaluate(D, X, Db, Xb, sname)
                r.update({"window": wname, "tranches": n, "mean_month_gross": float(to_monthly(Dg)["pnl"].mean())})
                rows.append(r); monthly[(wname, sname)] = to_monthly(D)["pnl"]
    tab = pd.DataFrame(rows).set_index(["window", "strategy"])

    # robustness: other split years for the main strategy (Sharpe before/after)
    main_name = "Richness, continuous, sqrt(IC)"
    size_main = strategies[main_name][0]
    split = []
    for y in [2005, 2008, 2011, 2015]:
        for lab, (a, b) in [("before", (start, f"{y - 1}-12-31")), ("after", (f"{y}-01-01", "2025-08-29"))]:
            D, X, _ = ladder_daily(T, size_main, a, b, cal, top=20, long_mult=sqrt_ic)
            Db, Xb, _ = ladder_daily(T, None, a, b, cal, top=20, long_mult=sqrt_ic)
            m, mb = to_monthly(D)["pnl"], to_monthly(Db)["pnl"]
            split.append({"split": y, "part": lab, "sharpe_strategy": sharpe(m.values), "sharpe_uncond": sharpe(mb.values)})
    split = pd.DataFrame(split).set_index(["split", "part"])

    # deflated Sharpe of the best strategy, walk-forward full
    srs = [m.mean() / m.std(ddof=1) for (w, s), m in monthly.items() if w == "Walk-forward, full" and m.std() > 0]
    v = float(np.var(srs, ddof=1))
    dsr = {s: round(deflated_sharpe(m.values, N_TRIALS, v), 3) for (w, s), m in monthly.items() if w == "Walk-forward, full"}

    # signal diagnostics: does richness predict tranche P&L (gross, net)?
    j = fc.join(base[["gross", "net"]]).dropna(subset=["richness", "net"])
    from src.stats import ols_newey_west
    diag = {}
    for col in ["gross", "net"]:
        o = ols_newey_west(j[col].values, j[["richness"]].values, 2, names=["richness"])
        diag[col] = {"slope": round(float(o.params[1]), 2), "t": round(float(o.tstat[1]), 2), "R2": round(float(o.r2), 3)}
    q = pd.qcut(j.richness, 5, labels=["Q1 (cheap)", "Q2", "Q3", "Q4", "Q5 (rich)"])
    quint = j.groupby(q, observed=True)[["richness", "gross", "net"]].mean()
    quint["n"] = j.groupby(q, observed=True).size()

    fe = (fc.rc_realized - fc.rc_hat).dropna()
    lines = [
        f"## Walk-forward forecast: first date {start.date()}, out-of-sample RMSE {np.sqrt((fe ** 2).mean()):.4f}, "
        f"mean richness {fc.richness.mean():.4f}, mean cost (vega-neutral) {fc['c_vega-neutral'].mean():.4f}, "
        f"share of months richness > cost: {(fc.richness > fc['c_vega-neutral']).mean():.2f} (vega-neutral), "
        f"{(fc.richness > fc['c_sqrt(IC)']).mean():.2f} (sqrt(IC))",
        "## Results (monthly ladder, per unit vega budget; CIs 95% block bootstrap; p = one-sided Sharpe diff vs vega-matched unconditional)\n"
        + tab.round(3).to_string(),
        "## Robustness to the split year (main: continuous, sqrt(IC))\n" + split.round(2).to_string(),
        f"## Deflated Sharpe (walk-forward full, {N_TRIALS} trials)\n" + json.dumps(dsr, indent=1),
        "## Does richness predict tranche P&L (top-20, vega-neutral, held to expiry)?\n" + json.dumps(diag, indent=1)
        + "\n" + quint.round(3).to_string(),
    ]
    txt = "\n\n".join(lines)
    (res / "richness_report.txt").write_text(txt + "\n")
    print(txt)
    pickle.dump({"fc": fc, "tab": tab, "split": split, "dsr": dsr, "diag": diag, "quint": quint, "monthly": monthly},
                open(data / "richness_out.pkl", "wb"))

    # figures
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(fc.index, fc.richness, color="C0", lw=1.2, label="Richness: IC minus forecast RC")
    ax.plot(fc.index, fc["c_sqrt(IC)"], color="C3", lw=1.2, label="Cost hurdle at entry (correlation points)")
    ax.fill_between(fc.index, fc["c_sqrt(IC)"], fc.richness, where=fc.richness > fc["c_sqrt(IC)"], color="C0", alpha=0.25)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Correlation points")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(res / "fig_richness_signal.png", dpi=150)

    fig, ax = plt.subplots(figsize=(9, 4))
    w = "Walk-forward, full"
    for s, c, ls in [("Unconditional, sqrt(IC)", "grey", "--"), ("Richness, continuous, sqrt(IC)", "C0", "-"),
                     ("Unconditional, vega-neutral", "grey", ":"), ("Richness, continuous, vega-neutral", "C1", "-")]:
        m = monthly[(w, s)]
        ax.plot(m.index.to_timestamp(), m.cumsum(), color=c, ls=ls, lw=1.6 if "Richness" in s else 1.0, label=s)
    ax.axvline(pd.Timestamp("2011-01-01"), color="black", lw=0.6, ls=":")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative net P&L, vol pts per unit of vega budget")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(res / "fig_richness_cumulative.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    xx = np.arange(len(quint))
    ax.bar(xx - 0.2, quint.gross, 0.4, label="Gross", color="C2")
    ax.bar(xx + 0.2, quint.net, 0.4, label="Net", color="C0")
    ax.set_xticks(xx); ax.set_xticklabels(quint.index, fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Average P&L per tranche (vol pts)")
    ax.set_xlabel("Richness quintile at entry")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(res / "fig_richness_quintiles.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
