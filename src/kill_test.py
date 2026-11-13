"""Pre-registered kill test (sections 8-11 and amendments 1-2), run once on 2011-2025.

Also reports the same statistics on the estimation sample (in-sample, for information only).
Run: python -m src.kill_test DATA_DIR RESULTS_DIR
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.corr import realized_correlation
from src.stats import block_bootstrap_sharpe_diff, deflated_sharpe, max_drawdown, sharpe

SAMPLES = {"estimation": ("1996-01-01", "2010-12-31"), "test": ("2011-01-01", "2025-08-29")}
KS = [0, 0.25, 0.5, 0.75, 1.0]


class Book:
    def __init__(self, data: Path, res: Path):
        self.data = data
        self.T = pickle.load(open(data / "tranches_obj.pkl", "rb"))
        self.sig = pd.read_pickle(data / "signals_monthly.pkl")
        self.spx = pd.read_pickle(data / "spx_daily.pkl")
        fz = json.load(open(res / "forecast_frozen.json"))
        self.rmse = fz["primary"]["rmse_est"]
        self.c = json.load(open(res / "cost_threshold_frozen.json"))["c"]
        v = self.spx.loc[:"2010-12-31", "vix"].dropna()
        self.vix80, self.vix95 = float(v.quantile(0.80)), float(v.quantile(0.95))
        self.cache = {}

    # ---------- sizes ----------
    def rule(self, k=0.5, vix=True, vp=False):
        s = self.sig
        ok = s.cp > self.c + k * self.rmse
        if vix:
            ok &= s.vix < self.vix80
        if vp:
            ok &= s.vp > 0
        return ok.astype(float)

    def continuous(self, k=0.5):
        s = self.sig
        x = ((s.cp - self.c).clip(lower=0) / (k * self.rmse)).clip(upper=1.0)
        return x.where(s.vix < self.vix80, 0.0)

    def every_month(self):
        return pd.Series(1.0, index=self.sig.index)

    # ---------- stops ----------
    def stop_day(self, t0):
        if t0 in self.cache:
            return self.cache[t0]
        E = self.T.E[t0]
        dates = E["dates"]
        vix = self.spx["vix"].reindex(dates).values
        ic = self.sig.ic.get(t0, np.nan)
        if not hasattr(self, "_R"):
            mem = pd.read_pickle(self.data / "members_daily.pkl").dropna(subset=["secid"])
            self._R = mem.pivot_table(index="date", columns="secid", values="logret")
            self._RI = self.spx["spx_logret"]
        sids = [s for s in E["sids"][1:]]
        w = pd.Series(E["w"][1:], index=sids)
        stop = None
        win = self.spx.index[max(self.spx.index.get_loc(dates[0]) - 25, 0): self.spx.index.get_loc(dates[-1]) + 1]
        RM = self._R.reindex(index=win, columns=sids)
        RI = self._RI.reindex(win)
        for d in range(1, len(dates)):
            if np.isfinite(vix[d]) and vix[d] > self.vix95:
                stop = d
                break
            j = win.get_loc(dates[d]) + 1
            ri = RI.iloc[j - 21: j]
            rm = RM.iloc[j - 21: j]
            rc = realized_correlation(ri, rm, w)
            if np.isfinite(rc) and np.isfinite(ic) and rc - ic > 0.15:
                stop = d
                break
        self.cache[t0] = stop
        return stop

    # ---------- ladder ----------
    def monthly(self, size, sample, scale=1.0, **kw):
        a, b = map(pd.Timestamp, SAMPLES[sample])
        stops = kw.pop("stops", False)
        daily = []
        expo = []
        for t0 in self.T.entries():
            if not (a <= t0 <= b):
                continue
            z = float(size.get(t0, 0.0))
            if z <= 0:
                continue
            kk = dict(kw)
            if stops:
                kk["stop"] = self.stop_day(t0)
            if kk.get("long_mult") == "sqrt_ic":
                kk["long_mult"] = float(np.sqrt(max(self.sig.ic.get(t0, np.nan), 0)))
            d = self.T.daily(t0, legs=True, **kk) * (z / 3.0) * scale
            d = d[d.index <= b]
            daily.append(d)
            live = d.index[: (len(d) if kk.get("stop") is None else kk["stop"] + 1)]
            expo.append(pd.Series(z / 3.0, index=live))
        idx = self.spx.loc[a:b].index
        if not daily:
            z = pd.Series(0.0, index=pd.period_range(a, b, freq="M"))
            return pd.DataFrame({"pnl": z, "long": z, "short": z}), 0.0, 0
        D = pd.concat(daily).groupby(level=0).sum().reindex(idx, fill_value=0.0)
        X = pd.concat(expo).groupby(level=0).sum().reindex(idx, fill_value=0.0)
        M = D.groupby(D.index.to_period("M")).sum()
        return M, float(X.mean()), len(daily)


def stats_block(m, name, bench=None, n_tr=None, expo=None):
    x = m["pnl"].values
    out = {"strategy": name, "tranches": n_tr, "avg_vega": round(expo, 3) if expo is not None else None,
           "mean_month": round(float(x.mean()), 3), "sharpe": round(sharpe(x), 3),
           "max_dd": round(max_drawdown(x), 2), "hit": round(float((x[x != 0] > 0).mean()), 3) if (x != 0).any() else np.nan,
           "long_leg": round(float(m["long"].sum()), 1), "short_leg": round(float(m["short"].sum()), 1)}
    if bench is not None:
        d, p = block_bootstrap_sharpe_diff(x, bench["pnl"].values, block=3, draws=10_000, seed=0)
        out["sharpe_diff_vs_bench"] = round(d, 3)
        out["p_boot"] = round(p, 4)
        out["dd_ratio"] = round(max_drawdown(x) / max_drawdown(bench["pnl"].values), 3)
    return out


def matched_bench(B, sample, expo_target):
    mb, eb, nb = B.monthly(B.every_month(), sample)
    s = expo_target / eb if eb > 0 else 0.0
    return mb * s, eb * s, nb


def run(data: Path, res: Path):
    B = Book(data, res)
    P4 = pickle.load(open(data / "part4_models.pkl", "rb"))
    feats = pd.read_pickle(data / "part4_features.pkl")
    fz4 = json.load(open(res / "part4_frozen.json"))
    Xall = ((feats[fz4["features"]] - P4["mu"]) / P4["sd"])
    okx = Xall.notna().all(axis=1)

    def p4_size(pred, m):
        if m <= 0:
            return pd.Series(0.0, index=pred.index)
        return (pred / m).clip(lower=0, upper=1)

    pred_r = pd.Series(P4["ridge"].predict(Xall[okx].values), index=Xall[okx].index)
    pred_g = pd.Series(P4["gbm"].predict(Xall[okx].values), index=Xall[okx].index)
    # ridge refits at neighbouring lambdas (robustness), same estimation data
    est_idx = pd.read_pickle(data / "tranche_net_unconditional.pkl")
    est = Xall[okx].join(est_idx).dropna()
    est = est[est.exit <= pd.Timestamp("2010-12-31")]
    lam = fz4["ridge"]["lambda"]
    alt = {}
    for l2 in [lam / 10 ** 0.5, lam * 10 ** 0.5]:
        r2 = Ridge(alpha=l2).fit(est[fz4["features"]].values, est.net.values)
        alt[l2] = pd.Series(r2.predict(Xall[okx].values), index=Xall[okx].index)

    strategies = {
        "Primary (k=0.5, VIX filter)": (B.rule(0.5), {}),
        "V1 CP filter only": (B.rule(0.5, vix=False), {}),
        "V2 primary + VP>0": (B.rule(0.5, vp=True), {}),
        "V3 primary + stops": (B.rule(0.5), {"stops": True}),
        "V4 long leg x sqrt(IC)": (B.rule(0.5), {"long_mult": "sqrt_ic"}),
        "V5 top-30 basket": (B.rule(0.5), {"top": 30}),
        "V6 costs doubled": (B.rule(0.5), {"cost_mult": 2.0}),
        "V7 continuous sizing": (B.continuous(0.5), {}),
        "P4 ridge (final trade)": (p4_size(pred_r, fz4["ridge"]["m_p90"]), {}),
        "P4 gradient boosting": (p4_size(pred_g, fz4["gbm"]["m_p90"]), {}),
    }
    report = {}
    monthly_all = {}
    for sample in ["estimation", "test"]:
        rows = []
        bench_raw, eb_raw, nb = B.monthly(B.every_month(), sample)
        rows.append(stats_block(bench_raw, "Unconditional (every month), unscaled", n_tr=nb, expo=eb_raw))
        for name, (size, kw) in strategies.items():
            kw2 = dict(kw)
            if name == "V6 costs doubled":
                bkw = {"cost_mult": 2.0}
            elif name == "V5 top-30 basket":
                bkw = {"top": 30}
            else:
                bkw = {}
            m, e, n = B.monthly(size, sample, **kw2)
            bm, be, _ = B.monthly(B.every_month(), sample, **bkw)
            bm = bm * (e / be) if be > 0 else bm * 0
            rows.append(stats_block(m, name, bench=bm, n_tr=n, expo=e))
            monthly_all[(sample, name)] = m["pnl"]
        monthly_all[(sample, "bench_unscaled")] = bench_raw["pnl"]
        report[sample] = pd.DataFrame(rows).set_index("strategy")

    # k robustness and part-4 robustness (test sample)
    rob = []
    for k in KS:
        m, e, n = B.monthly(B.rule(k), "test")
        bm, be, _ = B.monthly(B.every_month(), "test")
        bm = bm * (e / be) if be > 0 else bm * 0
        rob.append(stats_block(m, f"Primary k={k}", bench=bm, n_tr=n, expo=e))
    for lab, mm in [("m=p75", fz4["ridge"]["m_p75"]), ("m=p90", fz4["ridge"]["m_p90"]), ("m=p100", fz4["ridge"]["m_p100"])]:
        m, e, n = B.monthly(p4_size(pred_r, mm), "test")
        if n == 0:
            rob.append({"strategy": f"P4 ridge {lab}", "tranches": 0, "note": "m <= 0: rule degenerate, no trades"})
            continue
        bm, be, _ = B.monthly(B.every_month(), "test")
        bm = bm * (e / be)
        rob.append(stats_block(m, f"P4 ridge {lab}", bench=bm, n_tr=n, expo=e))
    for l2, pr in alt.items():
        m, e, n = B.monthly(p4_size(pr, fz4["ridge"]["m_p90"]), "test")
        if n == 0:
            rob.append({"strategy": f"P4 ridge lambda={l2:.3g}", "tranches": 0})
            continue
        bm, be, _ = B.monthly(B.every_month(), "test")
        bm = bm * (e / be)
        rob.append(stats_block(m, f"P4 ridge lambda={l2:.3g}", bench=bm, n_tr=n, expo=e))
    report["robustness_test"] = pd.DataFrame(rob).set_index("strategy")

    # P4 vs part-3 primary (test)
    a = monthly_all[("test", "P4 ridge (final trade)")].values
    b = monthly_all[("test", "Primary (k=0.5, VIX filter)")].values
    d, p = block_bootstrap_sharpe_diff(a, b, block=3, draws=10_000, seed=0)

    # deflated Sharpe, 10 trials (non-annualized Sharpe variance across trials)
    names = list(strategies)
    srs = []
    for n_ in names:
        x = monthly_all[("test", n_)].values
        srs.append(x.mean() / x.std(ddof=1) if x.std() > 0 else 0.0)
    v = float(np.var(srs, ddof=1))
    dsr = {n_: round(deflated_sharpe(monthly_all[("test", n_)].values, 10, v), 4) for n_ in names}

    # sub-periods (test)
    sub = {}
    for n_ in ["Primary (k=0.5, VIX filter)", "P4 ridge (final trade)", "bench_unscaled"]:
        x = monthly_all[("test", n_)]
        sub[n_] = {lab: round(sharpe(x[(x.index >= pd.Period(a_, "M")) & (x.index <= pd.Period(b_, "M"))].values), 3)
                   for lab, a_, b_ in [("2011-2014", "2011-01", "2014-12"), ("2015-2019", "2015-01", "2019-12"),
                                       ("2020-2025", "2020-01", "2025-08")]}

    # decision for primary
    t = report["test"].loc["Primary (k=0.5, VIX filter)"]
    r_ = report["robustness_test"].loc[[f"Primary k={k}" for k in KS]]
    c1 = t.sharpe_diff_vs_bench > 0 and t.p_boot < 0.10
    c2 = t.dd_ratio <= 2 / 3
    c3 = bool(((r_.sharpe_diff_vs_bench > 0) & (r_.dd_ratio <= 2 / 3)).all())
    flip = bool((np.sign(r_.sharpe_diff_vs_bench.values) != np.sign(r_.sharpe_diff_vs_bench.values[0])).any())
    if c1 and c2 and c3:
        verdict = "KEEP"
    elif (not c1 and t.dd_ratio > 2 / 3) or flip:
        verdict = "DROP"
    else:
        verdict = "INCONCLUSIVE"
    t4 = report["test"].loc["P4 ridge (final trade)"]
    v4 = ("KEEP" if (t4.sharpe_diff_vs_bench > 0 and t4.p_boot < 0.10 and t4.dd_ratio <= 2 / 3)
          else "DROP" if (not (t4.sharpe_diff_vs_bench > 0 and t4.p_boot < 0.10) and t4.dd_ratio > 2 / 3)
          else "INCONCLUSIVE")

    lines = []
    for k_, df in report.items():
        lines.append(f"## {k_}\n" + df.to_string())
    lines.append(f"## Part 4 ridge vs part-3 primary (test): Sharpe diff {d:.3f}, one-sided p {p:.4f}")
    lines.append("## Deflated Sharpe (test, 10 trials)\n" + json.dumps(dsr, indent=1))
    lines.append("## Sharpe by sub-period (test)\n" + json.dumps(sub, indent=1))
    lines.append(f"## Decision\nPrimary: criteria 1={c1}, 2={c2}, 3={c3}, sign flip across k={flip} -> {verdict}\n"
                 f"Final trade (ridge): {v4}\n"
                 f"Thresholds: c={B.c}, RMSE={B.rmse:.4f}, VIX80={B.vix80:.2f}, VIX95={B.vix95:.2f}")
    txt = "\n\n".join(lines)
    (res / "kill_test_report.txt").write_text(txt + "\n")
    pickle.dump({"report": report, "monthly": monthly_all}, open(data / "kill_test_out.pkl", "wb"))
    for k_, df in report.items():
        df.to_csv(res / f"kill_test_{k_}.csv")
    print(txt)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
