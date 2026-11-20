"""Can machine learning time the trade? Walk-forward ridge and gradient boosting.

Target: net P&L of the main trade (top-20, held to expiry, long leg x sqrt(IC)).
Predictors (all observed at entry; the cost predictor uses only the entry half-spreads, fixing the
look-ahead of the earlier version that used exit spreads): CP-style richness, IC 3M, IC slope,
VIX, index volatility premium, index skew premium, cross-sectional dispersion of member IVs, entry cost.
Each January the models are refit on the tranches already closed; ridge penalty by inner
expanding-window CV. Size = min(max(pred, 0) / m, 1), m = 90th percentile of training fits.
Run: python -m src.ml_walkforward DATA_DIR RESULTS_DIR
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.ladder import ladder_daily, to_monthly
from src.richness import entry_costs, evaluate
from src.stats import ols_newey_west

GRID = 10.0 ** np.arange(-2, 3.01, 0.5)
GBM = dict(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
FEATS = ["richness", "ic91", "ic_slope", "vix", "vp", "skew", "iv_disp", "entry_cost"]


def main(data: Path, res: Path):
    T = pickle.load(open(data / "tranches_expiry.pkl", "rb"))
    sig = pd.read_pickle(data / "signals_monthly.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    f = pd.read_pickle(data / "part4_features.pkl")
    fc = pd.read_pickle(data / "richness_signal.pkl")
    cal = spx.index
    sqrt_ic = lambda t: float(np.sqrt(max(sig.ic.get(t, np.nan), 0.0)))
    X = f[["ic91", "ic_slope", "vix", "vp", "skew", "iv_disp"]].copy()
    X["richness"] = fc.richness
    X["entry_cost"] = entry_costs(T, 20, sqrt_ic)
    X = X[FEATS].dropna()
    y = pd.Series({t: T.daily(t, top=20, long_mult=sqrt_ic(t)).sum() for t in T.entries()})
    ex = pd.Series({t: T.E[t]["dates"][-1] for t in T.entries()})
    d = X.join(y.rename("y")).join(ex.rename("exit")).dropna(subset=FEATS)

    preds = {"ridge": {}, "gbm": {}}
    ms = {"ridge": {}, "gbm": {}}
    for Y in range(2003, 2026):
        y0 = pd.Timestamp(f"{Y}-01-01")
        tr = d[(d.exit < y0) & d.y.notna()]
        te = d[(d.index >= y0) & (d.index < pd.Timestamp(f"{Y + 1}-01-01"))]
        if len(tr) < 48 or te.empty:
            continue
        mu, sd = tr[FEATS].mean(), tr[FEATS].std()
        Xtr, Xte = ((tr[FEATS] - mu) / sd).values, ((te[FEATS] - mu) / sd).values
        # inner CV for lambda: validate on the last 3 years of the training window, one year at a time
        best, best_l = np.inf, 1.0
        for lam in GRID:
            err = []
            for V in range(Y - 3, Y):
                vs = pd.Timestamp(f"{V}-01-01")
                a = (tr.exit < vs).values
                b = ((tr.index >= vs) & (tr.index < pd.Timestamp(f"{V + 1}-01-01")))
                if a.sum() < 36 or b.sum() == 0:
                    continue
                m_ = Ridge(alpha=lam).fit(Xtr[a], tr.y.values[a])
                err.extend((m_.predict(Xtr[np.asarray(b)]) - tr.y.values[np.asarray(b)]) ** 2)
            if err and np.mean(err) < best:
                best, best_l = np.mean(err), lam
        r = Ridge(alpha=best_l).fit(Xtr, tr.y.values)
        g = GradientBoostingRegressor(**GBM).fit(Xtr, tr.y.values)
        for name, mdl in [("ridge", r), ("gbm", g)]:
            m = float(np.percentile(mdl.predict(Xtr), 90))
            for t, p in zip(te.index, mdl.predict(Xte)):
                preds[name][t] = p
                ms[name][t] = m
    out_rows = []
    a, b = "2003-01-01", "2025-08-29"
    Db, Xb, nb = ladder_daily(T, None, a, b, cal, top=20, long_mult=sqrt_ic)
    r0 = evaluate(Db, Xb, Db, Xb, "Unconditional (main trade)"); r0["tranches"] = nb
    out_rows.append(r0)
    diag = {}
    for name in ["ridge", "gbm"]:
        p = pd.Series(preds[name]); m = pd.Series(ms[name])
        size = (p / m.where(m > 0)).clip(lower=0, upper=1).fillna(0.0)
        D, Xx, n = ladder_daily(T, size, a, b, cal, top=20, long_mult=sqrt_ic)
        r = evaluate(D, Xx, Db, Xb, f"ML timing: {name}"); r["tranches"] = n
        out_rows.append(r)
        j = pd.DataFrame({"p": p, "y": y}).dropna()
        o = ols_newey_west(j.y.values, j[["p"]].values, 2, names=["pred"])
        diag[name] = {"oos_corr_pred_actual": round(float(j.corr().iloc[0, 1]), 3), "slope": round(float(o.params[1]), 3),
                      "t": round(float(o.tstat[1]), 2), "oos_R2_vs_zero": round(float(1 - ((j.y - j.p) ** 2).sum() / (j.y ** 2).sum()), 3),
                      "share_traded": round(float((size > 0).mean()), 3)}
    tab = pd.DataFrame(out_rows).set_index("strategy")
    txt = ("## Walk-forward ML timing, 2003-2025 (monthly ladder per unit vega budget)\n" + tab.round(3).to_string()
           + "\n\n## Out-of-sample forecast quality\n" + json.dumps(diag, indent=1))
    (res / "ml_walkforward_report.txt").write_text(txt + "\n")
    pickle.dump({"tab": tab, "diag": diag}, open(data / "ml_walkforward_out.pkl", "wb"))
    print(txt)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
