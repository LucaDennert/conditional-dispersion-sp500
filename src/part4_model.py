"""Final trade (part 4, pre-registration amendment 2): forecast the net P&L of a tranche.

Estimation uses only tranches whose exit date is on or before 2010-12-31. Writes the frozen model
to results/part4_frozen.json and the predictor panel to DATA/part4_features.pkl.
Run: python -m src.part4_model DATA_DIR RESULTS_DIR
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

EST_END = pd.Timestamp("2010-12-31")
FEATURES = ["cp", "ic91", "ic_slope", "vix", "vp", "skew", "iv_disp", "cost_rt"]
GRID = 10.0 ** np.arange(-2, 3.01, 0.5)
GBM = dict(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


def features(data: Path) -> pd.DataFrame:
    sig = pd.read_pickle(data / "signals_monthly.pkl")
    pm = pd.read_pickle(data / "premium_monthly.pkl")
    sk = pd.read_pickle(data / "spx_var_strike.pkl")
    tc = pd.read_pickle(data / "tranche_costs.pkl")
    ivm = pd.read_pickle(data / "iv_members.pkl")
    w = pd.read_pickle(data / "weights_monthly.pkl").dropna(subset=["secid"])
    disp = {}
    for t in sig.index:
        ww = w[w["month"] == t.to_period("M") - 1]
        iv = ivm[ivm.date == t].set_index("secid").iv91.reindex(ww.secid)
        ok = iv.notna().values & (iv.values > 0)
        if ok.sum() < 50:
            continue
        a = ww.weight.values[ok] / ww.weight.values[ok].sum()
        mu = a @ iv.values[ok]
        disp[t] = float(np.sqrt(a @ (iv.values[ok] - mu) ** 2))
    f = pd.DataFrame({
        "cp": sig.cp, "ic91": pm.ic91, "ic_slope": pm.ic91 - pm.ic30, "vix": sig.vix, "vp": sig.vp,
        "skew": sk.vol_strike_mf - sk.atm, "iv_disp": pd.Series(disp), "cost_rt": tc.cost_rt,
    })
    return f


def tranche_targets(data: Path) -> pd.DataFrame:
    T = pickle.load(open(data / "tranches_obj.pkl", "rb"))
    rows = []
    for t in T.entries():
        d = T.daily(t)
        rows.append({"date": t, "exit": d.index[-1], "net": float(d.sum())})
    return pd.DataFrame(rows).set_index("date")


def main(data: Path, res: Path):
    f = features(data)
    f.to_pickle(data / "part4_features.pkl")
    y = tranche_targets(data)
    y.to_pickle(data / "tranche_net_unconditional.pkl")
    est = f.join(y).dropna()
    est = est[est.exit <= EST_END]
    mu, sd = est[FEATURES].mean(), est[FEATURES].std()
    X = ((est[FEATURES] - mu) / sd).values
    Y = est.net.values

    # expanding-window CV: validation year V, train on tranches exited before 1 Jan V
    cv = []
    for lam in GRID:
        err = []
        for V in range(2002, 2011):
            vs = pd.Timestamp(f"{V}-01-01")
            tr = (est.exit < vs).values
            va = np.asarray((est.index >= vs) & (est.index < pd.Timestamp(f"{V + 1}-01-01")))
            if tr.sum() < 30 or va.sum() == 0:
                continue
            mdl = Ridge(alpha=lam).fit(X[tr], Y[tr])
            err.extend((mdl.predict(X[va]) - Y[va]) ** 2)
        cv.append(float(np.mean(err)))
    k = int(np.argmin(cv))
    lam = float(GRID[k])
    ridge = Ridge(alpha=lam).fit(X, Y)
    fit = ridge.predict(X)
    gbm = GradientBoostingRegressor(**GBM).fit(X, Y)
    fit_g = gbm.predict(X)
    # CV MSE of always predicting the training mean, for reference
    base = []
    for V in range(2002, 2011):
        vs = pd.Timestamp(f"{V}-01-01")
        tr = (est.exit < vs).values
        va = np.asarray((est.index >= vs) & (est.index < pd.Timestamp(f"{V + 1}-01-01")))
        if tr.sum() >= 30 and va.sum():
            base.extend((Y[tr].mean() - Y[va]) ** 2)
    out = {
        "features": FEATURES, "mean": mu.round(6).to_dict(), "sd": sd.round(6).to_dict(),
        "ridge": {"lambda": lam, "intercept": float(ridge.intercept_),
                  "coef": dict(zip(FEATURES, np.round(ridge.coef_, 6).tolist())),
                  "cv_mse_grid": dict(zip([f"{g:.4g}" for g in GRID], np.round(cv, 4).tolist())),
                  "cv_mse_mean_forecast": round(float(np.mean(base)), 4),
                  "in_sample_r2": round(float(1 - np.var(Y - fit) / np.var(Y)), 4),
                  "m_p75": float(np.percentile(fit, 75)), "m_p90": float(np.percentile(fit, 90)),
                  "m_p100": float(np.max(fit)),
                  "share_positive_in_sample": round(float((fit > 0).mean()), 3)},
        "gbm": {"params": GBM, "m_p90": float(np.percentile(fit_g, 90)),
                "in_sample_r2": round(float(1 - np.var(Y - fit_g) / np.var(Y)), 4),
                "share_positive_in_sample": round(float((fit_g > 0).mean()), 3)},
        "n_est": int(len(Y)), "target_mean_est": round(float(Y.mean()), 3),
    }
    (res / "part4_frozen.json").write_text(json.dumps(out, indent=2) + "\n")
    pickle.dump({"ridge": ridge, "gbm": gbm, "mu": mu, "sd": sd}, open(data / "part4_models.pkl", "wb"))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
