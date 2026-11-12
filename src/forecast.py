"""Correlation forecast (PREREGISTRATION.md, section 5).

Fisher-z HAR regression estimated by OLS on monthly entry dates of the estimation sample,
Newey-West errors with 3 lags. Only entry dates whose 53-day target window ends inside the
estimation sample are used, so no test-sample information enters the estimates.
Coefficients are then frozen and applied to every date.

Usage: python -m src.forecast DATA_DIR RESULTS_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.corr import fisher_z, inv_fisher_z
from src.signals import HOLD
from src.stats import ols_newey_west

EST_END = pd.Timestamp("2010-12-31")
TEST_START = pd.Timestamp("2011-01-01")
NW_LAGS = 3

SPECS = {
    "primary": ["rc21", "rc63", "rc252", "ic"],   # pre-registered
    "no_ic": ["rc21", "rc63", "rc252"],           # reported only
    "pure_rc21": None,                            # RCfcst = RC21, reported only
    "pure_rc63": None,                            # RCfcst = RC63, reported only
}


def estimation_mask(sig: pd.DataFrame, spx_dates: pd.DatetimeIndex) -> pd.Series:
    pos = pd.Series(np.arange(len(spx_dates)), index=spx_dates)
    end_idx = pos.reindex(sig.index) + HOLD
    end_date = pd.Series(spx_dates[np.minimum(end_idx.values, len(spx_dates) - 1)], index=sig.index)
    return (end_date <= EST_END) & sig["rc_fwd"].notna()


def fit(sig: pd.DataFrame, cols: list[str], mask: pd.Series):
    d = sig.loc[mask, cols + ["rc_fwd"]].dropna()
    y = fisher_z(d["rc_fwd"].values)
    X = fisher_z(d[cols].values)
    res = ols_newey_west(y, X, NW_LAGS, names=cols)
    fitted = inv_fisher_z(np.column_stack([np.ones(len(X)), X]) @ res.params)
    rmse = float(np.sqrt(np.mean((fitted - d["rc_fwd"].values) ** 2)))
    return res, rmse, len(d)


def predict(sig: pd.DataFrame, cols: list[str], params: np.ndarray) -> pd.Series:
    X = fisher_z(sig[cols].values)
    return pd.Series(inv_fisher_z(np.column_stack([np.ones(len(X)), X]) @ params), index=sig.index)


def main(data_dir: str, results_dir: str) -> None:
    data, out = Path(data_dir), Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    sig = pd.read_pickle(data / "signals_monthly.pkl")
    spx_dates = pd.read_pickle(data / "spx_daily.pkl").index
    mask = estimation_mask(sig, spx_dates)

    frozen, lines = {}, []
    for name, cols in SPECS.items():
        if cols is None:
            src = name.split("_")[1]
            sig[f"fcst_{name}"] = sig[src]
            e = sig.loc[mask, [src, "rc_fwd"]].dropna()
            rmse = float(np.sqrt(np.mean((e[src] - e["rc_fwd"]) ** 2)))
            frozen[name] = {"rmse_est": rmse}
            lines.append(f"[{name}] RCfcst = {src}; RMSE (estimation) {rmse:.4f}")
            continue
        res, rmse, n = fit(sig, cols, mask)
        sig[f"fcst_{name}"] = predict(sig, cols, res.params)
        frozen[name] = {"cols": cols, "params": res.params.tolist(), "se_nw": res.se.tolist(),
                        "r2_z": res.r2, "rmse_est": rmse, "n_est": n}
        lines.append(f"[{name}] n = {n}, RMSE (correlation units, estimation) {rmse:.4f}\n{res.summary()}")

    # out-of-sample forecast accuracy, reported for every spec
    test = sig.index >= TEST_START
    for name in SPECS:
        e = sig.loc[test, [f"fcst_{name}", "rc_fwd"]].dropna()
        frozen[name]["rmse_test"] = float(np.sqrt(np.mean((e.iloc[:, 0] - e["rc_fwd"]) ** 2)))
    lines.append("Out-of-sample RMSE (2011-2025): " + ", ".join(f"{k} {v['rmse_test']:.4f}" for k, v in frozen.items()))

    sig["cp"] = sig["ic"] - sig["fcst_primary"]
    sig.to_pickle(data / "signals_monthly.pkl")
    (out / "forecast_frozen.json").write_text(json.dumps(frozen, indent=2))
    (out / "forecast_estimation.txt").write_text("\n\n".join(lines) + "\n")
    print("\n\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
