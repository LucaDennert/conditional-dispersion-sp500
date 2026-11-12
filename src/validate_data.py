"""Sanity checks on the raw and built data (run after src.build_data).

Usage: python -m src.validate_data RAW_DIR DATA_DIR
Prints a report; every check states the expected value so a reader can judge it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(raw_dir: str, data_dir: str) -> None:
    raw, D = Path(raw_dir), Path(data_dir)
    m = pd.read_pickle(D / "members_daily.pkl")
    spx = pd.read_pickle(D / "spx_daily.pkl")
    w = pd.read_pickle(D / "weights_monthly.pkl")

    print("== CRSP S&P 500 members")
    n = m.groupby("date").size()
    print(f"names per day: {n.min()}-{n.max()} (expected ~500) | duplicate permno-date: {m.duplicated(['permno', 'date']).sum()} (expected 0)")
    tot = m.groupby("date")["cap"].sum() / 1e9
    print("total cap, $tn, 1999-12-31 / 2008-12-31 / 2024-12-31:",
          [round(tot.asof(pd.Timestamp(d)), 1) for d in ["1999-12-31", "2008-12-31", "2024-12-31"]],
          "(expected ~12 / ~8 / ~50)")
    m = m.sort_values(["permno", "date"])
    m["cap_lag"] = m.groupby("permno")["cap"].shift()
    x = m.dropna(subset=["cap_lag", "ret"])
    pr = (x["ret"] * x["cap_lag"]).groupby(x["date"]).sum() / x.groupby("date")["cap_lag"].sum()
    j = pd.concat([pr.rename("crsp"), spx["spx_ret"]], axis=1).dropna()
    d = j["crsp"] - j["spx_ret"]
    print(f"cap-weighted CRSP total return vs SPX price return: corr {j.corr().iloc[0, 1]:.4f} (expected >0.999), "
          f"mean gap {d.mean() * 252:.2%}/yr (expected ~ dividend yield), tracking error {d.std() * np.sqrt(252):.2%}/yr")
    print(f"weights sum to one each month: {np.allclose(w.groupby('date')['weight'].sum(), 1)}")
    big = m[m["ret"].abs() > 0.5]
    print(f"daily |return| > 50%: {len(big)} member-days, {big['date'].dt.year.isin([2008, 2009]).mean():.0%} in 2008-09 (crisis failures/bailouts)")

    print("\n== SPX and VIX (IvyDB)")
    for dt, ref in [("2008-10-10", "899.22 / 69.95"), ("2020-03-23", "2237.40 / 61.59"), ("2025-08-29", "6460.26 / 15.36")]:
        print(f"{dt}: SPX {spx['spx'].asof(pd.Timestamp(dt))}, VIX {spx['vix'].asof(pd.Timestamp(dt))} (published closes {ref})")
    print(f"VIX max {spx['vix'].max()} on {spx['vix'].idxmax().date()} (published record close 82.69 on 2020-03-16)")

    print("\n== SPX standardized options")
    s = pd.read_csv(raw / "optionm_stdopd_spx.csv.gz", parse_dates=["date"])
    s91 = s[s["days"] == 91].pivot_table(index="date", columns="cp_flag", values="impl_volatility")
    print(f"91d call vs put IV, mean abs gap: {(s91['C'] - s91['P']).abs().mean() * 100:.2f} vol pts (expected < 1)")
    y = spx[["iv30", "vix"]].dropna()
    print(f"corr(30d ATM IV, VIX) {y['iv30'].corr(y['vix']):.3f} (expected >0.98); VIX - ATM IV {(y['vix'] - 100 * y['iv30']).mean():.2f} pts (expected positive: skew)")

    print("\n== Zero curve")
    z = pd.read_csv(raw / "optionm_zerocd.csv.gz", parse_dates=["date"])
    z3 = z[z["days"].between(60, 120)].groupby("date")["rate"].mean()
    print("3m zero rate %, 2008-12-31 / 2015-06-30 / 2023-10-31:",
          [round(z3.asof(pd.Timestamp(d)), 2) for d in ["2008-12-31", "2015-06-30", "2023-10-31"]], "(LIBOR/SOFR-based curve)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
