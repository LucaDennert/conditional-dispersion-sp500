"""Turn the raw WRDS downloads into clean panels used by every later step.

Usage:  python -m src.build_data RAW_DIR OUT_DIR

RAW_DIR must contain the files produced by notebooks/01 or by the equivalent WRDS web queries:
  crsp_sp500_constituents_daily.csv.gz   CRSP S&P 500 constituents (CIZ), daily, with DlyCap/DlyRet
  optionm_crsp_link.csv                  WRDS OptionMetrics-CRSP link (secid, sdate, edate, PERMNO, score)
  optionm_stdopd_spx.csv.gz              IvyDB standardized options, SPX (secid 108105)
  optionm_stdopd_members*.csv.gz         IvyDB standardized options, members, days in {30, 91}
  optionm_secprd_spx_vix.csv.gz          IvyDB security prices, SPX and VIX
Outputs (pickle; this environment has no parquet engine) in OUT_DIR: members_daily, weights_monthly, spx_daily, iv_members.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

END = pd.Timestamp("2025-08-29")   # last IvyDB date available on WRDS
SPX = 108105


def load_link(raw: Path) -> pd.DataFrame:
    lk = pd.read_csv(raw / "optionm_crsp_link.csv", parse_dates=["sdate", "edate"])
    lk.columns = [c.lower() for c in lk.columns]
    return lk[lk["score"] == 1][["permno", "secid", "sdate", "edate"]]


def build_members(raw: Path, link: pd.DataFrame) -> pd.DataFrame:
    c = pd.read_csv(
        raw / "crsp_sp500_constituents_daily.csv.gz",
        usecols=["PERMNO", "DlyCalDt", "DlyCap", "DlyRet"],
        parse_dates=["DlyCalDt"], na_values=["NA", "", "."],
    ).rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyCap": "cap", "DlyRet": "ret"})
    c = c[c["date"] <= END].drop_duplicates(["permno", "date"])
    c["logret"] = np.log1p(c["ret"])
    # attach the OptionMetrics secid valid on each date
    m = c.merge(link, on="permno", how="inner")
    m = m[(m["date"] >= m["sdate"]) & (m["date"] <= m["edate"])]
    m = m.drop_duplicates(["permno", "date"])[["permno", "date", "secid"]]
    # keep every CRSP member-day; secid is NaN where no valid link exists
    return c.merge(m, on=["permno", "date"], how="left")


def build_weights(members: pd.DataFrame) -> pd.DataFrame:
    """Market-cap weights of all index members on the last trading day of each month."""
    d = members.dropna(subset=["cap"])
    last = d.groupby(d["date"].dt.to_period("M"))["date"].transform("max")
    me = d[d["date"] == last].copy()
    me["weight"] = me["cap"] / me.groupby("date")["cap"].transform("sum")
    me["month"] = me["date"].dt.to_period("M")
    return me[["month", "date", "permno", "secid", "cap", "weight"]]


def build_spx(raw: Path) -> pd.DataFrame:
    px = pd.read_csv(raw / "optionm_secprd_spx_vix.csv.gz", parse_dates=["date"])
    spx = px[px["ticker"] == "SPX"].set_index("date")[["close", "return"]].rename(columns={"close": "spx", "return": "spx_ret"})
    vix = px[px["ticker"] == "VIX"].set_index("date")[["close"]].rename(columns={"close": "vix"})
    std = pd.read_csv(raw / "optionm_stdopd_spx.csv.gz", parse_dates=["date"])
    iv = std[std["days"].isin([30, 91])].groupby(["date", "days"])["impl_volatility"].mean().unstack()
    iv.columns = [f"iv{int(c)}" for c in iv.columns]
    fwd = std[std["days"] == 91].groupby("date")["forward_price"].first().rename("fwd91")
    out = spx.join(vix, how="left").join(iv, how="left").join(fwd, how="left")
    out["spx_logret"] = np.log1p(out["spx_ret"])
    return out[out.index <= END]


def build_iv_members(raw: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw / "optionm_stdopd_members*.csv.gz")))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        for s in pd.read_csv(f, usecols=["secid", "date", "days", "cp_flag", "impl_volatility"],
                             parse_dates=["date"], chunksize=2_000_000):
            s = s[s["days"].isin([30, 91])]
            parts.append(s.groupby(["secid", "date", "days"])["impl_volatility"].mean())
    # call/put average; a (secid, date, days) split across chunks is re-averaged by its parts
    iv = pd.concat(parts).groupby(level=[0, 1, 2]).mean().unstack()
    iv.columns = [f"iv{int(c)}" for c in iv.columns]
    return iv.reset_index()


def main(raw_dir: str, out_dir: str) -> None:
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    link = load_link(raw)
    members = build_members(raw, link)
    members.to_pickle(out / "members_daily.pkl")
    weights = build_weights(members)
    weights.to_pickle(out / "weights_monthly.pkl")
    spx = build_spx(raw)
    spx.to_pickle(out / "spx_daily.pkl")
    ivm = build_iv_members(raw)
    if len(ivm):
        ivm.to_pickle(out / "iv_members.pkl")

    # data-quality report
    per_day = members.groupby("date").size()
    w_nosecid = weights[weights["secid"].isna()].groupby("date")["weight"].sum()
    print(f"members_daily: {len(members):,} rows, {members['permno'].nunique()} permnos, "
          f"{per_day.min()}-{per_day.max()} names per day")
    print(f"missing returns: {members['ret'].isna().mean():.4%}   missing cap: {members['cap'].isna().mean():.4%}")
    print(f"weight without an OptionMetrics secid (month-end): mean {w_nosecid.mean() if len(w_nosecid) else 0:.4%}, "
          f"max {w_nosecid.max() if len(w_nosecid) else 0:.4%}")
    print(f"spx_daily: {spx.index.min().date()} to {spx.index.max().date()}, {len(spx)} days, "
          f"missing iv91 {spx['iv91'].isna().mean():.2%}, missing vix {spx['vix'].isna().mean():.2%}")
    if len(ivm):
        print(f"iv_members: {len(ivm):,} rows, {ivm['secid'].nunique()} secids")


if __name__ == "__main__":
    main(*sys.argv[1:3])
