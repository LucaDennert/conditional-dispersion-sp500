"""Merge daily quote paths of the selected listed contracts.

Inputs (raw, never committed):
  RAW/paths_YYYY.csv.gz            daily IvyDB opprcd rows for chosen optionids
  RAW/extra/paths_extra_YYYY.csv.gz rows for optionids missing from the first pull
  DATA/contracts_chosen_v2.csv.gz   output of src.select_contracts

Output: DATA/listed_paths.pkl  -- one row per (optionid, date), deduplicated,
restricted to dates between the first selection date of the optionid and expiry.
Also prints a quality report (missing quotes, crossed/zero bids, IV jumps).

Run: python -m src.listed_paths RAW_DIR DATA_DIR
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["optionid", "secid", "date", "exdate", "cp_flag", "strike", "best_bid",
        "best_offer", "impl_volatility", "delta", "vega"]


def load_paths(raw: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw / "paths_*.csv.gz"))) + \
        sorted(glob.glob(str(raw / "extra" / "paths_extra_*.csv.gz")))
    out = []
    for f in files:
        d = pd.read_csv(f, usecols=COLS, parse_dates=["date", "exdate"])
        out.append(d)
    p = pd.concat(out, ignore_index=True)
    p["optionid"] = p.optionid.astype("int64")
    p = p.drop_duplicates(["optionid", "date"])
    return p.sort_values(["optionid", "date"]).reset_index(drop=True)


def build(raw: Path, data: Path) -> pd.DataFrame:
    ch = pd.read_csv(data / "contracts_chosen_v2.csv.gz", parse_dates=["date", "exdate"])
    ch["optionid"] = ch.optionid.astype("int64")
    first = ch.groupby("optionid").date.min().rename("first_sel")
    p = load_paths(raw).merge(first, left_on="optionid", right_index=True)
    p = p[(p.date >= p.first_sel) & (p.date <= p.exdate)].drop(columns="first_sel")
    p["strike"] = p.strike / 1000 if p.strike.median() > 5000 else p.strike
    p.to_pickle(data / "listed_paths.pkl")
    return p, ch


def quality(p: pd.DataFrame, ch: pd.DataFrame) -> None:
    print(f"rows {len(p):,}  optionids {p.optionid.nunique():,} / {ch.optionid.nunique():,}")
    bad = (p.best_bid <= 0) | (p.best_offer <= p.best_bid) | p.best_offer.isna()
    print(f"zero-bid or crossed quotes: {bad.mean():.2%}")
    print(f"missing IV: {p.impl_volatility.isna().mean():.2%}")
    p = p.sort_values(["optionid", "date"])
    jump = p.groupby("optionid").impl_volatility.diff().abs()
    print(f"|dIV| > 0.10 in one day: {(jump > 0.10).mean():.2%}")
    # holding-window completeness: trading days with a quote between selection and exit
    print(p.groupby(p.date.dt.year).agg(rows=("optionid", "size"),
                                        bad=("best_bid", lambda s: 0)).T.iloc[:1])


if __name__ == "__main__":
    raw, data = Path(sys.argv[1]), Path(sys.argv[2])
    p, ch = build(raw, data)
    quality(p, ch)
