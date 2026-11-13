"""Pre-registered contract selection (PREREGISTRATION.md, section 7), applied to the option chains
pulled on each entry date.

For SPX and each top-50 member: standard monthly expiry (IvyDB expiry_indicator empty) closest to
91 calendar days within 60-130 days; strike closest to spot with both call and put quoted
(bid > 0, ask > bid); if several contracts qualify (e.g. different roots), the tighter spread.

Usage: python -m src.select_contracts CHAIN_DIR TOP50_JSON_GZ OUT_CSV
"""
from __future__ import annotations

import glob
import gzip
import json
import sys

import pandas as pd

SPX = 108105


def select(chain: pd.DataFrame, top50: dict) -> pd.DataFrame:
    c = chain[(chain["best_bid"] > 0) & (chain["best_offer"] > chain["best_bid"])].copy()
    c = c[c["expiry_indicator"].isna()]
    keep = pd.DataFrame([(pd.Timestamp(d), s) for d, ids in top50.items() for s in ids], columns=["date", "secid"])
    c = c.merge(keep.assign(_top=1), on=["date", "secid"], how="left")
    c = c[(c["secid"] == SPX) | (c["_top"] == 1)].drop(columns="_top")
    c["spread"] = c["best_offer"] - c["best_bid"]
    # best quote per (name, date, expiry, strike, side) across roots
    c = c.sort_values("spread").drop_duplicates(["secid", "date", "exdate", "strike", "cp_flag"])
    c = c[c.groupby(["secid", "date", "exdate", "strike"])["cp_flag"].transform("nunique") == 2]
    c["dte_gap"] = ((c["exdate"] - c["date"]).dt.days - 91).abs()
    c = c[c["dte_gap"] == c.groupby(["secid", "date"])["dte_gap"].transform("min")]
    c["k_gap"] = (c["strike"] - c["spot"]).abs()
    c = c[c["k_gap"] == c.groupby(["secid", "date"])["k_gap"].transform("min")]
    c["pair_spread"] = c.groupby(["secid", "date", "exdate", "strike"])["spread"].transform("sum")
    best = c.sort_values("pair_spread").drop_duplicates(["secid", "date"])[["secid", "date", "exdate", "strike"]]
    return c.merge(best, on=["secid", "date", "exdate", "strike"])


if __name__ == "__main__":
    chain_dir, top50_path, out = sys.argv[1:4]
    chain = pd.concat([pd.read_csv(f, parse_dates=["date", "exdate"]) for f in sorted(glob.glob(f"{chain_dir}/chain_*.csv.gz"))],
                      ignore_index=True)
    top50 = json.load(gzip.open(top50_path, "rt"))
    ch = select(chain, top50)
    ch.to_csv(out, index=False)
    print(len(ch), "contracts;", ch.groupby("date")["secid"].nunique().describe().round(1).to_dict())
    print("legs per name-date:", ch.groupby(["date", "secid"]).size().value_counts().to_dict())
