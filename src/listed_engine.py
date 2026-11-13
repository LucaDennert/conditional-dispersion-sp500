"""Listed-contract P&L engine (pre-registration section 7, amendment A2).

For every entry date (first trading day of the month) and every selected straddle (SPX and the
top-50 members), simulate one LONG straddle:
    entry       at the ask (cost = half-spread vs mid), day 0
    marking     daily at mid
    hedge       daily at the close with IvyDB delta (call + put), short delta x underlying
    financing   option premium and hedge at the 91-day zero rate
    hedge cost  0.5 bp (index) / 1 bp (stocks) of |change in hedge notional|, incl. unwind
    exit        10 trading days before expiry, at the bid (cost = half-spread)
    splits      prices and deltas rescaled by K_entry / K_t when IvyDB adjusts the strike
    missing     a straddle without a valid quote is carried forward for at most 3 trading days;
                after that it is closed at the last available price.
Everything is normalized per vol point of the straddle's vega at entry, so legs combine linearly:
a short SPX straddle is minus the long one with the same costs.

Output: DATA/engine_legs.pkl, one row per (entry, secid, day) with
    mid   mid-to-mid option P&L + hedge P&L + financing (normalized, cost-free)
    hc    hedge cost of that day (normalized)
    hs    half-spread of the straddle that day (normalized), used for entry/exit/stop costs
    hpos  hedge notional held after that day's rebalance (normalized), for stop unwinds
    live  1 while the position is open (day <= exit day of that straddle)
Run: python -m src.listed_engine DATA_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.signals import entry_dates
from src.simple_dispersion import rate_91

EXIT_BUFFER = 10
CARRY = 3
SPX_SECID = 108105
HEDGE_BP = {"index": 0.5e-4, "stock": 1e-4}
DT = 1 / 252.0


def simulate_straddle(mid, hs, delta, S, R, r, bp):
    """Arrays over trading days 0..n (n = scheduled exit). Returns per-day arrays and exit index."""
    n = len(mid) - 1
    gap = np.isnan(mid)
    e = n
    if gap.any():
        e = int(np.argmax(gap)) - 1          # last day with a (carried) quote
    out_mid = np.zeros(n + 1)
    out_hc = np.zeros(n + 1)
    hpos = np.zeros(n + 1)
    h_prev = 0.0
    for d in range(0, e + 1):
        h = delta[d] * S[d] if d < e else 0.0   # unwind on the exit day
        out_hc[d] = abs(h - h_prev) * bp
        hpos[d] = h
        if d >= 1:
            hold = delta[d - 1] * S[d - 1]
            out_mid[d] = (mid[d] - mid[d - 1]) - hold * (R[d] - r * DT) - r * mid[d - 1] * DT
        h_prev = h
    live = np.zeros(n + 1)
    live[: e + 1] = 1
    hs_out = np.where(live > 0, hs, np.nan)
    return out_mid, out_hc, hs_out, hpos, live, e


def build(data: Path) -> pd.DataFrame:
    ch = pd.read_csv(data / "contracts_chosen_v2.csv.gz", parse_dates=["date", "exdate"])
    ch["optionid"] = ch.optionid.astype("int64")
    paths = pd.read_pickle(data / "listed_paths.pkl")
    spx = pd.read_pickle(data / "spx_daily.pkl")
    mem = pd.read_pickle(data / "members_daily.pkl").dropna(subset=["secid"])
    z = pd.read_pickle(data / "zero_curve.pkl")
    cal = spx.index
    pos = {d: i for i, d in enumerate(cal)}

    S_m = mem.assign(p=mem.prc.abs()).pivot_table(index="date", columns="secid", values="p").reindex(cal).ffill(limit=5)
    R_m = mem.pivot_table(index="date", columns="secid", values="ret").reindex(cal)
    S_i = spx["spx"]
    Rx_i = spx["spx"].pct_change()

    q = paths.set_index(["optionid", "date"])[["best_bid", "best_offer", "delta", "strike"]]
    by_opt = {k: g.droplevel(0) for k, g in q.groupby(level=0)}

    rows = []
    for t0 in entry_dates(spx):
        c = ch[ch.date == t0]
        if c.empty:
            continue
        i0 = pos[t0]
        r = rate_91(z, t0)
        for sid, g in c.groupby("secid"):
            if len(g) != 2 or set(g.cp_flag) != {"C", "P"}:
                continue
            vega = float(g.vega.sum()) / 100.0
            if not np.isfinite(vega) or vega <= 0.005:
                continue
            exd = g.exdate.iloc[0]
            k = cal.searchsorted(exd, side="left")
            ie = k - EXIT_BUFFER
            if ie <= i0 or ie >= len(cal):
                continue
            days = cal[i0: ie + 1]
            mids, hss, dels = [], [], []
            for oid in g.optionid:
                o = by_opt.get(oid)
                if o is None:
                    break
                o = o.reindex(days)
                # stock splits: IvyDB keeps the optionid and adjusts the strike (contract covers more
                # shares). Express prices and delta per ORIGINAL contract: factor = K_entry / K_t.
                k_t = o.strike.ffill().bfill()
                adj = (k_t.iloc[0] / k_t).fillna(1.0)
                # a zero bid with a positive ask is a quote (option close to worthless), not a missing price
                ok = (o.best_bid >= 0) & (o.best_offer > 0) & (o.best_offer >= o.best_bid)
                m = ((o.best_bid + o.best_offer) / 2 * adj).where(ok)
                h = ((o.best_offer - o.best_bid) / 2 * adj).where(ok)
                mids.append(m)
                hss.append(h)
                dels.append(o.delta * adj)
            if len(mids) != 2:
                continue
            mid = (mids[0] + mids[1])
            hs = (hss[0] + hss[1])
            if not np.isfinite(mid.iloc[0]):
                continue
            mid = mid.ffill(limit=CARRY).values
            hs = hs.ffill(limit=CARRY).values
            delta = (dels[0].ffill() + dels[1].ffill()).ffill().fillna(0.0).values
            if sid == SPX_SECID:
                f91 = spx["fwd91"].iat[i0]
                qy = r - np.log(f91 / S_i.iat[i0]) / (91 / 365) if np.isfinite(f91) else 0.0
                S = S_i.iloc[i0: ie + 1].values
                R = (Rx_i.iloc[i0: ie + 1] + qy * DT).fillna(0.0).values
                bp = HEDGE_BP["index"]
            else:
                if sid not in S_m.columns:
                    continue
                S = S_m[sid].iloc[i0: ie + 1].ffill().bfill().values
                R = R_m[sid].iloc[i0: ie + 1].fillna(0.0).values
                bp = HEDGE_BP["stock"]
            if not np.isfinite(S).all():
                continue
            om, hc, hso, hpos, live, e = simulate_straddle(mid, hs, delta, S, R, r, bp)
            n = len(days)
            rows.append(pd.DataFrame({
                "entry": t0, "secid": int(sid), "day": np.arange(n), "date": days,
                "mid": om / vega, "hc": hc / vega, "hs": hso / vega, "hpos": hpos / vega, "live": live,
                "exit_day": e, "sched_exit": n - 1,
            }))
    out = pd.concat(rows, ignore_index=True)
    for col in ["mid", "hc", "hs", "hpos", "live"]:
        out[col] = out[col].astype("float32")
    return out


if __name__ == "__main__":
    data = Path(sys.argv[1])
    df = build(data)
    df.to_pickle(data / "engine_legs.pkl")
    print(len(df), df.entry.nunique(), df.groupby("entry").secid.nunique().describe().round(1).to_dict())
