"""Monthly ladder of tranches: one tranche opened per entry date, each with one-third of the vega
budget times its size. Returns daily and monthly P&L per unit of vega budget (vol points)."""
import numpy as np
import pandas as pd


def ladder_daily(T, sizes: pd.Series, start, end, cal, **kw):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    parts, expo, n = [], [], 0
    for t0 in T.entries():
        if not (a <= t0 <= b):
            continue
        z = float(sizes.get(t0, 0.0)) if sizes is not None else 1.0
        if not np.isfinite(z) or z <= 0:
            continue
        kk = dict(kw)
        if callable(kk.get("long_mult")):
            kk["long_mult"] = kk["long_mult"](t0)
        d = T.daily(t0, legs=True, **kk) * (z / 3.0)
        d = d[d.index <= b]
        parts.append(d)
        expo.append(pd.Series(z / 3.0, index=d.index))
        n += 1
    idx = cal[(cal >= a) & (cal <= b)]
    if not parts:
        z = pd.DataFrame(0.0, index=idx, columns=["long", "short", "pnl"])
        return z, pd.Series(0.0, index=idx), 0
    D = pd.concat(parts).groupby(level=0).sum().reindex(idx, fill_value=0.0)
    X = pd.concat(expo).groupby(level=0).sum().reindex(idx, fill_value=0.0)
    return D, X, n


def to_monthly(D):
    return D.groupby(D.index.to_period("M")).sum()
