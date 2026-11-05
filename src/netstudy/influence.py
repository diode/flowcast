"""Phase 2 step 3: how strongly each road follows its neighbours in everyday traffic (docs/phase2.md).

  deviation  each camera edge's 5-minute count minus its usual count (median over the days of the
             same day type at the same time of day), 06-22h
  pairs      connected edges: `up` ends where `down` starts (no U-turns)
  lag        correlation of up's deviation now with down's deviation L slots later, within each day,
             for L = -3..6 (-15 to +30 minutes); `lag_min` is where it peaks
  control    the same correlation at lag 0 between edges that share no junction: how much any two
             edges move together because the whole town is busier or quieter
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .detect import _in_hours, _keys

LAGS = range(-3, 7)


def deviations(c5: pd.DataFrame) -> pd.DataFrame:
    k = _keys(c5.index)
    dev = c5 - c5.groupby([k.weekend, k.tod]).transform("median")
    return dev[_in_hours(dev.index)]


def connected(edges: pd.DataFrame) -> list[tuple[str, str]]:
    e = edges.set_index("edge_id")
    return [(a, b) for a in e.index for b in e.index
            if e.at[a, "to_node"] == e.at[b, "from_node"] and e.at[b, "to_node"] != e.at[a, "from_node"]]


def lag_corr(dev: pd.DataFrame, a: str, b: str, lag: int) -> float:
    """corr(a at t, b at t + lag slots), pairs taken within the same day only."""
    day = _keys(dev.index).day
    xs, ys = [], []
    for d in np.unique(day):
        x, y = dev[a].to_numpy()[day == d], dev[b].to_numpy()[day == d]
        if lag >= 0:
            xs.append(x[:len(x) - lag]), ys.append(y[lag:])
        else:
            xs.append(x[-lag:]), ys.append(y[:lag])
    return float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])


def influence(c5: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """(one row per connected pair: kind, r at each lag, peak lag and r; control correlations)."""
    dev = deviations(c5)
    e = edges.set_index("edge_id")
    rows = []
    for a, b in connected(edges[edges.edge_id.isin(dev.columns)]):
        r = {lag: lag_corr(dev, a, b, lag) for lag in LAGS}
        peak = max(r, key=r.get)
        rows.append({"up": a, "down": b, "via": e.at[a, "to_node"],
                     "kind": f"{e.at[a, 'kind']}->{e.at[b, 'kind']}",
                     "r0": round(r[0], 3), "r_5min": round(r[1], 3), "lag_min": 5 * peak, "r_peak": round(r[peak], 3)})
    ends = {x: {e.at[x, "from_node"], e.at[x, "to_node"]} for x in dev.columns}
    control = pd.Series([lag_corr(dev, a, b, 0) for a, b in itertools.combinations(dev.columns, 2)
                         if not ends[a] & ends[b]])
    return pd.DataFrame(rows).sort_values("r0", ascending=False, ignore_index=True), control
