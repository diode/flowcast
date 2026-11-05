"""Phase 2 step 2: find incidents from the cameras alone (docs/phase2.md, section 3).

Per camera edge, only what a city would have: the loop count every 5 minutes. Edge counts alone
swing too much from day to day (a quiet neighbourhood looks like a closed road), so each edge's
count is compared with what the traffic leaving its start junction right now predicts:

  count     vehicles per 5-minute slot, and in the last 15 minutes (the slot and the two before it:
            known when it ends)
  expected  the junction's outgoing count x the edge's usual share of it (the other days, same day
            type and time of day)
  residual  (count - expected) / sqrt(expected); its usual range per edge (LOW_Q to HIGH_Q quantile
            on the other days, 06-22h) marks counts that are unusually low or high
  event     origin: a road gone dead. While its 5-minute counts stay at zero, the vehicles expected
            in those slots add up; the event is known once `min_missing` are missing, or once the
            road has been empty for LONG_SLOTS slots with LONG_MISSING missing (a closure stops all
            traffic; low shares happen anyway, zero rarely). Busy roads are found within minutes,
            quiet ones after 25 minutes. The event ends with the first vehicle.
  spread    the detour roads (the other roads leaving the origin's start junction, and those entering
            its end junction) whose 15-minute residual is outside its usual range for SPREAD_SLOTS slots in a row,
            from the event's start to SPREAD_AFTER_MIN after its end: `more` or `fewer`
Edges leaving a boundary point (in_*) are the only road from there, so they have no share to test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MISSING = 20.0          # vehicles expected while the road stayed empty: known at once
LONG_SLOTS, LONG_MISSING = 5, 5.0   # or: empty for 5 slots (25 minutes) with at least 5 missing
SPREAD_SLOTS = 2
SPREAD_AFTER_MIN = 60
LOW_Q, HIGH_Q = 0.025, 0.975    # usual range of the residual (checked: 0.005-0.1, docs/phase2.md)
HOURS = (6, 22)


def counts(obs: pd.DataFrame, slots: int = 3) -> pd.DataFrame:
    """Counts over the last `slots` 5-minute slots (3: 15 minutes), slots x camera edges."""
    w = obs.pivot(index="ts", columns="edge_id", values="flow").astype(float)
    w.index = pd.to_datetime(w.index)
    return w.sort_index().rolling(slots, min_periods=1).sum()


def _keys(index: pd.DatetimeIndex) -> pd.DataFrame:
    start = index - pd.Timedelta(minutes=5)                   # slot ending 00:00 belongs to the day before
    return pd.DataFrame({"day": start.date, "weekend": start.weekday >= 5,
                         "tod": start.hour * 12 + start.minute // 5}, index=index)


def _in_hours(index: pd.DatetimeIndex) -> np.ndarray:
    return ((index.hour >= HOURS[0]) & (index.hour < HOURS[1]))


def junction_out(c: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Per edge: the count of all camera edges leaving the same start junction (NaN at boundaries)."""
    e = edges[edges.edge_id.isin(c.columns)]
    out = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for node, group in e.groupby("from_node"):
        if not node.startswith("F_"):
            out[group.edge_id.tolist()] = np.repeat(c[group.edge_id].sum(axis=1).to_numpy()[:, None], len(group), 1)
    return out


class Normal:
    """What the other days say: usual shares and the usual range of residuals, per edge."""

    def __init__(self, c: pd.DataFrame, edges: pd.DataFrame, skip_day=None):
        self.edges = edges
        k = _keys(c.index)
        keep = (k.day != skip_day).to_numpy()
        keys = [k.weekend[keep], k.tod[keep]]
        self.share = c[keep].groupby(keys).sum() / junction_out(c, edges)[keep].groupby(keys).sum()
        other = c[keep]
        exp = self.expected(other)
        r = residual(other, exp)[_in_hours(other.index)]
        self.low, self.high = r.quantile(LOW_Q), r.quantile(HIGH_Q)

    def expected(self, day: pd.DataFrame) -> pd.DataFrame:
        k = _keys(day.index)
        p = self.share.reindex(pd.MultiIndex.from_arrays([k.weekend, k.tod]))[day.columns].to_numpy()
        return junction_out(day, self.edges) * p


def detour(edges: pd.DataFrame, origin: str) -> list[str]:
    """Roads a diversion around `origin` uses first: others out of its start, others into its end."""
    e = edges.set_index("edge_id")
    u, v = e.at[origin, "from_node"], e.at[origin, "to_node"]
    return [x for x in e.index if x != origin and not (e.at[x, "from_node"] == v and e.at[x, "to_node"] == u)
            and (e.at[x, "from_node"] == u or e.at[x, "to_node"] == v)]


def residual(c: pd.DataFrame, exp: pd.DataFrame) -> pd.DataFrame:
    return (c - exp) / np.sqrt(exp.clip(lower=1.0))


def _runs(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(first, last) slot of each stretch of SPREAD_SLOTS or more consecutive flagged slots."""
    f = flag.to_numpy()
    out, i = [], 0
    while i < len(f):
        if f[i]:
            j = i
            while j + 1 < len(f) and f[j + 1]:
                j += 1
            if j - i + 1 >= SPREAD_SLOTS:
                out.append((flag.index[i], flag.index[j]))
            i = j + 1
        else:
            i += 1
    return out


def dead_roads(day5: pd.DataFrame, exp5: pd.DataFrame, min_missing: float) -> list[tuple]:
    """(edge, first empty slot, slot when min_missing vehicles were missing, last empty slot)."""
    hours = _in_hours(day5.index)
    out = []
    for e in day5.columns:
        x, ex = day5[e].to_numpy(), exp5[e].to_numpy()
        missing, first, known = 0.0, None, None
        for i in range(len(x)):
            if hours[i] and x[i] < 0.5 and not np.isnan(ex[i]):
                first = i if first is None else first
                missing += ex[i]
                if known is None and (missing >= min_missing
                                      or (i - first + 1 >= LONG_SLOTS and missing >= LONG_MISSING)):
                    known = i
                continue
            if known is not None:
                out.append((e, day5.index[first], day5.index[known], day5.index[i - 1]))
            missing, first, known = 0.0, None, None
        if known is not None:
            out.append((e, day5.index[first], day5.index[known], day5.index[len(x) - 1]))
    return out


def detect(day5: pd.DataFrame, normal: Normal, min_missing: float = MIN_MISSING) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Events (origin, start, known, end) and their spread edges, for one day's 5-minute counts."""
    day = day5.rolling(3, min_periods=1).sum()
    r = residual(day, normal.expected(day))
    events, spread = [], []
    for origin, start, known, end in dead_roads(day5, normal.expected(day5), min_missing):
        eid = f"{origin}@{start:%Y-%m-%dT%H:%M}"
        events.append({"event_id": eid, "origin": origin, "start": start, "known": known, "end": end})
        win = (day.index >= start) & (day.index <= end + pd.Timedelta(minutes=SPREAD_AFTER_MIN))
        for e in detour(normal.edges, origin):
            if e not in day.columns or np.isnan(normal.high.get(e, np.nan)):
                continue
            s = r.loc[win, e]
            for effect, flag in (("more", s > normal.high[e]), ("fewer", s < normal.low[e])):
                runs = _runs(flag)
                if runs:
                    spread.append({"event_id": eid, "edge_id": e, "effect": effect, "first": runs[0][0],
                                   "delay_min": (runs[0][0] - start).total_seconds() / 60})
    return pd.DataFrame(events, columns=["event_id", "origin", "start", "known", "end"]), \
        pd.DataFrame(spread, columns=["event_id", "edge_id", "effect", "first", "delay_min"])
