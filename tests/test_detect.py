import numpy as np
import pandas as pd

from netstudy.detect import Normal, detect, detour, dead_roads

# Junction A splits traffic from in_A over A_B (60%) and A_C (40%); B_C joins the two ends.
EDGES = pd.DataFrame([
    ("in_A", "F_A", "A"), ("A_B", "A", "B"), ("A_C", "A", "C"), ("B_C", "B", "C"), ("C_B", "C", "B"),
], columns=["edge_id", "from_node", "to_node"])
IDX = pd.date_range("2026-10-05 00:05", periods=288 * 3, freq="5min")   # Mon-Wed


def normal_days() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = rng.poisson(20, len(IDX)).astype(float)
    ab = rng.binomial(n.astype(int), 0.6).astype(float)
    return pd.DataFrame({"in_A": n, "A_B": ab, "A_C": n - ab, "B_C": 5.0, "C_B": 5.0}, index=IDX)


def test_detour_roads():
    assert detour(EDGES, "A_B") == ["A_C", "C_B"]          # other road out of A; the other road into B
    assert detour(EDGES, "B_C") == ["A_C"]                 # C_B is the reverse, not a detour


def test_dead_road_needs_enough_missing_vehicles():
    idx = IDX[:20]
    day = pd.DataFrame({"x": [3.0] * 8 + [0.0] * 6 + [3.0] * 6}, index=idx + pd.Timedelta(hours=8))
    exp = pd.DataFrame({"x": 4.0}, index=day.index)
    (edge, first, known, last), = dead_roads(day, exp, min_missing=10)
    assert edge == "x" and first == day.index[8] and known == day.index[10] and last == day.index[13]
    (_, _, known, _), = dead_roads(day, exp / 4, min_missing=10)   # 1 expected per slot: only after
    assert known == day.index[12]                                   # 5 empty slots (LONG_SLOTS)
    assert dead_roads(day, exp / 8, min_missing=10) == []          # 0.5 per slot: never enough


def test_closure_found_with_its_detour():
    base = normal_days()
    day = base[base.index.date == pd.Timestamp("2026-10-07").date()].copy()
    closed = (day.index.hour == 8)                                  # A_B closed 08:00-09:00
    day.loc[closed, "A_C"] += day.loc[closed, "A_B"]
    day.loc[closed, "A_B"] = 0.0
    ev, sp = detect(day, Normal(base.rolling(3, min_periods=1).sum(), EDGES, skip_day=pd.Timestamp("2026-10-07").date()))
    assert ev.origin.tolist() == ["A_B"] and ev.start.iloc[0].hour == 8
    assert (ev.known.iloc[0] - ev.start.iloc[0]) <= pd.Timedelta(minutes=10)  # ~12 vehicles expected per slot
    assert ("A_C", "more") in set(zip(sp.edge_id, sp.effect))
