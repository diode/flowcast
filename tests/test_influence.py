import numpy as np
import pandas as pd

from netstudy.influence import connected, lag_corr

EDGES = pd.DataFrame([("A_B", "A", "B"), ("B_A", "B", "A"), ("B_C", "B", "C")],
                     columns=["edge_id", "from_node", "to_node"])


def test_connected_skips_u_turns():
    assert connected(EDGES) == [("A_B", "B_C")]


def test_lag_corr_finds_the_delay_within_days():
    idx = pd.date_range("2026-10-05 06:05", periods=100, freq="5min").append(
        pd.date_range("2026-10-06 06:05", periods=100, freq="5min"))
    x = np.random.default_rng(0).normal(size=len(idx))
    y = np.roll(x, 2)                                          # b follows a 10 minutes later
    dev = pd.DataFrame({"a": x, "b": y}, index=idx)
    assert lag_corr(dev, "a", "b", 2) > 0.95
    assert abs(lag_corr(dev, "a", "b", 0)) < 0.3
