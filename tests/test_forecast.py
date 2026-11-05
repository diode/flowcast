import numpy as np
import pandas as pd

from netstudy.forecast import NetworkForecaster, camera_series, neighbours, settings

EDGES = pd.DataFrame([
    ("in_A", "F_A", "A", "cam1"), ("A_B", "A", "B", "cam2"), ("B_A", "B", "A", "cam3"),
    ("B_C", "B", "C", "cam4"), ("A_C", "A", "C", None),
], columns=["edge_id", "from_node", "to_node", "camera_id"])


def test_neighbours_skip_u_turns_and_edges_without_camera():
    nb = neighbours(EDGES)
    assert nb["A_B"] == {"up": ["in_A"], "down": ["B_C"], "junction": ["A_B"]}   # B_A: U-turn; A_C: no camera
    assert nb["in_A"]["junction"] == []                                          # starts at a boundary


def test_network_features_add_neighbours_to_v1():
    idx = pd.date_range("2026-10-01 00:05", periods=400, freq="5min")
    rng = np.random.default_rng(0)
    w = {v: pd.DataFrame(rng.poisson(5, (len(idx), 4)).astype(float), index=idx,
                         columns=["in_A", "A_B", "B_A", "B_C"]) for v in ("flow", "pcu", "load_ratio")}
    s = camera_series("A_B", w, neighbours(EDGES)["A_B"])
    X = NetworkForecaster(settings(), "A_B")._features(s, "flow", 3)
    assert {"flow_now", "up|in_A_now", "up|in_A_lag2", "down|B_C_mean3", "jout_now"} <= set(X.columns)
    assert X["up|in_A_lag1"].iloc[10] == s["up|in_A"].iloc[9]
