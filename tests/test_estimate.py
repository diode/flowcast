import pandas as pd
import pytest

from netstudy.estimate import determined, estimate_flow, junction_rows, level_of

# Two junctions joined by a two-way road, each with its neighbourhood road (as in maple_falls).
EDGES = pd.DataFrame([
    ("in_J1", "F_J1", "J1", "in"), ("out_J1", "J1", "F_J1", "out"),
    ("in_J2", "F_J2", "J2", "in"), ("out_J2", "J2", "F_J2", "out"),
    ("J1_J2", "J1", "J2", "road"), ("J2_J1", "J2", "J1", "road"),
], columns=["edge_id", "from_node", "to_node", "kind"])
# Vehicles per slot that balance at both junctions: J1 10 + 5 in = 10 + 5 out, J2 10 + 5 in = 5 + 10 out
FLOW = pd.DataFrame({"in_J1": 10.0, "out_J1": 5.0, "in_J2": 5.0, "out_J2": 10.0, "J1_J2": 10.0, "J2_J1": 5.0},
                    index=[f"t{i}" for i in range(6)])


def test_junction_rows_and_determined():
    js, a = junction_rows(EDGES)
    assert js == ["J1", "J2"]
    assert a.loc["J1", "in_J1"] == 1 and a.loc["J1", "J1_J2"] == -1 and a.loc["J2", "J1_J2"] == 1
    assert determined(a[["J1_J2"]].to_numpy()).tolist() == [True]                # only unknown: solved
    assert determined(a[["J1_J2", "J2_J1"]].to_numpy()).tolist() == [False, False]  # both ways: only the difference


def test_m1_exact_when_determined_and_prior_when_not():
    m1 = estimate_flow(FLOW, EDGES, ["J1_J2"])["M1"]
    assert m1["J1_J2"].to_numpy() == pytest.approx(10.0)
    both = estimate_flow(FLOW, EDGES, ["J1_J2", "J2_J1"])
    est = both["M1"].iloc[0]
    assert est["J1_J2"] - est["J2_J1"] == pytest.approx(5.0)    # the equations fix the difference ...
    assert est.mean() == pytest.approx(7.5)                      # ... the prior (mean of visible cameras) the level


def test_level_of_uses_v1_thresholds():
    assert level_of(pd.Series([0.1, 0.35, 0.5, 0.8, 1.2])).tolist() == ["free", "free", "moderate", "heavy", "jammed"]
