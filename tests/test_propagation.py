import pandas as pd

from netstudy.propagation import hop_any, hops

# A chain A -> B -> C with a two-way road B-C and neighbourhood roads at A and C.
EDGES = pd.DataFrame([
    ("in_A", "F_A", "A"), ("A_B", "A", "B"), ("B_C", "B", "C"), ("C_B", "C", "B"), ("out_C", "C", "F_C"),
], columns=["edge_id", "from_node", "to_node"])


def test_hops_follow_traffic_without_u_turns():
    up, down = hops(EDGES, "B_C")
    assert up == {"B_C": 0, "A_B": 1, "in_A": 2}             # C_B -> B_C would be a U-turn
    assert down == {"B_C": 0, "out_C": 1}


def test_hop_any_ignores_direction():
    assert hop_any(EDGES, "B_C") == {"B_C": 0, "A_B": 1, "C_B": 1, "out_C": 1, "in_A": 2}
