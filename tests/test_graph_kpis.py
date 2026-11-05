
import pandas as pd
import pytest
import yaml

from netstudy.graph import build_graph
from netstudy.kpis import compute_kpis, free_flow_speed
from netstudy.network import build_town, camera_edges, cameras_on


def test_maple_falls_graph(tmp_path):
    with open("sim/maple_falls.yaml", encoding="utf-8") as f:
        net_cfg = yaml.safe_load(f)["network"]
    net_file = build_town(net_cfg, tmp_path)
    cams = cameras_on(net_file, camera_edges(net_cfg), 60)
    nodes, edges = build_graph(net_file, cams)
    assert len(nodes) == 10 and (nodes.kind == "junction").sum() == 5
    assert len(edges) == 26 and edges.camera_id.notna().all()
    assert edges.set_index("edge_id").loc["J1_J2", ["from_node", "to_node", "kind"]].tolist() == ["J1", "J2", "road"]
    assert dict(zip(edges.edge_id, edges.camera_id)) == {c["edge"]: c["camera_id"] for c in cams}
    assert (edges.kind == "road").sum() == 16


def test_free_flow_speed_from_quiet_hours():
    edges = pd.DataFrame({"edge_id": ["a", "b"], "speed_kmh": [40.0, 40.0]})
    truth = pd.DataFrame({"edge_id": ["a", "a", "a"], "speed": [20.0, 24.0, 5.0],
                          "ts": ["2026-10-01T01:00:00", "2026-10-01T03:00:00", "2026-10-01T18:00:00"]})
    free = free_flow_speed(truth, edges)
    assert free["a"] == pytest.approx(22.0)        # median of the night slots; 18:00 ignored
    assert free["b"] == pytest.approx(40.0)        # never had traffic at night: speed limit


def test_kpis_on_hand_made_state():
    edges = pd.DataFrame({"edge_id": ["a", "b"], "length_m": [100.0, 300.0], "speed_kmh": [40.0, 40.0]})
    free = pd.Series({"a": 20.0, "b": 20.0}, name="free_kmh")
    truth = pd.DataFrame({
        "edge_id": ["a", "b"], "ts": ["t1", "t1"], "total": [10, 2], "pcu": [12.0, 1.0], "speed": [4.0, 6.0],
        "wait_s": [1800.0, 0.0], "time_loss_s": [2700.0, 0.0], "veh_h": [0.75, 0.25], "incident": [1, 0]})
    k = compute_kpis(truth, edges, free).iloc[0]
    assert k.veh_in_town == 12 and k.pcu_in_town == 13.0
    # a (100 m of 400 m): 0.2 x free flow with 10 vehicles; b is slow too, but 2 vehicles is no queue
    assert k.share_congested == pytest.approx(0.25)
    assert k.speed_index == pytest.approx(0.2 * 0.75 + 0.3 * 0.25, abs=1e-3)
    assert k.delay_veh_h == pytest.approx(0.75) and k.stopped_veh_h == pytest.approx(0.5)
    assert k.worst_edges == "a:0.20, b:0.30" and k.incident_edges == 1
