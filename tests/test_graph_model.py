import math
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pytest

from netstudy.graph_model import GraphModel, overflow_s, webster_s

ROOT = Path(__file__).resolve().parents[1]
ASHFORD = ROOT / "data" / "ashford" / "traffic.db"
needs_ashford = pytest.mark.skipif(not ASHFORD.exists(), reason="no Ashford data")
DAY = date(2026, 11, 4)


@pytest.fixture(scope="module")
def model():
    from netstudy.cli import load_sim_config
    sim, out = load_sim_config(ROOT / "towns" / "ashford.yaml")
    return GraphModel.fit(sim, out, town="ashford")


def test_delay_formulas():
    assert webster_s(90, 0.3, 0.0) == pytest.approx(0.5 * 90 * 0.49)
    assert webster_s(90, 0.3, 0.5) > webster_s(90, 0.3, 0.1)
    assert overflow_s(0.3, 1000) < 1 < overflow_s(1.2, 1000)            # small below capacity, a queue above
    assert math.isinf(overflow_s(0.5, 0))


def test_no_sumo_and_only_camera_tables():
    """The graph model imports nothing of the simulation and reads only camera-derived tables and the incident log."""
    code = ("import sys, netstudy.graph_model; bad = [m for m in sys.modules if m in ('sumolib', 'traci', 'libsumo') "
            "or m in ('netstudy.simulate', 'netstudy.direct', 'netstudy.whatif', 'netstudy.predict', "
            "'netstudy.record', 'netstudy.network')]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    src = (ROOT / "src" / "netstudy" / "graph_model.py").read_text(encoding="utf-8")
    assert set(re.findall(r"FROM (\w+)", src)) <= {"edges", "edge_state", "incident_log", "sqlite_master"}


@needs_ashford
def test_turn_shares_sum_to_one(model):
    s = model.turns.groupby(["day_kind", "hour", "from_edge"]).share.sum()
    assert ((s - 1).abs() < 1e-6).all()


@needs_ashford
def test_closed_road_carries_nothing_and_answers_fast(model):
    t = time.time()
    r = model.road_change("J1_C", "closed", DAY, "08:00", 60)
    assert time.time() - t < 5
    e = r["edges"]
    assert (e[(e.edge_id == "J1_C") & e.change_on].flow == 0).all()
    assert "J1_J2" in r["summary"]["affected_edges"]                       # the detour round by Mill Yard
    assert r["summary"]["cleared"] is not None


@needs_ashford
def test_traffic_in_equals_out(model):
    """Per junction and slot the predicted changes balance (no queue: jammed still lets J1_C's traffic through)."""
    r = model.road_change("J1_C", "jammed", DAY, "08:00", 60)
    e = r["edges"].merge(model.edges[["edge_id", "from_node", "to_node"]], on="edge_id")
    e = e[e.change_on]
    for j in ("C", "J1", "J2", "J3"):
        d_in = e[e.to_node == j].groupby("slot").flow_change.sum()
        d_out = e[e.from_node == j].groupby("slot").flow_change.sum()
        assert ((d_in - d_out).abs() < 1.0).all(), (j, (d_in - d_out).to_dict())


@needs_ashford
def test_expected_day_and_assumed_holiday(model):
    u = model.expected_day(DAY)
    assert not u.assumed.any() and (u.flow_p10 <= u.flow_p50).all() and (u.flow_p50 <= u.flow_p90).all()
    assert model.expected_day(date(2026, 12, 25)).assumed.all()          # no recorded holiday: assumed
