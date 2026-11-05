import re
from pathlib import Path

import pytest
import yaml

from netstudy.network import movement_edges, signal_states

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "data" / "ashford" / "network" / "net.net.xml"
TOWN = ROOT / "towns" / "ashford.yaml"
needs_net = pytest.mark.skipif(not NET.exists(), reason="Ashford network not built")


def signals() -> dict:
    return yaml.safe_load(TOWN.read_text(encoding="utf-8"))["network"]["town"]["signals"]


def test_movement_edges():
    assert movement_edges("C", "J2>hood") == ("J2_C", "out_C")
    assert movement_edges("J1", "hood>C") == ("in_J1", "J1_C")


@needs_net
def test_timing_sheet_is_what_sumo_runs():
    """The town config's plans give exactly the phases in the network the sample data was made with."""
    text = NET.read_text(encoding="utf-8")
    for j, phases in signal_states(NET, signals()).items():
        block = re.search(rf'<tlLogic id="{j}".*?</tlLogic>', text, flags=re.S).group(0)
        assert [(float(d), s) for d, s in re.findall(r'duration="([\d.]+)"\s+state="(\w+)"', block)] == phases


@needs_net
def test_unknown_movement_refused():
    bad = {"C": {"phases": [{"green_s": 30, "go": ["J1>J1"]}]}}
    with pytest.raises(SystemExit, match="not a turn"):
        signal_states(NET, bad)
