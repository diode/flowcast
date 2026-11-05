from types import SimpleNamespace

import pandas as pd

from netstudy.whatif import _hhmm, _run_args

SIM = {"demand": {"zones": {"J2": {"peak_veh_per_hour": 700}}}}


def test_demand_scenario_scales_a_copy_of_the_zone():
    run = SimpleNamespace(kind="scenario", start_s=300.0, end_s=86400.0)
    sim, spec = _run_args(SIM, {"kind": "demand", "zone": "J2", "factor": 1.3}, run)
    assert sim["demand"]["zones"]["J2"]["peak_veh_per_hour"] == 910 and spec == []
    assert SIM["demand"]["zones"]["J2"]["peak_veh_per_hour"] == 700          # the config itself unchanged


def test_closure_and_placebo_specs():
    run = SimpleNamespace(kind="scenario", start_s=27000.0, end_s=32400.0)
    _, spec = _run_args(SIM, {"kind": "closure", "edge": "J1_J2"}, run)
    assert spec == [{"edge": "J1_J2", "block": 2, "start_s": 27000.0, "end_s": 32400.0}]
    placebo = SimpleNamespace(kind="placebo", start_s=27000.0, end_s=27300.0, car="in_J3 out_J5")
    _, spec = _run_args(SIM, {"kind": "closure", "edge": "J1_J2"}, placebo)
    assert spec[0]["car"] == ["in_J3", "out_J5"]


def test_hhmm():
    assert _hhmm(pd.Timedelta(hours=7, minutes=45)) == "07:45"
