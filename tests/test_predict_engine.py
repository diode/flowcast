"""The predict switches of architecture-next 1.6: engine graph (default), engine sumo refused; assist glimmer / off."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from netstudy.cli import cmd_predict
from netstudy.predict_data import predict_settings

TOWNS = Path(__file__).resolve().parents[1] / "towns"


def test_defaults_and_assist_off():
    assert predict_settings({}) == {"engine": "graph", "assist": "glimmer"}
    assert predict_settings({"predict": {"engine": "graph", "assist": "off"}})["assist"] == "off"


@pytest.mark.parametrize("p, msg", [({"engine": "sumo"}, "refused"), ({"engine": "lstm"}, "expected one of"),
                                    ({"assist": "muse_spark"}, "not built yet"), ({"assist": "gpt"}, "expected one of")])
def test_refused(p, msg):
    with pytest.raises(SystemExit, match=msg):
        predict_settings({"predict": p})


def test_sumo_refused_in_a_real_town_too():
    with pytest.raises(SystemExit, match="refused"):
        predict_settings({"source": {"kind": "real"}, "predict": {"engine": "sumo"}})


@pytest.mark.parametrize("town", sorted(p.stem for p in TOWNS.glob("*.yaml")))
def test_study_towns_use_the_graph_engine(town):
    import yaml

    sim = yaml.safe_load((TOWNS / f"{town}.yaml").read_text(encoding="utf-8"))
    assert predict_settings(sim) == {"engine": "graph", "assist": "glimmer"}


def test_netstudy_predict_does_not_run_sumo():
    args = SimpleNamespace(sim_config=str(TOWNS / "ashford.yaml"), precompute=False, edge=None)
    with pytest.raises(SystemExit, match="disabled.*graph-model"):
        cmd_predict(args)


def test_tools_refuse_times_outside_the_filmed_windows():
    from netstudy import dashboard_data as dd
    from netstudy.predict_data import load_tools

    if "ashford" not in dd.networks():
        pytest.skip("no data/ashford results")
    tools = load_tools(dd.Network("ashford"))
    ok = tools.call("set_road", {"edge": "J1_C", "level": "jammed", "minutes": 60, "date": "2026-11-04", "time": "09:00"})
    assert "case" in ok
    for name, args in [("set_road", {"edge": "J1_C", "level": "jammed", "minutes": 60, "date": "2026-11-04",
                                     "time": "12:00"}), ("expected_state", {"date": "2026-11-04", "time": "20:00"})]:
        r = tools.call(name, args)
        assert "outside the filmed windows" in r["error"] and "morning 08:30–11:00" in r["error"]


def test_glimmer_not_started_while_another_gpu_is_busy(monkeypatch):
    from netstudy import glimmer_server as gs

    monkeypatch.setattr(gs, "status", lambda: "off")
    monkeypatch.setenv("FLOWCAST_ONE_GPU_AT_A_TIME", "1")          # the workspace rule (workspace.yaml / env)
    monkeypatch.setenv("FLOWCAST_GLIMMER_GPU", "0")
    monkeypatch.setattr(gs, "gpu_load", lambda: {0: {"name": "GPU A", "util_pct": 0, "mem_mib": 300, "watts": 10},
                                                 1: {"name": "GPU B", "util_pct": 95, "mem_mib": 9000, "watts": 250}})
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(gs, "EXE", Path(__file__))                  # exists; never run
    monkeypatch.setattr(gs, "MODEL", Path(__file__))
    with pytest.raises(SystemExit, match="GPU B is under load"):
        gs.start()
    monkeypatch.setattr(gs, "gpu_load", lambda: {1: {"name": "GPU B", "util_pct": 79, "mem_mib": 295, "watts": 16}})
    assert gs.others_busy() is None                                   # the desktop's utilization spikes: not a job
