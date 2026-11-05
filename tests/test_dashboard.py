from pathlib import Path

import pandas as pd
import pytest

from netstudy import dashboard_data as dd
from netstudy.places import city_name

HAVE = set(dd.networks())
needs = lambda n: pytest.mark.skipif(n not in HAVE, reason=f"no data/{n} results")  # noqa: E731


def test_speed_level_bins():
    assert [dd.speed_level(x) for x in (0.95, 0.6, 0.3, 0.1)] == ["free", "moderate", "heavy", "jammed"]
    assert dd.speed_level(float("nan")) is None


@needs("maple_falls")
def test_maple_falls_state_and_map():
    net = dd.Network("maple_falls")
    ts = net.slot("2026-10-06", "18:00")
    s = net.state_at(ts)
    assert len(s) == 26 and s.level.notna().all() and s.observed.eq(1).all()
    fig = dd.map_figure(net, s, "level")
    drawn = [t for t in fig.data if t.name in set(s.edge_id)]
    assert len(drawn) == 26
    d = net.edge_day("J1_J2", "2026-10-06")
    assert len(d) == 288 and d.usual_flow.notna().all()


@needs("maple_falls")
def test_maple_falls_phase_results_present():
    net = dd.Network("maple_falls")
    assert len(net.closures()) == 32 and not net.footprint("J1_J2@08:00").empty
    assert set(net.scenarios()) >= {"close_J1_J2_am"}
    base, run = net.scenario_state("close_J1_J2_am", net.scenario_days("close_J1_J2_am")[0], "08:00")
    assert run.set_index("edge_id").at["J1_J2", "flow"] < base.set_index("edge_id").at["J1_J2", "flow"]
    assert set(net.forecast_scores().model) == {"v1", "network", "persistence", "same_time_yesterday"}


APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.mark.parametrize("name", sorted(HAVE))
def test_technical_view_runs_every_tab(name):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.switch_page("technical.py").run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value(name).run()
    assert at.title[0].value.startswith(city_name(name))
    assert not at.exception
    for view in ("Cameras", "True state"):
        at.sidebar.radio[0].set_value(view).run()
        at.sidebar.radio[1].set_value("Speed vs free flow").run()
        assert not at.exception
    at.sidebar.selectbox[1].set_value(dd.Network(name).days()[0]).run()      # a studied day of this city
    assert not at.sidebar.select_slider                                     # no slider: one button per period
    pick = at.sidebar.segmented_control[0]
    pick.set_value(pick.options[-1]).run()
    assert not at.exception


def test_choose_time_offers_only_slots_with_results():
    if "ashford" not in HAVE:
        pytest.skip("no data/ashford results")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params.update({"city": "ashford"})
    at.session_state["day"], at.session_state["hhmm"] = "2026-10-06", "12:00"   # a SUMO day; 12:00 between windows
    at.run()
    assert not at.exception
    pick = next(s for s in at.segmented_control if s.label == "Time")
    assert pick.options == ["Morning", "Noon", "Evening", "Night"]         # names only on the buttons
    assert pick.value == "Noon 12:30–15:00" and at.session_state["hhmm"] == "12:35"   # 12:00: the nearest period
    assert any("12:00 is not recorded: showing Noon (12:30–15:00)" in c.value for c in at.caption)
    pick.set_value("Night 20:30–23:00").run()
    assert not at.exception and at.session_state["hhmm"] == "20:35"
    next(s for s in at.segmented_control if s.label == "Time").set_value("Evening 17:00–19:30").run()
    assert at.session_state["hhmm"] == "17:05"                     # every pick moves the time, not only the first
    at.selectbox(key="day_overview").set_value("2026-10-05").run()  # the filmed day: only the old windows' results
    pick = next(s for s in at.segmented_control if s.label == "Time")
    assert not at.exception and not at.select_slider and pick.options == ["Evening", "Night"]
    assert at.session_state["hhmm"] == "17:35"                     # 17:05 not filmed: the period's first result


# ------------------------------------------------------------------ Ashford (the study city, camera data)
@needs("ashford")
def test_ashford_state_and_map():
    net = dd.Network("ashford")
    s = net.state_at(net.slot("2026-10-05", "08:30"))           # the rendered day, morning window
    assert len(s) == 20 and s.flow.notna().all()
    fig = dd.map_figure(net, s, "level")
    assert len([t for t in fig.data if t.name in set(s.edge_id)]) == 20
    assert len(net.edge_day("J1_C", "2026-10-06")) == 288


def _predict(q: dict):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params.update({"mode": "Forecast", "city": "ashford", **q})
    at.run()
    assert not at.exception
    return at, " ".join(m.value for m in at.markdown)


@needs("ashford")
def test_ashford_predict_day_with_badges():
    at, html = _predict({"date": "2026-11-04", "time": "09:00"})
    assert "Wednesday 04 November 2026" in html and "Recommended actions" in html
    assert "observed · 20 roads" in html and "method accuracy in simulation (expected day)" in html
    assert "Ask the Traffic Assistant" in html and "Glimmer" not in html    # the box (its server may be off)


@needs("ashford")
def test_ashford_predict_road_change_from_the_graph_model():
    at, html = _predict({"date": "2026-11-04", "time": "17:30", "edge": "C_J2", "level": "closed", "minutes": "60"})
    assert "junctions affected" in html and "Divert traffic" in html
    assert "method accuracy in simulation (this test case)" in html      # C_J2 closed 17:30 60 is a whatif case
    assert "Simulating with SUMO" not in html


@needs("ashford")
def test_ashford_predict_cases_add_show_and_remove():
    at, html = _predict({"date": "2026-11-04", "time": "09:00", "edge": "J1_C", "level": "closed", "minutes": "60"})
    assert "Impact of case 1" in html and "Set a road's traffic level" not in html   # the form is off: Glimmer
    at.session_state["p_link_change"] = {"edge": "J1_C", "level": "jammed", "minutes": 60}   # a second case
    at.run()
    html = " ".join(m.value for m in at.markdown)
    assert not at.exception and "Impact of case 2" in html
    at.button(key="p_case_1").click().run()                         # back to the first case
    assert "Impact of case 1" in " ".join(m.value for m in at.markdown)
    at.button(key="p_del_1").click().run()                          # removing the shown case: expected traffic
    html = " ".join(m.value for m in at.markdown)
    assert "Expected road map" in html and "Impact of case" not in html
    at.session_state["p_q_road"] = "J3_C"                           # a map click: a question about the road
    at.run()
    assert not at.exception and at.session_state["p_q"].startswith("What if ")
    assert "Wednesday 2026-11-04 at 09:00" in at.session_state["p_q"]
    from netstudy import overview as ov
    net = dd.Network("ashford")
    marks = [{"edge": "J1_C", "label": "1", "case": 1, "on": True, "here": True, "tip": "1"},
             {"edge": "J3_C", "label": "3", "case": 3, "on": False, "here": False, "tip": "3"}]
    pins = next(s for s in ov.map_option(net, net.state_at(net.slot("2026-10-05", "08:00")), ov.places(net),
                                         marks=marks)["series"] if s["name"] == "Your cases")
    assert [p["case"] for p in pins["data"]] == [1, 3]                # every case pinned on its road
    at.button(key="p_clear").click().run()                          # remove all
    html = " ".join(m.value for m in at.markdown)
    assert not at.exception and "Forecast cases" in html and not any(b.key == "p_case_2" for b in at.button)


@needs("ashford")
def test_ashford_glimmer_samples():
    import sys
    sys.path.insert(0, str(Path(APP).parent))
    import predict_view as pv

    qs = pv.sample_questions("ashford")
    assert qs and not any("Riverside Road" in q for q in qs)        # the refusal questions are not samples
    assert pv.sample_questions("brookhaven")[0].startswith("Hillside to Brookhaven Centre")


@needs("ashford")
def test_ashford_predict_offers_only_the_filmed_periods():
    at, html = _predict({"date": "2026-11-04", "time": "12:00"})
    pick = next(s for s in at.segmented_control if s.label == "Time")
    assert pick.options == ["Morning", "Noon", "Evening", "Night"]         # names only on the buttons
    assert pick.value == "Noon 12:30–15:00"                          # 12:00 from the link: the nearest period
    assert any("12:00 is not recorded: showing Noon (12:30–15:00)" in c.value for c in at.caption)
    assert not at.warning and "Expected at 12:30" in html
    pick.set_value("Night 20:30–23:00").run()
    assert not at.exception and "Expected at 20:30" in " ".join(m.value for m in at.markdown)


@needs("ashford")
def test_ashford_technical_predict_and_paired_runs():
    at, _ = _predict({"date": "2026-11-04", "time": "09:00", "edge": "J1_C", "level": "jammed", "minutes": "60"})
    at.switch_page("technical.py").run()
    assert not at.exception and at.title[0].value.startswith("Forecast · Ashford")
    case = next(s for s in at.selectbox if s.label == "Test case")
    for cid in ("J1_C_closed_0800_60", "J3_C_slow_1230_60"):          # the second is outside the windows
        case.set_value(cid).run()
        assert not at.exception
    assert any("SUMO: change run" in c.value for c in at.caption)


@needs("ashford")
@needs("brookhaven")
def test_predict_resets_on_a_town_change():
    at, html = _predict({"date": "2026-11-04", "time": "09:00", "edge": "J1_C", "level": "closed", "minutes": "60"})
    assert "Impact of case 1" in html and at.session_state["p_cases"]
    at.session_state["p_answer"] = {"town": "ashford", "question": "q", "text": "- x", "numbers_ok": True,
                                    "facts": [], "seconds": 1}
    at.selectbox(key="city_predict").set_value("brookhaven").run()
    html = " ".join(m.value for m in at.markdown)
    assert not at.exception and "Impact of case" not in html and "Expected road map" in html
    assert not at.session_state["p_cases"] if "p_cases" in at.session_state else True
    assert "p_answer" not in at.session_state and at.session_state["p_town"] == "brookhaven"


@pytest.mark.parametrize("name", sorted(HAVE))
def test_overview_studies_found_is_never_empty(name):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params.update({"city": name})
    at.run()
    html = " ".join(m.value for m in at.markdown)
    assert not at.exception and "What our studies found" in html
    assert "No studies have been run" not in html and "Forecasts checked" in html
    net = dd.Network(name)
    if net.has("study_closures"):
        assert "Most critical roads" in html and "Best time for roadworks" in html
