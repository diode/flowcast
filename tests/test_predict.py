from datetime import date
from pathlib import Path

import pytest

from netstudy import dashboard_data as dd
from netstudy import demand, predict
from netstudy import predict_data as pdd

HAVE = set(dd.networks())
needs = lambda n: pytest.mark.skipif(n not in HAVE, reason=f"no data/{n} results")  # noqa: E731
# Cities with a week of studied days: a usual day to compare with, 3 base days to simulate from.
WEEK = sorted(n for n in HAVE if len(dd.Network(n).days()) >= 7)


def test_calendar_and_time_context():
    cal = predict.calendar()
    assert len(cal) == 61 and cal.date.min() == date(2026, 11, 1)
    assert predict.day_kind(date(2026, 11, 27)) == "festival" and predict.day_kind(date(2026, 12, 25)) == "holiday"
    assert predict.day_kind(date(2026, 11, 7)) == "weekend" and predict.day_kind(date(2026, 11, 4)) == "working"
    assert predict.time_context("working", "08:30") == "working-day morning peak"
    assert predict.time_context("holiday", "21:00") == "holiday after-evening"
    assert predict.time_context("festival", "16:30", predict.event_on(date(2026, 11, 8))).startswith("crowds arriving")


def test_edit_trips(tmp_path):
    lines = ["<routes>", '  <vType id="car"/>']
    lines += [f'  <trip id="t{i}" type="car" depart="{i * 36.0:.1f}" from="a" to="b"/>' for i in range(200)]   # 0-2 h
    f = tmp_path / "trips.rou.xml"
    f.write_text("\n".join(lines + ["</routes>"]), encoding="utf-8")
    factors = [0.5, 2.0] + [1.0] * 22
    n = demand.edit_trips(f, {"hour_factors": factors, "venue": {"to": ["v"], "from": ["w"], "arrive": [3600, 7200],
                                                                 "leave": [0, 0], "share": 1.0}}, seed=3)
    text = f.read_text(encoding="utf-8")
    first = text.count('depart="') - text.count('to="v"')
    assert 100 * 0.5 * 0.7 < sum(1 for ln in text.splitlines() if 'depart="' in ln and float(ln.split('depart="')[1].split('"')[0]) < 3600) < 100 * 0.5 * 1.3
    assert text.count('to="v"') == 100                   # each trip of hour 1 once more, to the venue
    assert n == first + text.count('to="v"')
    departs = [float(ln.split('depart="')[1].split('"')[0]) for ln in text.splitlines() if 'depart="' in ln]
    assert departs == sorted(departs)


@pytest.mark.parametrize("name", WEEK)
def test_case_spec_and_runs(name):
    net = dd.Network(name)
    edge = net.edges().edge_id.iloc[0]
    s = predict.case_spec(name, date(2026, 11, 27), "18:00", {"edge": edge, "level": "closed", "minutes": 60}, net.db)
    assert s["kind"] == "festival" and len(s["base_days"]) == predict.REPLICATES
    assert all(date.fromisoformat(d).weekday() == 4 for d in s["base_days"][:1])           # a Friday
    roles = [r["role"] for r in predict.runs_for(s)]
    assert roles.count("case") == roles.count("placebo") == roles.count("base") == predict.REPLICATES
    assert s["demand"]["venue"]["to"] and s["change"]["incident"]["block"] == 99
    w = predict.case_spec(name, date(2026, 11, 4), "08:00", None, net.db)
    assert w["demand"] is None and predict.runs_for(w) == []                                # history only


@needs("maple_falls")
def test_usual_prediction_and_stored_case():
    net = dd.Network("maple_falls")
    spec = predict.case_spec("maple_falls", date(2026, 11, 4), "08:00", None, net.db)
    exp = pdd.expected(net, spec)
    assert exp["source"] == "usual" and len(exp["state"]) == 26
    t = pdd.kpi_tiles(exp, "08:00")["veh_in_town"]
    assert t["lo"] <= t["now"] <= t["hi"]
    case = predict.case_spec("maple_falls", date(2026, 11, 4), "08:00",
                             {"edge": "J1_J2", "level": "jammed", "minutes": 60}, net.db)
    if not all(v == "done" for v in pdd.case_status(net, case).values()):
        pytest.skip("case not simulated (netstudy predict --date 2026-11-04 --time 08:00 --edge J1_J2)")
    r = pdd.case_result(net, case)
    j = r["junction"].set_index("junction")
    assert j.at["J1", "affected"] == 1 and j.at["J1", "days_agree"] == 3
    assert {a["junction"] for a in r["actions"]} >= {"J1"}
    assert any("Divert" in a["action"] for a in r["actions"])


APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.mark.parametrize("name", sorted(HAVE))
def test_predict_pages_run(name):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params.update({"mode": "Predict", "city": name, "date": "2026-11-04", "time": "08:00"})
    at.run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Wednesday 04 November 2026" in html and "Recommended actions" in html
    at.switch_page("technical.py").run()
    assert not at.exception and at.title[0].value.startswith("Forecast")


@pytest.mark.skipif(not HAVE, reason="no studied city in data/ yet")
def test_retrospective_is_unchanged_default():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    assert any("Vehicles on the roads" in m.value for m in at.markdown)             # the retrospective tiles
    assert not any("pbanner" in m.value for m in at.markdown)


@needs("maple_falls")
def test_map_roads_are_clickable():
    from netstudy import overview as ov

    net = dd.Network("maple_falls")
    spec = predict.case_spec("maple_falls", date(2026, 11, 4), "08:00", None, net.db)
    opt = ov.map_option(net, pdd.expected(net, spec)["state"], ov.places(net), highlight="J4_J5")
    hit = next(s for s in opt["series"] if s["name"] == "_roads")
    assert {d["edge"] for d in hit["data"]} == set(net.edges().edge_id)          # every road can be clicked
    assert opt["series"][0]["name"] == "Selected road" and "_roads" not in opt["legend"]["data"]


@needs("maple_falls")
def test_a_map_click_sets_the_road_picker():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params.update({"mode": "Predict", "city": "maple_falls", "date": "2026-11-04", "time": "08:00"})
    at.run()
    at.session_state["p_road_pending"] = "J4_J5"           # what _pick_road leaves after a click
    at.run()
    road = next(s for s in at.selectbox if s.label.startswith("Road"))
    assert road.value == "J4_J5" and not at.exception
