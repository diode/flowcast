import json
import math
from pathlib import Path

import pytest

from netstudy import dashboard_data as dd
from netstudy import overview as ov
from netstudy.places import city_name

HAVE = set(dd.networks())
needs = lambda n: pytest.mark.skipif(n not in HAVE, reason=f"no data/{n} results")  # noqa: E731
# Cities with a week of studied days: a usual day to compare with, 3 base days to simulate from.
WEEK = sorted(n for n in HAVE if len(dd.Network(n).days()) >= 7)
DAY, TIME = "2026-10-06", "18:00"


def test_city_names():
    assert [city_name(n) for n in ("maple_falls", "westbrook", "harbor_city")] == ["Maple Falls", "Westbrook", "Harbor City"]
    assert ov.flow_score(1.08) == 100 and ov.flow_score(0.674) == 67 and ov.flow_score(float("nan")) == 0


@pytest.mark.parametrize("name", sorted(HAVE))
def test_every_road_and_junction_has_a_unique_name(name):
    net = dd.Network(name)
    pl = ov.places(net)
    roads = [pl.road(e) for e in net.edges().edge_id]
    assert len(set(roads)) == len(roads)
    assert not any(e in r for e, r in zip(net.edges().edge_id, roads) if len(e) > 3)      # no raw ids shown
    j = net.nodes().query("kind == 'junction'").node_id
    assert len({pl.node(n) for n in j}) == len(j)


@needs("maple_falls")
def test_maple_falls_names():
    net = dd.Network("maple_falls")
    pl = ov.places(net)
    assert pl.road("J1_J2") == "Oakwood Avenue → Downtown"
    assert pl.road("in_J2") == "Downtown local streets (inbound)"


@needs("westbrook")
def test_westbrook_names():
    pl = ov.places(dd.Network("westbrook"))
    assert pl.road("A1B1") == "Station Street eastbound to Birch Avenue"
    assert pl.road("A0left0") == "Market Street westbound, to west gateway"


@pytest.mark.parametrize("name", WEEK)
def test_widgets(name):
    net = dd.Network(name)
    k, pl = ov.all_kpis(net), ov.places(net)
    state = net.state_at(net.slot(DAY, TIME))
    h = ov.headline(net, k, state, DAY, TIME)
    for tile in h.values():
        assert math.isfinite(tile["now"]) and math.isfinite(tile["usual"]) and len(tile["spark"]) == 24
    assert sum(ov.road_levels(state).values()) == len(state)
    m = ov.map_option(net, state, pl)
    drawn = sum(len(s["data"]) for s in m["series"] if s["type"] == "lines" and s["name"] != "_roads")
    assert drawn == len(state)
    w = ov.weekly_profile(k)
    assert len(w) == 7 * 24
    for option in (m, ov.gauge_option(50), ov.donut_option(ov.road_levels(state)), ov.rhythm_option(k, DAY, TIME),
                   ov.heatmap_option(w), ov.ranking_option(ov.slowest_roads(state, pl))):
        json.dumps(option)                                       # what streamlit-echarts sends


@needs("maple_falls")
def test_maple_falls_findings():
    net = dd.Network("maple_falls")
    assert ov.detection_summary(net) == pytest.approx(dict(closures=32, found=32, minutes=20.0, false_per_day=1.53), abs=0.01)
    assert {c["id"] for c in ov.whatif_cards(net)} == {"close_J1_J2_am", "close_J2_J3_pm", "centre_growth"}
    assert set(ov.forecast_accuracy(net)) == {15, 30, 60}


APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.mark.parametrize("name", sorted(HAVE))
def test_overview_page_runs(name):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    at.selectbox[0].set_value(name).run()
    assert not at.exception
    pick = next(s for s in at.segmented_control if s.label == "Time")   # one button per period, no slider
    pick.set_value(pick.options[0]).run()
    at.selectbox[1].set_value(dd.Network(name).days()[0]).run()      # a studied day of this city
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert city_name(name) in html and "Vehicles on the roads" in html
