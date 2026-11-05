from pathlib import Path

import pytest

from netstudy.assist import Tools, numbers_ok

ROOT = Path(__file__).resolve().parents[1]
needs_ashford = pytest.mark.skipif(not (ROOT / "data" / "ashford" / "traffic.db").exists(), reason="no Ashford data")


@pytest.fixture(scope="module")
def tools():
    from netstudy.cli import load_sim_config
    sim, out = load_sim_config(ROOT / "towns" / "ashford.yaml")
    return Tools.load("ashford", sim, out)


def test_numbers_check():
    results = [{"flow_vph": 486, "speed_kmh": 25.8}]
    assert numbers_ok("486 veh/h at 26 km/h", "", results)[0]          # rounding a tool's 25.8 is fine
    assert numbers_ok("from 08:00 on 2026-11-04", "tomorrow 8am", [{"date": "2026-11-04", "from": "08:00"}])[0]
    ok, bad = numbers_ok("about 322 vehicles", "", results)
    assert not ok and bad == ["322"]


@needs_ashford
def test_find_road(tools):
    assert tools.call("find_road", {"text": "Elm Park to Ashford Centre"})["candidates"][0]["edge"] == "J1_C"
    assert tools.call("find_road", {"text": "Hillcrest Road"})["candidates"] == []


@needs_ashford
@pytest.mark.parametrize("args, error", [
    ({"edge": "J1_C", "level": "closed", "minutes": 180, "date": "2026-11-04", "time": "08:00"}, "30-120"),
    ({"edge": "Elm Park", "level": "closed", "minutes": 60, "date": "2026-11-04", "time": "08:00"}, "find_road"),
    ({"edge": "J1_C", "level": "blocked", "minutes": 60, "date": "2026-11-04", "time": "08:00"}, "level"),
    ({"edge": "J1_C", "level": "closed", "minutes": 60, "date": "2027-03-01", "time": "08:00"}, "calendar"),
    ({"edge": "J1_C", "level": "closed", "minutes": 60, "date": "2026-11-04", "time": "8am"}, "HH:MM"),
])
def test_set_road_checks(tools, args, error):
    r = tools.call("set_road", args)
    assert error in r["error"] and not tools.cases


@needs_ashford
def test_case_tools_read_only(tools):
    r = tools.call("set_road", {"edge": "J1_C", "level": "closed", "minutes": 60, "date": "2026-11-04", "time": "09:00"})
    imp = tools.call("impact", {"case": r["case"]})
    assert imp["roads"][0]["edge"] == "J1_C" and imp["roads"][0]["flow_vph"] == 0
    assert tools.call("actions", {"case": r["case"]})["actions"][0]["priority"] == "high"
    assert "error" in tools.call("impact", {"case": "c99"})


def test_numbers_check_reads_dashes_and_thousands():
    assert numbers_ok("+10–20 s per cycle, detour 1 322 m", "", [{"action": "+10–20 s", "length_m": 1322}])[0]
    assert numbers_ok("30–120 minutes only", "", [], given="A road change lasts 30 to 120 minutes")[0]


@needs_ashford
def test_fast_path_plan_prompt_run_and_facts(tools):
    from datetime import date

    from netstudy import assist

    pr = assist.plan_prompt("Ashford", date(2026, 11, 4), tools, (date(2026, 11, 27), "09:00"))
    assert "no date means Friday 2026-11-27, with no time 09:00" in pr and "Mon 2026-11-09" in pr
    assert "J1_C: Elm Park → Ashford Centre" in pr
    assert "J1_C" in assist.plan_schema(tools)["properties"]["changes"]["items"]["properties"]["edge"]["enum"]
    none = {"changes": [], "usual": [], "detour": [], "history": [], "day": [], "unknown": ""}
    plan = none | {"changes": [{"edge": "J1_C", "level": "jammed", "minutes": 60, "date": "2026-11-27",
                                "time": "09:00"}]}
    calls = assist.run_plan(tools, plan)
    assert [c["name"] for c in calls] == ["set_road", "impact", "actions", "day_kind"]
    fs = assist.facts(calls, plan)
    assert fs[0].startswith("Elm Park → Ashford Centre jammed for 60 min from Fri 27 Nov 09:00") and "Harvest Fair" in fs[-1]
    v = assist.verdicts(calls)
    d = v[0]["delay"]
    assert len(v) == 1 and not v[0]["best"] and v[0]["verdict"] == (
        "not" if d > assist.VERDICT_DELAY_VEH_H else "caution" if d > assist.VERDICT_DELAY_VEH_H / 2
        else v[0]["verdict"])                                   # a queue can still raise it
    acts = assist.action_items(calls)
    assert acts and acts[0]["priority"] == "high" and not any(a["action"].startswith("No action") for a in acts)
    late = none |{"changes": [{"edge": "J1_C", "level": "slow", "minutes": 60, "date": "2026-11-27", "time": "12:00"}]}
    assert assist.facts(assist.run_plan(tools, late), late)[0].startswith("Refused (set_road): time 12:00")
    assert assist.facts([], none | {"unknown": "Hillcrest Road"}) == ["Not a road or place in the town: Hillcrest Road."]


@needs_ashford
def test_muse_spark_backend_calls_the_meta_api(tools, monkeypatch, tmp_path):
    """Muse Spark: the same plan and insight calls as Glimmer, sent to the Meta Model API's chat completions with
    the key; the choice is kept (assist_backend) and refused without a key."""
    import io
    import json

    from netstudy import assist, assist_backend as ab

    monkeypatch.setattr(ab, "STATE", tmp_path / "backend.json")
    monkeypatch.setattr("netstudy.glimmer_server.stop", lambda: None)
    assert ab.current() == "glimmer" and ab.choose("muse_spark") is None and ab.current() == "muse_spark"
    monkeypatch.delenv(ab.KEY_ENV, raising=False)
    assert ab.status() == "no key"
    a = assist.Assistant("ashford", {}, tmp_path, tools=tools)
    with pytest.raises(OSError, match=ab.KEY_ENV):
        a._complete([{"role": "user", "content": "hi"}])
    monkeypatch.setenv(ab.KEY_ENV, "test-key")
    assert ab.status() == "ready"
    sent = []

    def fake(req, timeout):
        sent.append((req.full_url, dict(req.header_items()), json.loads(req.data)))
        return io.StringIO(json.dumps({"choices": [{"message": {"content": '```json\n{"ok": 1}\n```'}}]}))

    monkeypatch.setattr(assist.urllib.request, "urlopen", fake)
    assert a._complete([{"role": "user", "content": "plan"}], {"type": "object"}) == '{"ok": 1}'
    url, head, body = sent[0]
    assert url == f"{ab.API_URL}/chat/completions" and head["Authorization"] == "Bearer test-key"
    assert body["model"] == ab.API_MODEL and body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    a._complete([{"role": "user", "content": "insight"}])
    assert sent[1][2]["stop"] == ["\n\n"] and "response_format" not in sent[1][2]
