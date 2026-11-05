from datetime import datetime, timedelta

from netstudy.direct import PerfectCamera, _slot_rows
from netstudy.ingest import OBS_COLUMNS, slot_values

CAM = {"camera_id": "cam01", "edge": "A_B", "lanes": ["A_B_0", "A_B_1"], "capacity_pcu": 20}


class FakeTraci:
    """Two lanes of 100 m; vehicles: (lane, position from the lane start, type, m/s)."""

    def __init__(self, vehicles: dict):
        self.v = vehicles
        traci = self

        class Lane:
            @staticmethod
            def getLastStepVehicleIDs(ln):
                return [k for k, x in traci.v.items() if x[0] == ln]

        class Vehicle:
            getLanePosition = staticmethod(lambda k: traci.v[k][1])
            getTypeID = staticmethod(lambda k: traci.v[k][2])
            getSpeed = staticmethod(lambda k: traci.v[k][3])

        self.lane, self.vehicle = Lane, Vehicle


def camera(vehicles):
    return PerfectCamera(FakeTraci(vehicles), [CAM], {"A_B_0": 100.0, "A_B_1": 100.0}, 60.0, (5.0, 35.0))


def test_perfect_camera_sees_what_ingest_would():
    vehicles, speeds = camera({
        "a": ("A_B_0", 98.0, "car", 0.0),            # 2 m before the stop line, queued: in view, not between the lines
        "b": ("A_B_1", 80.0, "bus", 10.0),           # 20 m: between the speed lines
        "c": ("A_B_0", 30.0, "car", 13.9),           # 70 m: beyond the view
        "d": ("A_B_1", 70.0, "auto_rickshaw", 5.0),  # 30 m; vision has no such type: a car
    }).sample(CAM)
    assert [(v["type"], v["lane"], v["m"]) for v in vehicles] == [("car", 0, 2.0), ("bus", 1, 20.0), ("car", 1, 30.0)]
    assert speeds == {"all": [0.0, 36.0, 18.0], "band": [36.0, 18.0]}
    assert all("score" not in v for v in vehicles)    # no detection score: mean_score stays empty


def test_direct_rows_have_the_observations_structure():
    cam = camera({"b": ("A_B_1", 80.0, "bus", 10.0)})
    t0 = datetime(2026, 10, 6, 8, 0)
    samples = {"cam01": [(t0 + timedelta(seconds=12 * k), *cam.sample(CAM)) for k in range(1, 26)]}
    rows = _slot_rows([CAM], samples, t0 + timedelta(minutes=5), 300, 60.0, 10)
    assert len(rows) == 1 and len(rows[0]) == len(OBS_COLUMNS)
    row = dict(zip(OBS_COLUMNS, rows[0]))
    assert row["source"] == "sumo" and row["ts"] == "2026-10-06T08:05:00" and row["samples"] == 25
    assert row["mean_score"] is None and row["image_path"] == "" and row["period"] == ""
    assert row["total"] == 1 and row["pcu"] == 3.0
    assert row["flow"] == 50.0                   # Edie: 10 m/s over 60 m, for 300 s
    assert row["speed"] == 36.0
    assert samples["cam01"] == []                # used samples are dropped


def test_a_slot_with_too_few_samples_is_a_gap():
    cam = camera({})
    t0 = datetime(2026, 10, 6, 8, 0)
    samples = {"cam01": [(t0 + timedelta(seconds=12 * k), *cam.sample(CAM)) for k in range(1, 5)]}
    assert _slot_rows([CAM], samples, t0 + timedelta(minutes=5), 300, 60.0, 10) == []


def test_rewriting_a_day_keeps_the_previous_days_midnight_slot(tmp_path):
    import sqlite3

    import pandas as pd

    from netstudy.direct import write_rows
    from netstudy.ingest import SCHEMA

    db = tmp_path / "traffic.db"
    with sqlite3.connect(db) as c:
        c.executescript(SCHEMA)

    def day_rows(ts_list):
        base = dict.fromkeys(OBS_COLUMNS, None) | {"camera_id": "cam01", "total": 0, "pcu": 0, "occupancy": 0,
                                                     "load_ratio": 0, "source": "sumo"}
        return pd.DataFrame([base | {"ts": ts} for ts in ts_list], columns=OBS_COLUMNS)

    write_rows(db, "2026-10-06", day_rows(["2026-10-06T00:05:00", "2026-10-07T00:00:00"]))
    write_rows(db, "2026-10-07", day_rows(["2026-10-07T00:05:00", "2026-10-08T00:00:00"]))
    write_rows(db, "2026-10-07", day_rows(["2026-10-07T00:05:00", "2026-10-08T00:00:00"]))   # again: replaced
    with sqlite3.connect(db) as c:
        assert [r[0] for r in c.execute("SELECT ts FROM observations ORDER BY ts")] == [
            "2026-10-06T00:05:00", "2026-10-07T00:00:00", "2026-10-07T00:05:00", "2026-10-08T00:00:00"]


def test_vision_and_direct_share_one_slot_code():
    t = datetime(2026, 10, 6, 8, 5)
    seen = [{"type": "car", "lane": 0, "m": 10.0, "score": 0.8}]
    v = slot_values(CAM, [(t, seen, {"band": [30.0], "all": [30.0]})], t, 300, 60.0)
    d = slot_values(CAM, [(t, [{k: x for k, x in seen[0].items() if k != "score"}],
                           {"band": [30.0], "all": [30.0]})], t, 300, 60.0)
    assert v[:6] == d[:6] and v[8:] == d[8:]           # all but mean_score and processed_at
    assert v[6] == 0.8 and d[6] is None
