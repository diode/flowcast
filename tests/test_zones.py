from datetime import date

import numpy as np
import pytest
import yaml

from netstudy.zones import GRID, random_levels, slot_levels, write_zone_trips


def test_slot_levels_follow_table_and_day_type():
    slots = [{"from": "07:30", "to": "10:30", "level": 1.0, "days": "weekdays"},
             {"from": "10:30", "to": "16:30", "level": 0.5}]
    thu = slot_levels(slots, weekday=3, base=0.05, ramp_minutes=30)
    sun = slot_levels(slots, weekday=6, base=0.05, ramp_minutes=30)
    at = lambda lv, hhmm: lv[GRID == int(hhmm[:2]) * 60 + int(hhmm[3:])][0]  # noqa: E731
    assert at(thu, "09:00") == pytest.approx(1.0) and at(thu, "13:00") == pytest.approx(0.5)
    assert at(thu, "03:00") == pytest.approx(0.05)
    assert at(sun, "09:00") == pytest.approx(0.05)       # weekday-only slot skipped on Sunday
    assert 0.05 < at(thu, "07:30") < 1.0                 # ramps instead of jumping


def test_random_levels_differ_day_to_day():
    z = {"base_range": [0.15, 0.6], "surges_per_day": 3}
    a, b = random_levels(np.random.default_rng(1), z), random_levels(np.random.default_rng(2), z)
    assert np.corrcoef(a, b)[0, 1] < 0.8
    assert (a >= 0).all() and a.max() <= 1.5


def test_maple_falls_trips_go_between_zones(tmp_path):
    with open("sim/maple_falls.yaml", encoding="utf-8") as f:
        sim = yaml.safe_load(f)
    n = write_zone_trips(tmp_path / "t.rou.xml", date(2026, 10, 1), sim["demand"],
                         np.random.default_rng(0), {"car": 'vClass="passenger"'})
    text = (tmp_path / "t.rou.xml").read_text()
    assert n > 1000 and 'from="in_J1"' in text and 'to="out_J3"' in text
    assert all(f'from="in_{j}" to="out_{j}"' not in text for j in sim["demand"]["zones"])  # never to itself
