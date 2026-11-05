"""The calendar and road levels shared by Predict and the graph model: day kinds, (mock) events, holiday factors,
time-of-day words, and what busy / slow / jammed / closed mean. No simulation imports (graph_model.py uses this)."""
from __future__ import annotations

from datetime import date

import pandas as pd

LEVELS = {"busy": 0.6, "slow": 0.35, "jammed": None, "closed": None}     # share of the speed limit
JAMMED_KMH = 5.0
# Holiday: a Sunday's trips, a quieter morning, a busier evening and late evening.
HOLIDAY_FACTORS = [1.0] * 6 + [0.6, 0.6, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.25, 1.3, 1.3, 1.35, 1.35,
                               1.25, 1.1]
CAL_FROM, CAL_TO = date(2026, 11, 1), date(2026, 12, 31)

# Mock events (not real dates). venue: index into the town's venue list (venue_junctions).
EVENTS = [
    {"date": "2026-11-08", "kind": "festival", "name": "Festival of Lights", "start": "17:00", "end": "22:00",
     "venue": 0, "share": 0.35},
    {"date": "2026-11-16", "kind": "holiday", "name": "Founders' Day"},
    {"date": "2026-11-27", "kind": "festival", "name": "Harvest Fair", "start": "10:00", "end": "18:00",
     "venue": 2, "share": 0.25},
    {"date": "2026-12-04", "kind": "festival", "name": "Lantern Night", "start": "18:00", "end": "23:00",
     "venue": 4, "share": 0.3},
    {"date": "2026-12-12", "kind": "festival", "name": "Winter Market", "start": "09:00", "end": "14:00",
     "venue": 3, "share": 0.25},
    {"date": "2026-12-25", "kind": "holiday", "name": "Christmas Day"},
    {"date": "2026-12-31", "kind": "festival", "name": "New Year's Eve", "start": "20:00", "end": "23:55",
     "venue": 0, "share": 0.35},
]


# ------------------------------------------------------------------ calendar
def event_on(d: date) -> dict | None:
    return next((e for e in EVENTS if e["date"] == d.isoformat()), None)


def day_kind(d: date) -> str:
    e = event_on(d)
    if e:
        return e["kind"]
    return "weekend" if d.weekday() >= 5 else "working"


def calendar(start: date = CAL_FROM, end: date = CAL_TO) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": [x.date() for x in days], "kind": [day_kind(x.date()) for x in days],
                         "event": [(event_on(x.date()) or {}).get("name", "") for x in days]})


def _s(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    return 3600.0 * h + 60 * m


def time_context(kind: str, hhmm: str, event: dict | None = None) -> str:
    """Plain words for the time of day on this kind of day."""
    t = _s(hhmm) / 3600
    if event and event.get("start") and _s(event["start"]) - 3600 <= t * 3600 <= _s(event["end"]) + 3600:
        if t * 3600 < _s(event["start"]):
            return f"crowds arriving for {event['name']}"
        if t * 3600 <= _s(event["end"]):
            return f"during {event['name']}"
        return f"crowds leaving {event['name']}"
    if kind == "working":
        bands = [(5, "night"), (7, "early morning"), (10, "working-day morning peak"), (16, "working-day daytime"),
                 (20, "working-day evening peak"), (23, "working-day evening"), (24, "night")]
    else:
        word = "holiday" if kind in ("holiday", "festival") else "weekend"
        bands = [(6, "night"), (10, f"quiet {word} morning"), (16, f"{word} daytime"), (20, f"{word} evening outing"),
                 (23.5, f"{word} after-evening"), (24, "night")]
    return next(w for lim, w in bands if t < lim)
