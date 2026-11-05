"""Zone-based demand: each junction's neighbourhood starts and ends its own trips.

sim.yaml -> demand.zones, keyed by junction id (town network: trips start on in_<J>, end on out_<J>):

  routine  levels from time-slot tables, the same pattern every week:
             generation  trips leaving the zone (e.g. homes in the morning, offices in the evening)
             attraction  how strongly the zone draws trips (offices in the morning, homes in the evening)
           each slot: {from: "07:30", to: "10:30", level: 1.0, days: weekdays}
           days: all (default) | weekdays | weekend | saturday | sunday
  random   no time-of-day pattern: a drifting random level plus surges at random times,
           different every day

Levels are shares of the zone's peak_veh_per_hour. Between slots the level is `base_level`;
changes ramp over `ramp_minutes`.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

STEP_MIN = 5                                   # resolution of the level curves
GRID = np.arange(0, 1440, STEP_MIN)            # minute of day
DAYS = {"all": set(range(7)), "weekdays": set(range(5)), "weekend": {5, 6}, "saturday": {5}, "sunday": {6}}


def _minutes(hhmm: str) -> int:
    h, m = str(hhmm).split(":")[:2]
    return int(h) * 60 + int(m)


def slot_levels(slots: list[dict], weekday: int, base: float = 0.05, ramp_minutes: int = 30) -> np.ndarray:
    """Level per STEP_MIN of the day from a slot table (overlapping slots: highest wins)."""
    level = np.full(GRID.shape, float(base))
    for s in slots:
        if weekday not in DAYS[s.get("days", "all")]:
            continue
        a, b = _minutes(s["from"]), _minutes(s["to"])          # "24:00" = end of day
        inside = (GRID >= a) & (GRID < b)
        level[inside] = np.maximum(level[inside], float(s["level"]))
    k = max(1, ramp_minutes // STEP_MIN)
    padded = np.concatenate([level[-k:], level, level[:k]])        # the day wraps round midnight
    return np.convolve(padded, np.ones(k) / k, mode="same")[k:-k]


def random_levels(rng: np.random.Generator, z: dict) -> np.ndarray:
    """A level with no daily pattern: random base, AR(1) drift, random surges."""
    lo, hi = z.get("base_range", [0.15, 0.6])
    base = rng.uniform(lo, hi)
    phi, sd = float(z.get("drift_phi", 0.95)), float(z.get("drift_sd", 0.12))
    x = np.zeros(len(GRID))
    for i in range(1, len(GRID)):
        x[i] = phi * x[i - 1] + rng.normal(0.0, sd)
    level = base * np.exp(x)
    for _ in range(rng.poisson(float(z.get("surges_per_day", 3)))):
        start = rng.uniform(0, 1440)
        dur = rng.uniform(*z.get("surge_minutes", [15, 120]))
        mult = rng.uniform(*z.get("surge_factor", [1.5, 4.0]))
        level = np.where((GRID >= start) & (GRID < start + dur), level * mult, level)
    return np.clip(level, 0.0, float(z.get("max_level", 1.5)))


def zone_curves(day: date, zones: dict, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(generation, attraction) level per STEP_MIN for every zone on `day`."""
    out = {}
    for name, z in zones.items():
        if z.get("kind", "routine") == "random":
            out[name] = (random_levels(rng, z), random_levels(rng, z))
        else:
            base, ramp = float(z.get("base_level", 0.05)), int(z.get("ramp_minutes", 30))
            f = rng.normal(1.0, float(z.get("day_to_day_sd", 0.05)))     # routine: small daily wobble
            out[name] = (f * slot_levels(z.get("generation", []), day.weekday(), base, ramp),
                         slot_levels(z.get("attraction", z.get("generation", [])), day.weekday(), base, ramp))
    return out


def write_zone_trips(path: Path, day: date, demand: dict, rng: np.random.Generator, vtypes: dict,
                     begin_s: float = 0, end_s: float = 86400) -> int:
    """Trips departing in [begin_s, end_s): from each zone by its generation level, to another
    zone chosen by attraction level x size. Returns the trip count."""
    zones = demand["zones"]
    default_mix = demand.get("mix", {"car": 0.30, "two_wheeler": 0.45, "auto_rickshaw": 0.12, "bus": 0.05, "truck": 0.08})
    curves = zone_curves(day, zones, rng)
    names = list(zones)
    size = np.array([float(zones[n].get("peak_veh_per_hour", 500)) for n in names])
    trips = []
    for i, t0 in enumerate(GRID * 60):
        if t0 + STEP_MIN * 60 <= begin_s or t0 >= end_s:
            continue
        attract = np.array([curves[n][1][i] for n in names]) * size
        for zi, name in enumerate(names):
            z = zones[name]
            mix = z.get("mix", default_mix)
            types, probs = list(mix), np.array(list(mix.values()), dtype=float)
            n = rng.poisson(size[zi] * curves[name][0][i] * STEP_MIN / 60)
            w = attract.copy()
            w[zi] = 0.0
            if n == 0 or w.sum() <= 0:
                continue
            for dep in rng.uniform(max(t0, begin_s), min(t0 + STEP_MIN * 60, end_s), n):
                dst = names[rng.choice(len(names), p=w / w.sum())]
                trips.append((dep, f"in_{name}", f"out_{dst}", types[rng.choice(len(types), p=probs / probs.sum())]))
    trips.sort(key=lambda t: t[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write("<routes>\n")
        for vid, attrs in vtypes.items():
            f.write(f'  <vType id="{vid}" {attrs}/>\n')
        for k, (dep, src, dst, vt) in enumerate(trips):
            f.write(f'  <trip id="t{k}" type="{vt}" depart="{dep:.1f}" from={quoteattr(src)} '
                    f'to={quoteattr(dst)} departLane="best" departSpeed="max"/>\n')
        f.write("</routes>\n")
    return len(trips)
