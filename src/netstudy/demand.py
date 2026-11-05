"""Change a day's trips before it is simulated (docs/predict.md): the same edit works for our zone
trips and v1's, since both are SUMO <trip from= to=> files.

  hour_factors  24 numbers: trips departing in hour h are kept or copied so their number is x factor
                (0.8: one in five dropped; 1.3: three in ten copied, departing up to a minute later)
  venue         an event: {to: [edges], from: [edges], arrive: [s0, s1], leave: [s0, s1], share}
                trips departing in `arrive` are copied with probability `share` and sent to a venue
                edge instead; trips departing in `leave` are copied and start from one
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TRIP = re.compile(r'<trip id="([^"]+)"[^>]*depart="([0-9.]+)"[^>]*/>')


def _set(line: str, attr: str, value: str) -> str:
    return re.sub(rf'{attr}="[^"]*"', f'{attr}="{value}"', line, count=1)


def edit_trips(path: Path, spec: dict, seed: int = 0) -> int:
    """Rewrite the trips file in place; returns the new number of trips."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    head = [ln for ln in lines if "<trip " not in ln and "</routes>" not in ln]
    trips = [(float(m.group(2)), m.group(1), ln) for ln in lines if (m := TRIP.search(ln))]
    rng = np.random.default_rng(seed + 1)
    factors = spec.get("hour_factors")
    venue = spec.get("venue")
    out = []
    for depart, tid, ln in trips:
        f = float(factors[min(int(depart // 3600), 23)]) if factors else 1.0
        n = int(f) + int(rng.random() < f - int(f))
        for k in range(n):
            if k == 0:
                out.append((depart, ln))
            else:
                d = depart + float(rng.uniform(1, 60))
                out.append((d, _set(_set(ln, "id", f"{tid}_x{k}"), "depart", f"{d:.1f}")))
        if venue and n:
            for key, attr, edges in (("arrive", "to", venue["to"]), ("leave", "from", venue["from"])):
                s0, s1 = venue[key]
                if edges and s0 <= depart < s1 and rng.random() < float(venue.get("share", 0.3)):
                    other = 'from' if attr == 'to' else 'to'
                    keep = re.search(rf'{other}="([^"]*)"', ln).group(1)
                    choice = [e for e in edges if e != keep]
                    if choice:
                        e = choice[int(rng.integers(len(choice)))]
                        d = depart + float(rng.uniform(1, 60))
                        new = _set(_set(_set(ln, "id", f"{tid}_{key}"), "depart", f"{d:.1f}"), attr, e)
                        out.append((d, new))
    out.sort(key=lambda x: x[0])
    path.write_text("\n".join(head + [ln for _, ln in out] + ["</routes>"]) + "\n", encoding="utf-8")
    return len(out)
