"""The label store (docs/architecture-next.md 1.5): data/<city>/labels.db. People write it with the label tool;
with `source.skip_render_read: true` SUMO writes it instead (`netstudy labels --from-sim`), from the label
sidecars the recorder wrote next to each segment. Every row says which (`label_source` = people | sumo).

  frame_labels     vehicles of each type on the camera's approach in one frame
  crossing_labels  vehicles over the counting line in one segment
  speed_labels     one vehicle's speed between the speed lines
  incident_log     incidents: road, kind, start, end
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS frame_labels (camera_id TEXT, t TEXT, counts_json TEXT, label_source TEXT,
    labeller TEXT, labelled_at TEXT, PRIMARY KEY (camera_id, t, label_source));
CREATE TABLE IF NOT EXISTS crossing_labels (camera_id TEXT, segment_start TEXT, seconds REAL, vehicles INTEGER,
    types_json TEXT, label_source TEXT, labeller TEXT, labelled_at TEXT,
    PRIMARY KEY (camera_id, segment_start, label_source));
CREATE TABLE IF NOT EXISTS speed_labels (camera_id TEXT, t TEXT, type TEXT, kmh REAL, label_source TEXT,
    labeller TEXT, labelled_at TEXT);
CREATE TABLE IF NOT EXISTS incident_log (edge TEXT, kind TEXT, start TEXT, end TEXT, label_source TEXT,
    labeller TEXT, labelled_at TEXT, PRIMARY KEY (edge, start, label_source));
"""


def connect(out_dir: Path) -> sqlite3.Connection:
    c = sqlite3.connect(out_dir / "labels.db")
    c.executescript(SCHEMA)
    return c


def from_sim(sim: dict, out_dir: Path, day: str | None = None) -> dict[str, int]:
    """Load the recorder's SUMO label sidecars into labels.db (label_source = sumo)."""
    if (sim.get("source") or {}).get("kind", "simulation") != "simulation":
        raise SystemExit("labels --from-sim: this city is real, there is no simulation to take labels from")
    if not (sim.get("source") or {}).get("skip_render_read"):
        raise SystemExit("labels --from-sim needs source.skip_render_read: true (otherwise people label)")
    now = datetime.now().isoformat(timespec="seconds")
    n = {"segments": 0, "frames": 0, "speeds": 0, "incidents": 0}
    rec = out_dir / "recordings"
    with connect(out_dir) as c:
        for f in sorted(rec.glob(f"{day or '*'}/*/*.labels.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            start = datetime.fromisoformat(d["segment_start"])
            midnight = start.replace(hour=0, minute=0, second=0)
            cid = d["camera_id"]
            c.execute("DELETE FROM speed_labels WHERE camera_id = ? AND t >= ? AND t < ? AND label_source = 'sumo'",
                      (cid, start.isoformat(), (start + timedelta(seconds=d["frames"] / d["fps"])).isoformat()))
            types: dict[str, int] = {}
            for x in d["crossings"]:
                types[x["type"]] = types.get(x["type"], 0) + 1
            c.execute("INSERT OR REPLACE INTO crossing_labels VALUES (?, ?, ?, ?, ?, 'sumo', 'sumo', ?)",
                      (cid, start.isoformat(), d["frames"] / d["fps"], len(d["crossings"]), json.dumps(types), now))
            for x in d["frame_labels"]:
                t = (midnight + timedelta(seconds=x["t"])).isoformat(timespec="seconds")
                c.execute("INSERT OR REPLACE INTO frame_labels VALUES (?, ?, ?, 'sumo', 'sumo', ?)",
                          (cid, t, json.dumps(x["counts"]), now))
                n["frames"] += 1
            for x in d["speeds"]:
                t = (midnight + timedelta(seconds=x["t"])).isoformat(timespec="milliseconds")
                c.execute("INSERT INTO speed_labels VALUES (?, ?, ?, ?, 'sumo', 'sumo', ?)", (cid, t, x["type"], x["kmh"], now))
                n["speeds"] += 1
            n["segments"] += 1
        for f in sorted(rec.glob(f"{day or '*'}/incidents.labels.json")):
            for x in json.loads(f.read_text(encoding="utf-8")):
                c.execute("INSERT OR REPLACE INTO incident_log VALUES (?, ?, ?, ?, 'sumo', 'sumo', ?)",
                          (x["edge"], x["kind"], x["start"], x["end"], now))
                n["incidents"] += 1
    return n
