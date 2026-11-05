"""Real scoring (docs/architecture-next.md 1.5): measurements against labels only (labels.db), never a simulator
file. Every result names its `label_source`; one score never mixes people's and SUMO's labels.

  vision   per camera and 5-minute slot, observations (source = vision) against the labels of the same slot:
             flow    vision's flow against the vehicles over the counting line (slots whose segments are all
                     labelled)
             speed   vision's mean speed against the labelled speeds between the speed lines
             count   vehicles on the approach in the slot's frame against the frame label at that time
           -> vision_eval (per camera and 'all'), vision_eval_slots (per slot)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import connect as labels_db


def vision(out_dir: Path, label_source: str = "sumo", slot_s: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(out_dir / "traffic.db") as c:
        obs = pd.read_sql("SELECT camera_id, ts, total, flow, speed FROM observations WHERE source = 'vision'", c)
    with labels_db(out_dir) as c:
        cross = pd.read_sql("SELECT camera_id, segment_start, seconds, vehicles FROM crossing_labels "
                            "WHERE label_source = ?", c, params=(label_source,))
        speeds = pd.read_sql("SELECT camera_id, t, kmh FROM speed_labels WHERE label_source = ?", c, params=(label_source,))
        frames = pd.read_sql("SELECT camera_id, t, counts_json FROM frame_labels WHERE label_source = ?", c,
                             params=(label_source,))
    cross["start"] = pd.to_datetime(cross.segment_start)
    speeds["t"] = pd.to_datetime(speeds.t)
    frames = frames.set_index(["camera_id", "t"])
    rows = []
    for o in obs.itertuples():
        ts = datetime.fromisoformat(o.ts)
        lo = ts - timedelta(seconds=slot_s)
        seg = cross[(cross.camera_id == o.camera_id) & (cross.start >= lo) & (cross.start < ts)]
        full = abs(seg.seconds.sum() - slot_s) < 1
        sp = speeds[(speeds.camera_id == o.camera_id) & (speeds.t > lo) & (speeds.t <= ts)].kmh
        key = (o.camera_id, o.ts)
        count_label = sum(json.loads(frames.loc[key, "counts_json"]).values()) if key in frames.index else np.nan
        rows.append({"camera_id": o.camera_id, "ts": o.ts,
                     "flow_vision": o.flow, "flow_label": float(seg.vehicles.sum()) if full else np.nan,
                     "speed_vision": o.speed, "speed_label": float(sp.mean()) if len(sp) else np.nan,
                     "count_vision": o.total, "count_label": count_label})
    slots = pd.DataFrame(rows, columns=["camera_id", "ts", "flow_vision", "flow_label", "speed_vision", "speed_label",
                                        "count_vision", "count_label"])
    slots["label_source"] = label_source

    def summary(g: pd.DataFrame, cid: str) -> dict:
        f = g.dropna(subset=["flow_label", "flow_vision"])
        s = g.dropna(subset=["speed_label", "speed_vision"])
        n = g.dropna(subset=["count_label", "count_vision"])
        lab = f.flow_label.sum()
        return {"camera_id": cid, "slots": len(g), "flow_slots": len(f), "flow_label": lab, "flow_vision": f.flow_vision.sum(),
                "flow_wape": (f.flow_vision - f.flow_label).abs().sum() / lab if lab else np.nan,
                "flow_bias": (f.flow_vision.sum() - lab) / lab if lab else np.nan,
                "speed_mae_kmh": (s.speed_vision - s.speed_label).abs().mean() if len(s) else np.nan,
                "count_mae": (n.count_vision - n.count_label).abs().mean() if len(n) else np.nan,
                "label_source": label_source, "scored_at": datetime.now().isoformat(timespec="seconds")}

    summ = pd.DataFrame([summary(g, cid) for cid, g in slots.groupby("camera_id")] + [summary(slots, "all")])
    with sqlite3.connect(out_dir / "traffic.db") as c:
        for table, df in (("vision_eval", summ), ("vision_eval_slots", slots)):
            if c.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
                c.execute(f"DELETE FROM {table} WHERE label_source = ?", (label_source,))
            df.to_sql(table, c, if_exists="append", index=False)
    return summ, slots
