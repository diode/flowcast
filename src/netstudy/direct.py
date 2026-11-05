"""Direct observations (docs/architecture-next.md 1.3, render_days): a simulated day that is not rendered gets
its `observations` straight from SUMO, as a perfect camera would measure them. SUMO only makes the sample data:
the rows have the same table, columns and definitions as vision's (ingest.py), only `source = 'sumo'`.

  samples       at the same times as ingest (every `pair_every_s` on the clock), per camera: each vehicle on
                its approach lanes within `view_m` of the stop line (type, lane, metres before the stop line)
                and its speed, as SAM 3.1 sees vehicles and ingest matches them
  observations  per 5-minute slot, from those samples with ingest's own code (ingest.slot_values): the snapshot
                nearest the slot end, Edie's flow over the stretch, the mean speed between the speed lines

Only the cameras' edges are measured; nothing for roads without a camera, no whole-edge values, no SUMO
outputs kept (the day's work folder is deleted).

direct_check: on a rendered day, both vision's and SUMO's direct observations, per camera and slot ->
table direct_check. Their difference is the vision error the direct days do not have.
"""
from __future__ import annotations

import shutil
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import LENGTH_M, OBS_COLUMNS, ingest_settings, slot_values

TYPES = set(LENGTH_M)               # what vision can tell apart (vision.PROMPTS); anything else looks like a car


class PerfectCamera:
    """What each camera would see in one sample, read from SUMO: the vehicles ingest would place on its approach
    (Zone: lane, metres before the stop line) and their speeds (ingest matches them between two frames)."""

    def __init__(self, traci, cameras: list[dict], lane_len: dict[str, float], view_m: float, band: tuple):
        self.traci, self.cameras, self.lane_len, self.view_m = traci, cameras, lane_len, view_m
        self.near, self.far = band

    def sample(self, cam: dict) -> tuple[list[dict], dict]:
        traci = self.traci
        vehicles, every, band = [], [], []
        for i, ln in enumerate(cam["lanes"]):
            for v in traci.lane.getLastStepVehicleIDs(ln):
                m = self.lane_len[ln] - traci.vehicle.getLanePosition(v)
                if m > self.view_m:
                    continue
                vt = traci.vehicle.getTypeID(v)
                vehicles.append({"type": vt if vt in TYPES else "car", "lane": i, "m": round(m, 1)})
                kmh = round(3.6 * traci.vehicle.getSpeed(v), 1)
                every.append(kmh)
                if self.near <= m <= self.far:
                    band.append(kmh)
        return vehicles, {"band": band, "all": every}


def sumo_args(cfg: Path, step: float) -> list[str]:
    """SUMO started as the recording starts it (CARLA's co-simulation, sumo_simulation.py), so a direct day's
    traffic follows the same rules as a filmed one: step 1/fps, its lateral resolution and collision check,
    SUMO's own seed."""
    return ["--configuration-file", str(cfg), "--step-length", str(step), "--lateral-resolution", "0.25",
            "--collision.check-junctions"]


def direct_day(day: date, sim: dict, out_dir: Path, cameras: list[dict], end_s: float = 86400,
               incidents: list[dict] | None = None, tag: str | None = None, on_step=None,
               stats: dict | None = None) -> tuple:
    """One day in SUMO alone: (observations rows in OBS_COLUMNS order, the day's incidents for labels.db).
    For paired runs (whatif_labels.py): `incidents` replaces the random ones (plan_day), `tag` names the work
    folder, `on_step(traci, t)` runs after every step, and `stats` receives SUMO's teleports as (seconds, edge)."""
    import sumolib

    from .record import _incident_labels, measure_settings, recording_settings
    from .simulate import CameraMeter, DEFAULT_LEVELS, DEFAULT_PCU, _traci, car_lanes, plan_day, sumo_binary

    cfg = ingest_settings(sim)
    step = 1.0 / int(recording_settings(sim)["fps"])
    every, slot = float(cfg["pair_every_s"]), int(cfg["slot_s"])
    view_m = float(sim["network"].get("view_m", 60))
    band = tuple(sorted(float(x) for x in measure_settings(sim)["speed_lines_m"]))
    net_file = str(out_dir / "network" / "net.net.xml")
    work = out_dir / "work" / f"direct_{tag or f'{day:%Y%m%d}'}"
    midnight = datetime.combine(day, datetime.min.time())
    plan = plan_day(day, net_file, cameras, sim, str(work), end_s=end_s,    # seeded by the date: as the recording's
                    incidents=incidents, tag=tag)
    lanes = {e.getID(): car_lanes(e) for e in sumolib.net.readNet(net_file).getEdges()}
    incidents = _incident_labels(plan, midnight, lanes)
    traci = _traci()
    meter = CameraMeter(traci, cameras, plan, view_m, DEFAULT_PCU, DEFAULT_LEVELS)   # incidents only
    camera = PerfectCamera(traci, cameras, plan["lane_len"], view_m, band)
    times = sorted({round(k * every, 3) for k in range(1, int(end_s // every) + 1)} |
                   {float(t) for t in range(60, int(end_s) + 1, 60)})       # samples; incidents each minute, as recording
    samples = {c["camera_id"]: [] for c in cameras}
    rows = []
    extra = ["--no-warnings", "false", "--error-log", str(work / "messages.log")] if stats is not None else []
    traci.start([sumo_binary("sumo"), *sumo_args(plan["cfg"], step), *extra])
    try:
        for t in times:
            traci.simulationStep(t)
            if t % 60 == 0:
                meter.apply_incidents(t)
            if on_step:
                on_step(traci, t)
            if abs(t / every - round(t / every)) < 1e-6:
                ts = midnight + timedelta(seconds=t)
                for c in cameras:
                    samples[c["camera_id"]].append((ts, *camera.sample(c)))
            if t % slot == 0:
                ts = midnight + timedelta(seconds=t)
                rows += _slot_rows(cameras, samples, ts, slot, view_m, int(cfg["min_samples"]))
    finally:
        traci.close()
        if stats is not None and (work / "messages.log").exists():
            import re
            log_text = (work / "messages.log").read_text(encoding="utf-8", errors="replace")
            stats["teleports"] = [(float(t), e) for e, t in   # (seconds, edge) of every teleport
                                  re.findall(r"Teleporting vehicle '[^']*';.*?lane='([^']*?)_\d+'.*?time=(\d+(?:\.\d+)?)",
                                             log_text)]
        shutil.rmtree(work, ignore_errors=True)       # only the observations are kept
    return pd.DataFrame(rows, columns=OBS_COLUMNS), incidents


def _slot_rows(cameras, samples, ts, slot, view_m, min_samples) -> list[tuple]:
    lo = ts - timedelta(seconds=slot)
    out = []
    for c in cameras:
        cid = c["camera_id"]
        mine = [s for s in samples[cid] if lo < s[0] <= ts]
        samples[cid] = [s for s in samples[cid] if s[0] > ts]
        if len(mine) >= min_samples:
            v = slot_values(c, mine, ts, slot, view_m)
            out.append((cid, ts.isoformat(timespec="seconds"), "", "sumo", *v[:8], "", *v[8:], "sumo"))
    return out


def write_rows(db: Path, day: str, rows: pd.DataFrame):
    """Replace the day's direct observations (vision's rows of that day are left alone). A day's slots end
    after 00:00 up to 24:00, which is stamped with the next day's date."""
    d = date.fromisoformat(day)
    with sqlite3.connect(db) as c:
        c.execute("DELETE FROM observations WHERE source = 'sumo' AND ts > ? AND ts <= ?",
                  (f"{d}T00:00:00", f"{d + timedelta(days=1)}T00:00:00"))
        c.executemany(f"INSERT OR REPLACE INTO observations ({','.join(OBS_COLUMNS)}) "
                      f"VALUES ({','.join('?' * len(OBS_COLUMNS))})",
                      rows[OBS_COLUMNS].astype(object).where(rows[OBS_COLUMNS].notna(), None)
                      .itertuples(index=False, name=None))


def direct_check(sim: dict, out_dir: Path, day: str, cameras: list[dict]) -> pd.DataFrame:
    """A rendered day: vision's observations against SUMO's direct ones for the same camera and slot ->
    table direct_check (per slot, plus camera_id 'all' rows in direct_check_summary). The direct rows are not
    written to observations."""
    rows, _ = direct_day(date.fromisoformat(day), sim, out_dir, cameras)
    db = out_dir / "traffic.db"
    with sqlite3.connect(db) as c:
        vis = pd.read_sql("SELECT camera_id, ts, total, flow, speed FROM observations "
                          "WHERE source = 'vision' AND substr(ts, 1, 10) = ?", c, params=(day,))
    both = vis.merge(rows[["camera_id", "ts", "total", "flow", "speed"]], on=["camera_id", "ts"],
                     suffixes=("_vision", "_direct"))
    both["day"] = day

    def summary(g: pd.DataFrame, cid: str) -> dict:
        f, s = g.dropna(subset=["flow_vision", "flow_direct"]), g.dropna(subset=["speed_vision", "speed_direct"])
        lab = f.flow_direct.sum()
        return {"day": day, "camera_id": cid, "slots": len(g),
                "flow_wape": (f.flow_vision - f.flow_direct).abs().sum() / lab if lab else np.nan,
                "flow_bias": (f.flow_vision.sum() - lab) / lab if lab else np.nan,
                "speed_mae_kmh": (s.speed_vision - s.speed_direct).abs().mean() if len(s) else np.nan,
                "count_mae": (g.total_vision - g.total_direct).abs().mean() if len(g) else np.nan,
                "checked_at": datetime.now().isoformat(timespec="seconds")}

    summ = pd.DataFrame([summary(g, cid) for cid, g in both.groupby("camera_id")] + [summary(both, "all")])
    with sqlite3.connect(db) as c:
        for table, df in (("direct_check", both), ("direct_check_summary", summ)):
            if c.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
                c.execute(f"DELETE FROM {table} WHERE day = ?", (day,))
            df.to_sql(table, c, if_exists="append", index=False)
    return summ
