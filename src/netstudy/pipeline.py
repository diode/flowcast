"""netstudy run: a city end to end (docs/implementation-next.md step 2), unattended and resumable.

Rendered days (the first `render_days`):  record (SUMO + CARLA)  ->  labels (SUMO's, with skip_render_read)
                                          ->  ingest (SAM 3.1)  ->  direct_check (vision against direct.py)
Other days (simulation):  sumo, SUMO alone: direct observations (direct.py), same table and columns as vision's
At the end:  state (edge_state, edge_view)  ->  KPIs  ->  score (vision against the labels)

Each finished (day, stage) is kept in table `stages` of traffic.db and skipped next time. The CARLA server is
started when a recording needs it and restarted if it has died (it has crashed when a client disconnected);
it is stopped before ingest, so SAM 3.1 has the GPU to itself.
"""
from __future__ import annotations

import socket
import sqlite3
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from .tools import carla_server

STAGES_SQL = "CREATE TABLE IF NOT EXISTS stages (day TEXT, stage TEXT, seconds REAL, finished TEXT, PRIMARY KEY (day, stage))"


def carla_gpu(sim: dict) -> str | None:
    """CARLA's card by (part of) its name: the town's recording.carla_gpu, else the workspace's (workspace.carla_gpu),
    else None (the vision GPU's name, from nvidia-smi)."""
    from . import workspace

    name = (sim.get("recording") or {}).get("carla_gpu") or workspace.carla_gpu()
    if name:
        return name
    try:
        from .glimmer_server import gpu_load
        return gpu_load()[workspace.vision_gpu()]["name"]
    except Exception:  # noqa: BLE001 - no nvidia-smi: DXGI adapter 0
        return None


class Carla:
    """The CARLA server as a child process, off screen, on the card named `gpu` (looked up in Windows' adapter
    order each start: the order can change between boots)."""

    def __init__(self, port: int = 2000, log=print, gpu: str | None = None):
        self.port, self.log, self.proc, self.gpu = port, log, None, gpu

    def up(self) -> bool:
        with socket.socket() as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self, wait_s: float = 300):
        if self.up():
            return
        from .dxgi import adapter_index

        adapter = adapter_index(self.gpu) if self.gpu else 0
        self.log(f"starting the CARLA server on {self.gpu} (-graphicsadapter={adapter})")
        self.proc = subprocess.Popen([str(carla_server()), "-RenderOffScreen", f"-graphicsadapter={adapter}",
                                      f"-carla-rpc-port={self.port}"], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        t0 = time.time()
        while not self.up():
            if time.time() - t0 > wait_s:
                raise SystemExit("the CARLA server did not come up")
            time.sleep(2)
        time.sleep(10)                                   # the port opens before the world is ready

    def stop(self):
        subprocess.run(["taskkill", "/F", "/IM", "CarlaUE4-Win64-Shipping.exe"], capture_output=True)
        subprocess.run(["taskkill", "/F", "/IM", "CarlaUE4.exe"], capture_output=True)
        self.proc = None
        time.sleep(3)


def done_stages(db: Path) -> set[tuple[str, str]]:
    with sqlite3.connect(db) as c:
        c.execute(STAGES_SQL)
        return {(d, s) for d, s in c.execute("SELECT day, stage FROM stages")}


def mark(db: Path, day: str, stage: str, secs: float):
    with sqlite3.connect(db) as c:
        c.execute(STAGES_SQL)
        c.execute("INSERT OR REPLACE INTO stages VALUES (?, ?, ?, ?)",
                  (day, stage, round(secs, 1), datetime.now().isoformat(timespec="seconds")))


def render_cameras(sim: dict, cams: list) -> list[str] | None:
    """The cameras filmed on a rendered day (source.render_cameras): N picked at random with the town's seed, or a
    list of camera ids; None (not set) films every camera. The rest get SUMO's direct observations that day."""
    choice = (sim.get("source") or {}).get("render_cameras")
    ids = [c["camera_id"] for c in cams]
    if choice is None:
        return None
    if isinstance(choice, list):
        bad = sorted(set(map(str, choice)) - set(ids))
        if bad:
            raise SystemExit(f"source.render_cameras: no camera {bad}")
        return [str(x) for x in choice]
    import numpy as np

    n = int(choice)
    if not 1 <= n <= len(ids):
        raise SystemExit(f"source.render_cameras: {n}, expected 1..{len(ids)}")
    return sorted(np.random.default_rng(int(sim.get("seed", 0))).choice(ids, size=n, replace=False).tolist())


def _record(day: str, sim: dict, out_dir: Path, cams: list, port: int, film: list[str] | None = None) -> dict:
    from .record import record_day
    return record_day(date.fromisoformat(day), sim, out_dir, cams, port=port, film=film)


def _ingest(day: str, sim: dict, out_dir: Path, cams: list) -> int:
    from .ingest import Ingest
    return Ingest(sim, out_dir, cams).run(day, log_fn=lambda m: None)


def direct_rest(sim: dict, out_dir: Path, day: str, cams: list, film: list[str]) -> int:
    """A rendered day with only some cameras filmed: SUMO's direct observations (source 'sumo') for the others, from
    the same planned day; incidents come from the recording's sidecar, as for the filmed cameras."""
    from .direct import direct_day, write_rows

    rows, _ = direct_day(date.fromisoformat(day), sim, out_dir, cams)
    rows = rows[~rows.camera_id.isin(set(film))]
    write_rows(out_dir / "traffic.db", day, rows)
    return len(rows)


def _sumo_day(day: str, sim: dict, out_dir: Path, cams: list) -> tuple:
    """One day in SUMO alone, no CARLA and no video: what the cameras would measure (direct.py), and its incidents."""
    from .direct import direct_day
    return direct_day(date.fromisoformat(day), sim, out_dir, cams)


def sumo_only(sim: dict, out_dir: Path, days: list[str], cams: list, workers: int = 4, log=print):
    """Days without rendering: observations with source 'sumo' (same table and columns as vision's), incidents
    into labels.db (label_source 'sumo')."""
    from concurrent.futures import ProcessPoolExecutor

    from .direct import write_rows
    from .ingest import SCHEMA
    from .labels import connect

    db = out_dir / "traffic.db"
    with sqlite3.connect(db) as c:
        c.executescript(SCHEMA)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_sumo_day, d, sim, out_dir, cams): d for d in days}
        for fut, day in futures.items():
            t0 = time.time()
            rows, incidents = fut.result()
            write_rows(db, day, rows)
            now = datetime.now().isoformat(timespec="seconds")
            with connect(out_dir) as c:
                c.executemany("INSERT OR REPLACE INTO incident_log VALUES (?, ?, ?, ?, 'sumo', 'sumo', ?)",
                              [(x["edge"], x["kind"], x["start"], x["end"], now) for x in incidents])
            mark(db, day, "sumo", time.time() - t0)
            log(f"{day} SUMO only: {len(rows)} camera rows, {len(incidents)} incidents")


def _child_main(q, fn, args):
    q.put(fn(*args))


def in_child(fn, *args):
    """fn(*args) in a fresh Python process: a native crash (CARLA client, CUDA) becomes an exception here, and
    the GPU memory and CARLA connections of one day are gone before the next stage starts."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child_main, args=(q, fn, args))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"{fn.__name__} exited with code {p.exitcode}")
    return q.get(timeout=60)


def run(sim: dict, out_dir: Path, first: date, days: int, redo: str | None = None, port: int = 2000, log=print,
        render_days: int | None = None):
    """Simulation only: the first `render_days` days (source.render_days, or the argument; default: all) are
    rendered and read from video, the rest run in SUMO alone."""
    source = sim.get("source") or {}
    if render_days is None:
        render_days = source.get("render_days")
    if render_days is not None and source.get("kind") != "simulation":
        raise SystemExit("render_days is a simulation setting: in real mode the video is the cameras' own "
                         "recordings, and every recorded day is read")
    from .labels import from_sim
    from .network import load_cameras

    db = out_dir / "traffic.db"
    cams = load_cameras(out_dir / "cameras.json")
    carla = Carla(port, log, carla_gpu(sim))
    if redo:
        with sqlite3.connect(db) as c:
            c.execute(STAGES_SQL)
            c.execute("DELETE FROM stages WHERE stage = ?", (redo,))
    labels_on = bool((sim.get("source") or {}).get("skip_render_read"))
    film = render_cameras(sim, cams)
    if film is not None:
        log(f"rendered days film {len(film)} of {len(cams)} cameras: {', '.join(film)} "
            f"({', '.join(c['edge'] for c in cams if c['camera_id'] in film)}); the rest direct from SUMO")
    rendered = days if render_days is None else min(days, render_days)
    rest = [(first + timedelta(days=k)).isoformat() for k in range(rendered, days)]
    rest = [d for d in rest if (d, "sumo") not in done_stages(db)]
    if rest:
        log(f"{len(rest)} days in SUMO alone")
        sumo_only(sim, out_dir, rest, cams, log=log)
    for k in range(rendered):
        day = (first + timedelta(days=k)).isoformat()
        done = done_stages(db)
        if (day, "record") not in done:
            for attempt in (1, 2):                       # one retry with a fresh server
                carla.start()
                t0 = time.time()
                try:
                    stats = in_child(_record, day, sim, out_dir, cams, port, film)
                    mark(db, day, "record", time.time() - t0)
                    log(f"{day} recorded: {stats['segments']} segments in {time.time() - t0:.0f} s")
                    break
                except Exception as e:  # noqa: BLE001 - the server may have died; try once more
                    log(f"{day} recording failed ({e}); restarting CARLA")
                    carla.stop()
                    if attempt == 2:
                        raise
        if labels_on and (day, "labels") not in done:
            t0 = time.time()
            n = from_sim(sim, out_dir, day)
            mark(db, day, "labels", time.time() - t0)
            log(f"{day} labels: {n}")
        if (day, "ingest") not in done:
            carla.stop()                                 # SAM 3.1 gets the GPU
            t0 = time.time()
            n = in_child(_ingest, day, sim, out_dir, cams)
            mark(db, day, "ingest", time.time() - t0)
            log(f"{day} ingested: {n} segments in {time.time() - t0:.0f} s")
        if film is not None and (day, "direct_rest") not in done_stages(db):
            t0 = time.time()
            n = direct_rest(sim, out_dir, day, cams, film)
            mark(db, day, "direct_rest", time.time() - t0)
            log(f"{day} direct observations for the {len(cams) - len(film)} cameras not filmed: {n} rows")
        if render_days is not None and (day, "direct_check") not in done_stages(db):
            from .direct import direct_check
            t0 = time.time()
            a = direct_check(sim, out_dir, day, cams).iloc[-1]
            mark(db, day, "direct_check", time.time() - t0)
            log(f"{day} vision vs direct observations: flow WAPE {a.flow_wape:.2f} (bias {a.flow_bias:+.2f}), "
                f"speed MAE {a.speed_mae_kmh:.1f} km/h, count MAE {a.count_mae:.2f} over {int(a.slots)} slots")
    carla.stop()
    finish(sim, out_dir, log)


def finish(sim: dict, out_dir: Path, log=print):
    """State, KPIs and scores over everything ingested so far."""
    from .graph import build_state
    from .kpis import build_kpis
    from .score import vision

    db = out_dir / "traffic.db"
    state, _ = build_state(db, None, None, None, float(sim["network"].get("view_m", 60)))
    log(f"edge_state: {state.ts.nunique()} slots x {state.edge_id.nunique()} edges")
    k = build_kpis(db)
    log(f"network_kpi: {len(k)} slots")
    if (sim.get("source") or {}).get("skip_render_read"):
        summ, _ = vision(out_dir, "sumo")
        a = summ[summ.camera_id == "all"].iloc[0]
        log(f"vision vs SUMO labels: flow WAPE {a.flow_wape:.2f} (bias {a.flow_bias:+.2f}) over {int(a.flow_slots)} "
            f"slots, speed MAE {a.speed_mae_kmh:.1f} km/h, count MAE {a.count_mae:.2f}")
