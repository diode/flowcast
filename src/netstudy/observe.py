"""A real town's observations from its pushed recordings, with SAM 3.1 and no SUMO (netstudy observe).

  observe(sim, out_dir)   pickup (inbox -> recordings)  ->  ingest (SAM 3.1 -> vision_samples, observations)
                          ->  state + KPIs (edge_state, edge_view, network_kpi)  ->  graph model (Predict)
                          once, or with `follow` again every `poll_s` while recordings keep arriving
  status(sim, out_dir)    the onboarding checklist the technical view shows: each step done or not, what it found,
                          and the action that does it

SAM 3.1 runs on the workspace's vision GPU (workspace.vision_gpu, or a town's `ingest.gpu`; nvidia-smi / PCI
numbering): CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES are set before torch loads. With
workspace.one_at_a_time, ingest waits while another GPU is busy (the Traffic Assistant answering, a CARLA render).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

MIN_DAYS = 7                    # Predict's usual day needs a week of a day kind at least (graph_model.fit)


def use_gpu(sim: dict) -> int:
    """Point CUDA at the vision GPU (PCI numbering) unless the caller already chose; returns its number."""
    from . import workspace

    gpu = int((sim.get("ingest") or {}).get("gpu", workspace.vision_gpu()))
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    return gpu


def other_card_busy(gpu: int) -> str | None:
    """Why the other card counts as under heavy load (utilization with power; memory held alone is idle), else None."""
    import statistics

    from .glimmer_server import BUSY, gpu_load

    try:
        seen = []
        for _ in range(3):
            seen.append(next((g for i, g in gpu_load().items() if i != gpu), None))
            time.sleep(0.4)
    except (OSError, ValueError):                      # no nvidia-smi: nothing to check
        return None
    seen = [g for g in seen if g]
    if not seen:
        return None
    util = statistics.median(g["util_pct"] for g in seen)
    watts = statistics.median(g["watts"] for g in seen)
    if util > BUSY["util_pct"] and watts > BUSY["watts"]:
        return f"{seen[0]['name']} is under load ({util:.0f} %, {watts:.0f} W)"
    return None


def observe(sim: dict, out_dir: Path, follow: bool = False, poll_s: float = 30.0, refit_every_s: float = 3600,
            detector=None, log=print, check_gpu: bool = True) -> dict:
    from .graph_model import GraphModel, model_path
    from .inbox import pickup
    from .ingest import Ingest
    from .network import load_cameras
    from .pipeline import finish

    cams = load_cameras(out_dir / "cameras.json")
    gpu = use_gpu(sim)
    ing = Ingest(sim, out_dir, cams, detector=detector)
    if ing.skipped:
        log(f"not calibrated, their recordings wait: {', '.join(ing.skipped)}")
    total = {"stored": 0, "rejected": 0, "ingested": 0}
    last_fit = 0.0
    while True:
        got = pickup(sim, out_dir, cams, log=log)
        total["stored"] += got["stored"]
        total["rejected"] += got["rejected"]
        from . import workspace
        busy = other_card_busy(gpu) if check_gpu and detector is None and workspace.one_at_a_time() else None
        n = 0
        if busy:
            log(f"ingest waits: {busy} (workspace: one GPU under load at a time)")
        else:
            n = ing.run(log_fn=log)
            total["ingested"] += n
        if n:
            finish(sim, out_dir, log)
        if n and time.time() - last_fit > (refit_every_s if follow else 0):
            try:
                m = GraphModel.fit(sim, out_dir)
                m.save(model_path(out_dir))
                last_fit = time.time()
                log(f"graph model fitted on {len(m.days)} days ({m.days[0]} .. {m.days[-1]})")
            except Exception as e:  # noqa: BLE001 - too little data yet: say so, keep observing
                log(f"graph model not fitted yet: {e}")
        if not follow:
            return total
        time.sleep(poll_s)


# ------------------------------------------------------------------ the onboarding checklist
def _one(db: Path, sql: str, default=None):
    if not db.exists():
        return default
    try:
        with sqlite3.connect(db) as c:
            r = c.execute(sql).fetchone()
        return r[0] if r and r[0] is not None else default
    except sqlite3.Error:
        return default


def status(sim: dict, out_dir: Path, config: str) -> list[dict]:
    """[{step, done, found, action, run}]: `action` in words, `run` the netstudy command (None: nothing to run)."""
    from .inbox import inbox_dir, waiting
    from .realtown import osm_file

    db = out_dir / "traffic.db"
    cfg_arg = f"--sim-config {config}"
    steps = []

    def add(step, done, found, action, run=None):
        steps.append({"step": step, "done": bool(done), "found": found, "action": action, "run": run})

    f = osm_file(sim, out_dir)
    bbox = ((sim.get("network") or {}).get("osm") or {}).get("bbox")
    n_j = _one(db, "SELECT COUNT(*) FROM nodes WHERE kind = 'junction'", 0)
    n_e = _one(db, "SELECT COUNT(*) FROM edges", 0)
    has_map = (out_dir / "network" / "shapes.json").exists() and n_e
    add("Road map imported", has_map,
        f"{n_j} junctions, {n_e} road directions" if has_map else (f"{f.name} found, not imported yet" if f.exists()
                                                                    else f"no map file at {f}"),
        "Import the OpenStreetMap extract" if f.exists() or not bbox else "Download the roads in network.osm.bbox",
        f"netstudy onboard {cfg_arg}" + ("" if f.exists() else " --fetch"))
    cams_cfg = sim.get("cameras") or []
    cams = json.loads((out_dir / "cameras.json").read_text(encoding="utf-8")) if (out_dir / "cameras.json").exists() else []
    add("Cameras placed on the map", cams and len(cams) == len(cams_cfg),
        f"{len(cams)} of {len(cams_cfg)} in the config" if cams_cfg else "no `cameras` in the town config",
        "List each camera under `cameras` (id, at or edge), then place them" if not cams_cfg else "Place the cameras",
        f"netstudy cameras {cfg_arg}" if cams_cfg and has_map else None)
    cal = [c for c in cams if (c.get("calibration") or {}).get("H")]
    worst = max((c["calibration"]["rms_m"] for c in cal), default=None)
    add("Cameras calibrated", cams and len(cal) == len(cams),
        (f"{len(cal)} of {len(cams)}" + (f", worst fit {worst:.2f} m" if worst is not None else "")
         + "".join(f"; {c['camera_id']}: {c['calibration_error']}" for c in cams if c.get("calibration_error")))
        if cams else "no cameras yet",
        "Give each camera 4+ calibration points (pixel and map position), then place the cameras again; check "
        "the zone on a frame below", f"netstudy cameras {cfg_arg}" if cams else None)
    box = inbox_dir(sim, out_dir)
    wait = waiting(sim, out_dir)
    rej = len(list((box / "rejected").glob("*.reason.txt"))) if (box / "rejected").exists() else 0
    last = _one(db, "SELECT MAX(received_at) FROM received WHERE status = 'stored'")
    add("Recordings arriving", last or wait,
        f"inbox {box}: {len(wait)} waiting, {rej} rejected" + (f"; last stored {last}" if last else ""),
        "Point the recorders at the inbox (copy files in, or PUT them to `netstudy receive`)",
        f"netstudy receive {cfg_arg}")
    n_seg = sum(1 for _ in (out_dir / "recordings").glob("*/*/*")) if (out_dir / "recordings").exists() else 0
    n_ing = _one(db, "SELECT COUNT(*) FROM ingested", 0)
    add("Recordings read (SAM 3.1)", n_seg and n_ing >= n_seg,
        f"{n_ing} of {n_seg} segments", "Pick up the inbox and read the new segments (SAM 3.1, vision GPU)",
        f"netstudy observe {cfg_arg}")
    obs = _one(db, "SELECT COUNT(*) FROM observations", 0)
    days = _one(db, "SELECT COUNT(DISTINCT substr(ts, 1, 10)) FROM observations", 0)
    last_obs = _one(db, "SELECT MAX(ts) FROM observations")
    kpi_last = _one(db, "SELECT MAX(ts) FROM network_kpi")
    add("Observations and network state", obs and kpi_last and kpi_last >= (last_obs or ""),
        f"{obs} camera slots over {days} days" + (f", state up to {kpi_last}" if kpi_last else ", no state yet"),
        "Build the network state and KPIs from the observations", f"netstudy observe {cfg_arg}")
    from .graph_model import model_path
    mp = model_path(out_dir)
    fitted = datetime.fromtimestamp(mp.stat().st_mtime).isoformat(timespec="minutes") if mp.exists() else None
    add("Graph model for Forecast", mp.exists() and days >= MIN_DAYS,
        (f"fitted {fitted}" if fitted else "not fitted") + f"; {days} days observed ({MIN_DAYS}+ for a usual day)",
        "Fit the graph model once a week of recordings is in", f"netstudy graph-model {cfg_arg}" if obs else None)
    return steps
