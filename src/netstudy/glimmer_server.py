"""Start and stop Glimmer's llama-server for a Forecast session.

  start()   llama-server on the workspace's Traffic Assistant GPU (workspace.glimmer_gpu; CUDA_DEVICE_ORDER=PCI_BUS_ID,
            --device CUDA<n>); with workspace.one_at_a_time, refused while another GPU is under load (SAM 3.1,
            CARLA); returns at once, the model loads in the background
  stop()    ends the server this module started (its pid is kept in data/glimmer_server.json)
  status()  "off" | "loading" | "ready" | "running elsewhere" (a server on the port that this module did not start)

The command is docs/setup.md's. Log: data/glimmer_server.log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import workspace

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
MODEL = ROOT / "models" / "Muse-Glimmer-30B-Q3_K_M.gguf"
PORT = 8080
STATE = ROOT / "data" / "glimmer_server.json"
LOG = ROOT / "data" / "glimmer_server.log"
BUSY = {"mem_mib": 2000, "util_pct": 40, "watts": 80}
# a GPU job holds memory (SAM 3.1, CARLA: GBs) or draws power; the desktop alone spikes utilization to ~80 % at
# 5-30 W and ~300 MiB, so utilization counts only together with power, as the median of a few samples


def command() -> list[str]:
    return [str(EXE), "-m", str(MODEL), "--jinja", "-ngl", "99", "-c", "8192", "--port", str(PORT),
            "--device", f"CUDA{workspace.glimmer_gpu()}"]


def gpu_load() -> dict[int, dict]:
    """nvidia-smi per card, in PCI order (nvidia-smi's own): util %, memory MiB, name."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    cards = {}
    for line in out.strip().splitlines():
        i, name, util, mem, watts = (x.strip() for x in line.split(","))
        cards[int(i)] = {"name": name, "util_pct": float(util), "mem_mib": float(mem),
                         "watts": float(watts) if watts.replace(".", "", 1).isdigit() else 0.0}
    return cards


def others_busy(samples: int = 5, every_s: float = 0.5) -> str | None:
    """Why another GPU than the Traffic Assistant's counts as under load, or None: memory held, or utilization with
    power (medians of a few samples). Only checked with workspace.one_at_a_time."""
    import statistics
    import time

    seen = []
    for k in range(samples):
        own = workspace.glimmer_gpu()
        g = next((x for i, x in sorted(gpu_load().items()) if i != own), None)
        if g is None:
            return None
        seen.append(g)
        if k + 1 < samples:
            time.sleep(every_s)
    med = {c: statistics.median(x[c] for x in seen) for c in ("util_pct", "mem_mib", "watts")}
    if med["mem_mib"] > BUSY["mem_mib"] or (med["util_pct"] > BUSY["util_pct"] and med["watts"] > BUSY["watts"]):
        return (f"{seen[0]['name']} is under load: {med['util_pct']:.0f} %, {med['mem_mib']:.0f} MiB, "
                f"{med['watts']:.0f} W")
    return None


def _pid() -> int | None:
    try:
        return int(json.loads(STATE.read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError):
        return None


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return "llama-server" in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _healthy(timeout: float = 2) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def status() -> str:
    pid = _pid()
    mine = pid is not None and _alive(pid)
    if _healthy():
        return "ready" if mine else "running elsewhere"
    return "loading" if mine else "off"


def start() -> str:
    """Start the server; returns what happened. With workspace.one_at_a_time, refused while another GPU is busy."""
    s = status()
    if s != "off":
        return f"already {s}"
    if not EXE.exists() or not MODEL.exists():
        raise SystemExit(f"llama-server or the model is missing ({EXE}, {MODEL}); see docs/setup.md")
    busy = others_busy() if workspace.one_at_a_time() else None
    if busy:
        raise SystemExit(f"Glimmer not started: {busy} (workspace: one GPU under load at a time)")
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}          # CUDA numbers as nvidia-smi's
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(command(), cwd=ROOT, env=env, stdout=open(LOG, "w", encoding="utf-8"),
                            stderr=subprocess.STDOUT, creationflags=flags)
    STATE.write_text(json.dumps({"pid": proc.pid, "cmd": command()}), encoding="utf-8")
    return "starting"


def stop() -> str:
    pid = _pid()
    if pid is None or not _alive(pid):
        STATE.unlink(missing_ok=True)
        return "not running (or not started here)"
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, 15)
    STATE.unlink(missing_ok=True)
    return "stopped"
