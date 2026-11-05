"""Unattended: the pending towns end to end, one GPU under load at a time (docs/architecture-next.md 2.3).

  .venv\\Scripts\\python scripts\\run_towns.py [--towns brookhaven clearwater]       # log: data\\towns_run.log

Per town: wait until no Glimmer session (check_assist.py) and no SUMO-only job of that town is running, make sure
llama-server is stopped, then `netstudy run` (SUMO days on the CPU, the rendered day on the vision GPU: CARLA + SAM 3.1 on
source.render_cameras), `netstudy graph-model`, `netstudy whatif-labels`, check_predict.py. At the end one Glimmer
session on the Traffic Assistant GPU (nothing else under load then) for each town's questions, then the status check. A failed step is logged
and the next one runs (a town whose run failed skips its later steps).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from netstudy import workspace

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
NETSTUDY = str(ROOT / ".venv" / "Scripts" / "netstudy.exe")
LOG = ROOT / "data" / "towns_run.log"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
VISION_ONLY = {**ENV, "CUDA_VISIBLE_DEVICES": str(workspace.vision_gpu())}  # netstudy run: PyTorch (SAM 3.1) sees
                                                   # only the vision GPU; CARLA picks its card by name (dxgi.py)


def log(msg: str):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def running(pattern: str) -> list[str]:
    """Command lines of running processes that contain `pattern` (not this script)."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | "
                          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    me = str(os.getpid())
    return [ln for ln in out.splitlines() if pattern in ln and not ln.startswith(me + "\t")
            and "run_towns.py" not in ln and "Get-CimInstance" not in ln]


def wait_for(pattern: str, what: str, poll_s: int = 60):
    first = True
    while running(pattern):
        if first:
            log(f"waiting for {what} to finish")
            first = False
        time.sleep(poll_s)


def step(name: str, args: list[str], log_file: Path, env: dict | None = None) -> bool:
    log(f"{name}: {' '.join(args)}  (log {log_file.relative_to(ROOT)})")
    t0 = time.time()
    with open(log_file, "a", encoding="utf-8") as f:
        r = subprocess.run(args, cwd=ROOT, env=env or ENV, stdout=f, stderr=subprocess.STDOUT)
    log(f"{name}: {'done' if r.returncode == 0 else f'FAILED (exit {r.returncode})'} in {(time.time() - t0) / 60:.0f} min")
    return r.returncode == 0


def tail(path: Path, n: int = 6) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]) if path.exists() else ""


def glimmer(cmd: str) -> str:
    r = subprocess.run([PY, "-c", f"from netstudy import glimmer_server as gs; print(gs.{cmd}())"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout.strip() or r.stderr.strip().splitlines()[-1] if (r.stdout or r.stderr) else "").strip()


def town(name: str) -> bool:
    cfg = f"towns/{name}.yaml"
    d = ROOT / "data" / name
    wait_for("check_assist.py", "the Glimmer session")
    if glimmer("status") != "off":
        log(f"Glimmer server is {glimmer('status')}: stopping it before the vision-GPU work ({glimmer('stop')})")
    wait_for(f"towns/{name}.yaml", f"another {name} job")
    if not step(f"{name} run", [NETSTUDY, "run", "--sim-config", cfg, "--first", "2026-10-05", "--days", "30"],
                d / "run.log", VISION_ONLY):
        log(tail(d / "run.log", 15))
        return False
    log(tail(d / "run.log", 8))
    step(f"{name} graph-model", [NETSTUDY, "graph-model", "--sim-config", cfg], d / "graph_model.log")
    step(f"{name} whatif-labels", [NETSTUDY, "whatif-labels", "--sim-config", cfg, "--workers", "4"],
         d / "whatif_labels.log")
    if step(f"{name} check_predict", [PY, "scripts/check_predict.py", "--sim-config", cfg], d / "check_predict.log"):
        log(tail(d / "check_predict.log", 4))
    return True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--towns", nargs="+", default=["brookhaven", "clearwater"])
    p.add_argument("--no-glimmer", action="store_true")
    args = p.parse_args(argv)
    log(f"run_towns: {', '.join(args.towns)}")
    ok = [t for t in args.towns if town(t)]
    if ok and not args.no_glimmer:
        wait_for("netstudy", "vision-GPU / CPU jobs")
        log(f"Glimmer session for {', '.join(ok)}: {glimmer('start')}")
        for _ in range(100):
            if glimmer("status") == "ready":
                break
            time.sleep(3)
        for t in ok:
            f = ROOT / "data" / t / "check_assist.log"
            if step(f"{t} check_assist", [PY, "scripts/check_assist.py", "--sim-config", f"towns/{t}.yaml"], f):
                log(tail(f, 1))
        log(f"Glimmer: {glimmer('stop')}")
    step("check_e2e", [PY, "scripts/check_e2e.py"], ROOT / "data" / "e2e_run.log")
    log(tail(ROOT / "data" / "e2e_run.log", 1))
    log("run_towns: finished")


if __name__ == "__main__":
    sys.exit(main())
