"""netstudy commands run in the background for the technical view (onboard, cameras, observe, receive, graph-model):
one process per town and job name, its output in data/<town>/jobs/<name>.log, its state in <name>.json.

  start(out_dir, name, argv)   python -m netstudy.jobs <state file> <argv>; refused while that job runs
  state(out_dir, name)         {"state": running | done | failed | stopped, "code", "started", "ended", "argv"} or None
  stop(out_dir, name)          ends the job and its children
  tail(out_dir, name, n)       the log's last lines
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _files(out_dir: Path, name: str) -> tuple[Path, Path]:
    d = out_dir / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json", d / f"{name}.log"


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def state(out_dir: Path, name: str) -> dict | None:
    f, _ = _files(out_dir, name)
    if not f.exists():
        return None
    s = json.loads(f.read_text(encoding="utf-8"))
    if s.get("state") == "running" and not _alive(int(s["pid"])):
        s["state"] = "stopped"
    return s


def running(out_dir: Path, name: str) -> bool:
    s = state(out_dir, name)
    return bool(s and s["state"] == "running")


def start(out_dir: Path, name: str, argv: list[str]) -> str | None:
    """Start the job; a message instead when it already runs."""
    if running(out_dir, name):
        return f"{name} is already running"
    f, log = _files(out_dir, name)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    with open(log, "w", encoding="utf-8") as out:
        p = subprocess.Popen([sys.executable, "-u", "-m", "netstudy.jobs", str(f), *argv], stdout=out,
                             stderr=subprocess.STDOUT, creationflags=flags, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    f.write_text(json.dumps({"state": "running", "pid": p.pid, "argv": argv,
                             "started": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    return None


def stop(out_dir: Path, name: str) -> None:
    s = state(out_dir, name)
    if s and s["state"] == "running":
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(s["pid"]), "/T", "/F"], capture_output=True)
        else:
            os.kill(int(s["pid"]), 15)
        f, _ = _files(out_dir, name)
        f.write_text(json.dumps({**s, "state": "stopped", "ended": datetime.now().isoformat(timespec="seconds")}),
                     encoding="utf-8")


def tail(out_dir: Path, name: str, n: int = 30) -> str:
    _, log = _files(out_dir, name)
    if not log.exists():
        return ""
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def _main(argv: list[str]) -> int:
    """The child: run `netstudy <argv>` and write how it ended into the state file."""
    from .cli import main

    f, args = Path(argv[0]), argv[1:]
    code = 0
    try:
        main(args)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if not isinstance(e.code, int) and e.code is not None:
            print(e.code, flush=True)
    except Exception as e:  # noqa: BLE001 - reported in the log and the state file
        import traceback
        traceback.print_exc()
        print(f"failed: {e}", flush=True)
        code = 1
    s = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}        # as the parent wrote it at start
    f.write_text(json.dumps({**s, "state": "done" if code == 0 else "failed", "code": code,
                             "ended": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
