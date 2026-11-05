"""Recordings pushed by outside systems (a camera's recorder, a VMS export, a cloud sync): the town's inbox.

  inbox      a folder (`source.inbox`, default data/<town>/inbox; a network share works) that other systems copy
             files into, or that `netstudy receive` writes the files PUT to it into
  pickup()   every closed file in the inbox: checked, then moved to recordings/<date>/<period>/<camera>_<HHMMSS>.<ext>,
             where ingest reads it; refused ones go to inbox/rejected/ with a .reason.txt  -> table `received`

File names (the camera's local clock; the camera id as in cameras.json):
  <camera>_<YYYYMMDD>T<HHMMSS>.mp4            cam03_20261105T081500.mp4
  <camera>/<YYYY-MM-DD>/<HHMMSS>.mp4          cam03/2026-11-05/081500.mp4
  a sidecar <same name>.json {"camera_id", "start": "2026-11-05T08:15:00"} overrides the name
Video: .mp4 (H.264), .mkv, .avi or .mov; any length (1-15 minutes is usual); the frame rate is read from the file.
A file is closed when it has not changed for `source.inbox_quiet_s` seconds (default 10), or, with
`source.inbox_marker: true`, when <name>.done exists beside it (for senders that write slowly).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

VIDEO = (".mp4", ".mkv", ".avi", ".mov")
FLAT = re.compile(r"^(?P<cam>[A-Za-z0-9-]+)_(?P<d>\d{8})T(?P<t>\d{6})$")
SCHEMA = """CREATE TABLE IF NOT EXISTS received (file TEXT, camera_id TEXT, start TEXT, seconds REAL, bytes INTEGER,
    status TEXT, reason TEXT, stored TEXT, received_at TEXT)"""


def inbox_dir(sim: dict, out_dir: Path) -> Path:
    f = (sim.get("source") or {}).get("inbox")
    if not f:
        return out_dir / "inbox"
    p = Path(f)
    return p if p.is_absolute() else Path(sim.get("_config_dir") or ".") / p


def parse(path: Path, inbox: Path) -> tuple[str, datetime]:
    """(camera id, start) from the sidecar or the file name, else ValueError."""
    side = path.with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text(encoding="utf-8"))
        return str(meta["camera_id"]), datetime.fromisoformat(str(meta["start"]))
    m = FLAT.match(path.stem)
    if m:
        return m["cam"], datetime.strptime(m["d"] + m["t"], "%Y%m%d%H%M%S")
    rel = path.relative_to(inbox).parts
    if len(rel) == 3 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", rel[1]) and re.fullmatch(r"\d{6}", path.stem):
        return rel[0], datetime.strptime(f"{rel[1]} {path.stem}", "%Y-%m-%d %H%M%S")
    raise ValueError("name not understood: use <camera>_<YYYYMMDD>T<HHMMSS>.mp4, <camera>/<YYYY-MM-DD>/<HHMMSS>.mp4 "
                     "or a .json sidecar")


def waiting(sim: dict, out_dir: Path) -> list[Path]:
    box = inbox_dir(sim, out_dir)
    if not box.exists():
        return []
    return sorted(p for p in box.rglob("*") if p.suffix.lower() in VIDEO and "rejected" not in p.relative_to(box).parts
                  and ".partial" not in p.name)


def closed(path: Path, sim: dict) -> bool:
    src = sim.get("source") or {}
    if src.get("inbox_marker"):
        return Path(str(path) + ".done").exists()
    return time.time() - path.stat().st_mtime > float(src.get("inbox_quiet_s", 10))


def video_seconds(path: Path) -> float:
    """The video's length; ValueError when it cannot be read."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("not a video OpenCV can read (send H.264 MP4)")
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        ok, _ = cap.read()
        if not ok or fps <= 0 or n <= 0:
            raise ValueError("no frames, or no frame rate")
        return n / fps
    finally:
        cap.release()


def period_of(sim: dict, start: datetime) -> str:
    """The recording window the segment starts in (recording.windows), else 'day'."""
    hhmm = f"{start:%H:%M}"
    for w in (sim.get("recording") or {}).get("windows") or []:
        if w["start"] <= hhmm < w["end"]:
            return str(w["period"])
    return "day"


def pickup(sim: dict, out_dir: Path, cameras: list[dict], log=print) -> dict:
    """Move every closed inbox file to the recordings (see the module doc). Returns counts."""
    box = inbox_dir(sim, out_dir)
    known = {c["camera_id"] for c in cameras}
    out = {"stored": 0, "rejected": 0, "open": 0}
    db = out_dir / "traffic.db"
    with sqlite3.connect(db) as c:
        c.execute(SCHEMA)
    for f in waiting(sim, out_dir):
        if not closed(f, sim):
            out["open"] += 1
            continue
        now = datetime.now().isoformat(timespec="seconds")
        size = f.stat().st_size
        cid, start, secs, reason, dest = None, None, None, None, None
        try:
            cid, start = parse(f, box)
            if cid not in known:
                raise ValueError(f"camera {cid!r} is not in cameras.json (towns config `cameras`)")
            secs = video_seconds(f)
            dest = out_dir / "recordings" / f"{start:%Y-%m-%d}" / period_of(sim, start) / \
                f"{cid}_{start:%H%M%S}{f.suffix.lower()}"
            if dest.exists():
                raise ValueError(f"a recording for {cid} at {start:%Y-%m-%d %H:%M:%S} is already stored")
        except (ValueError, KeyError, OSError) as e:
            reason = str(e)
        with sqlite3.connect(db) as c:
            if reason is None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), dest)
                _drop_extras(f)
                out["stored"] += 1
                c.execute("INSERT INTO received VALUES (?, ?, ?, ?, ?, 'stored', NULL, ?, ?)",
                          (f.name, cid, start.isoformat(), round(secs, 1), size, str(dest), now))
                log(f"stored {f.name} -> {dest.relative_to(out_dir)}")
            else:
                rej = box / "rejected"
                rej.mkdir(parents=True, exist_ok=True)
                target = rej / f.name
                if target.exists():
                    target = rej / f"{f.stem}_{int(time.time())}{f.suffix}"
                shutil.move(str(f), target)
                _drop_extras(f)
                Path(str(target) + ".reason.txt").write_text(reason + "\n", encoding="utf-8")
                out["rejected"] += 1
                c.execute("INSERT INTO received VALUES (?, ?, ?, ?, ?, 'rejected', ?, ?, ?)",
                          (f.name, cid, start.isoformat() if start else None, secs, size, reason, str(target), now))
                log(f"rejected {f.name}: {reason}")
    return out


def _drop_extras(f: Path):
    for extra in (f.with_suffix(".json"), Path(str(f) + ".done")):
        if extra.exists():
            extra.unlink()


def receive(sim: dict, out_dir: Path, host: str = "0.0.0.0", port: int = 8765, token: str | None = None, log=print):
    """A small HTTP receiver for systems that push rather than copy: PUT /recordings/<camera>/<YYYYMMDDTHHMMSS>.mp4
    with the video as the body and `Authorization: Bearer <token>` (env FLOWCAST_INGEST_TOKEN). The file lands in the
    inbox under its flat name, written to a .partial first, so pickup never sees half a file. GET /health answers ok."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    token = token or os.environ.get("FLOWCAST_INGEST_TOKEN")
    if not token:
        raise SystemExit("set FLOWCAST_INGEST_TOKEN (the senders' shared secret) before starting the receiver")
    box = inbox_dir(sim, out_dir)
    box.mkdir(parents=True, exist_ok=True)
    path_re = re.compile(r"^/recordings/(?P<cam>[A-Za-z0-9-]+)/(?P<ts>\d{8}T\d{6})(?P<ext>\.(mp4|mkv|avi|mov))$")

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, text: str):
            body = (text + "\n").encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._reply(200, "ok") if self.path == "/health" else self._reply(404, "not found")

        def do_PUT(self):  # noqa: N802
            if self.headers.get("Authorization") != f"Bearer {token}":
                return self._reply(401, "bad or missing token")
            m = path_re.match(self.path)
            if not m:
                return self._reply(400, "use PUT /recordings/<camera>/<YYYYMMDDTHHMMSS>.mp4")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return self._reply(411, "Content-Length needed")
            final = box / f"{m['cam']}_{m['ts']}{m['ext']}"
            part = final.with_name(final.name + ".partial")
            with open(part, "wb") as out:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    out.write(chunk)
                    left -= len(chunk)
            if left:
                part.unlink(missing_ok=True)
                return self._reply(400, "body shorter than Content-Length")
            os.replace(part, final)
            log(f"received {final.name} ({n} bytes) from {self.client_address[0]}")
            self._reply(201, final.name)

        def log_message(self, *a):
            pass

    log(f"receiving recordings on http://{host}:{port}/recordings/<camera>/<YYYYMMDDTHHMMSS>.mp4 -> {box}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
