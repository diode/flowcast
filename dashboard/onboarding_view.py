"""Technical view, Onboarding & ingest: bring a real town in and keep its observations coming (no SUMO).

  checklist        observe.status: map, cameras, calibration, recordings arriving, SAM 3.1 reading, state, model;
                   each step's next action as a button (a background job, netstudy.jobs) and as its command
  jobs             what runs now, its log, Stop
  live intake      the HTTP receiver and `observe --follow` as long-running jobs
  inbox            files waiting, refused ones with the reason, the last ones stored
  cameras          per camera: its road, calibration fit, segments stored / read, last observation
  calibration      a camera's latest frame with its measured zone and its calibration points drawn on it
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from netstudy import jobs  # noqa: E402
from netstudy.cli import load_sim_config  # noqa: E402
from netstudy.places import TOWNS  # noqa: E402

JOB_OF = {"onboard": "onboard", "cameras": "cameras", "observe": "observe", "graph-model": "graph-model",
          "receive": "receive"}
ICON = {True: ":material/check_circle:", False: ":material/radio_button_unchecked:"}


def real_towns() -> list[str]:
    import yaml

    out = []
    for f in sorted(TOWNS.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if (cfg.get("source") or {}).get("kind") == "real":
            out.append(f.stem)
    return out


def _run(out_dir: Path, name: str, argv: list[str]):
    msg = jobs.start(out_dir, name, argv)
    if msg:
        st.toast(msg)
    st.rerun()


def panel(current: str | None = None) -> None:
    towns = real_towns()
    st.markdown("#### Onboarding & ingest")
    if not towns:
        st.info("No real town yet. Copy `towns/templates/real_town.yaml` to `towns/<your town>.yaml`, set its map "
                "(`network.osm.file` or `bbox`) and its cameras, then come back here. Simulated towns "
                "(Ashford, Brookhaven, Clearwater) are built with `netstudy run` instead (README, Simulation).")
        return
    from netstudy.places import city_name

    town = st.selectbox("Real town", towns, index=towns.index(current) if current in towns else 0, key="onb_town",
                        format_func=lambda t: f"{city_name(t)} ({t})")
    cfg_path = TOWNS / f"{town}.yaml"
    sim, out_dir = load_sim_config(cfg_path)
    cfg = str(cfg_path)
    from netstudy.observe import status

    steps = status(sim, out_dir, f"towns/{town}.yaml")
    a, b = st.columns([1.6, 1], gap="medium")
    with a.container(border=True):
        done = sum(s["done"] for s in steps)
        st.markdown(f"**Checklist** · {done} of {len(steps)} done")
        for i, s in enumerate(steps):
            c = st.columns([0.06, 0.7, 0.24], vertical_alignment="center")
            c[0].markdown(ICON[s["done"]])
            c[1].markdown(f"**{s['step']}**  \n{s['found']}" + ("" if s["done"] else f"  \n*Next: {s['action']}*"))
            if s["run"]:
                cmd = s["run"].split()[1]
                job = JOB_OF.get(cmd, cmd)
                if cmd == "receive":
                    continue                                   # long-running: below, with Start / Stop
                argv = [cmd, "--sim-config", cfg] + s["run"].split()[4:]
                if c[2].button("Run again" if s["done"] else "Run", key=f"onb_run_{i}", width="stretch",
                               disabled=jobs.running(out_dir, job), help=s["run"]):
                    _run(out_dir, job, argv)
        st.caption("Each button runs the command shown when you hover it, in the background; the same commands work "
                   "in a terminal (`.venv\\Scripts\\netstudy …`).")
    with b.container(border=True):
        st.markdown("**Live intake**")
        for job, label, argv, hint in (
                ("receive", "HTTP receiver", ["receive", "--sim-config", cfg],
                 "PUT /recordings/&lt;camera&gt;/&lt;YYYYMMDDTHHMMSS&gt;.mp4 on port 8765; needs FLOWCAST_INGEST_TOKEN"),
                ("observe-follow", "Read as they arrive", ["observe", "--sim-config", cfg, "--follow"],
                 "pick up the inbox and read new segments every 30 s (SAM 3.1 on the vision GPU)")):
            on = jobs.running(out_dir, job)
            c = st.columns([0.62, 0.38], vertical_alignment="center")
            c[0].markdown(f"{'🟢' if on else '⚪'} **{label}**  \n<span style='font-size:12px;color:#6b7280'>{hint}"
                          "</span>", unsafe_allow_html=True)
            if c[1].button("Stop" if on else "Start", key=f"onb_live_{job}", width="stretch"):
                jobs.stop(out_dir, job) if on else jobs.start(out_dir, job, argv)
                st.rerun()
        from netstudy.inbox import inbox_dir
        st.caption(f"Inbox: `{inbox_dir(sim, out_dir)}` — recorders copy files here, named "
                   "`<camera>_<YYYYMMDD>T<HHMMSS>.mp4` (README, Recordings).")
    _jobs(out_dir)
    t1, t2, t3 = st.tabs(["Cameras", "Inbox", "Calibration check"])
    with t1:
        _cameras(out_dir)
    with t2:
        _inbox(sim, out_dir)
    with t3:
        _calibration(sim, out_dir)


@st.fragment(run_every=5)
def _jobs(out_dir: Path) -> None:
    names = sorted(p.stem for p in (out_dir / "jobs").glob("*.json")) if (out_dir / "jobs").exists() else []
    shown = [(n, jobs.state(out_dir, n)) for n in names]
    shown = [(n, s) for n, s in shown if s]
    if not shown:
        return
    with st.container(border=True):
        st.markdown("**Jobs**")
        for n, s in sorted(shown, key=lambda x: x[1].get("started", ""), reverse=True)[:4]:
            word = {"running": "🟡 running", "done": "🟢 done", "failed": "🔴 failed", "stopped": "⚪ stopped"}[s["state"]]
            with st.expander(f"{n} · {word} · started {s.get('started', '')[11:16]}"
                             + (f", ended {s['ended'][11:16]}" if s.get("ended") else ""), expanded=s["state"] == "running"):
                st.code(jobs.tail(out_dir, n, 25) or "(no output yet)", language=None)
                if s["state"] == "running" and st.button("Stop", key=f"onb_stop_{n}"):
                    jobs.stop(out_dir, n)
                    st.rerun(scope="app")


def _q(db: Path, sql: str) -> pd.DataFrame:
    if not db.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(db) as c:
            return pd.read_sql(sql, c)
    except Exception:  # noqa: BLE001 - a table not there yet
        return pd.DataFrame()


def _cams(out_dir: Path) -> list[dict]:
    f = out_dir / "cameras.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _cameras(out_dir: Path) -> None:
    cams = _cams(out_dir)
    if not cams:
        st.caption("No cameras placed yet (checklist: Cameras placed on the map).")
        return
    db = out_dir / "traffic.db"
    ing = _q(db, "SELECT camera_id, COUNT(*) AS read, MAX(start) AS last_segment FROM ingested GROUP BY camera_id")
    obs = _q(db, "SELECT camera_id, COUNT(*) AS slots, MAX(ts) AS last_slot FROM observations GROUP BY camera_id")
    rec = pd.DataFrame([{"camera_id": p.stem.rsplit("_", 1)[0]} for p in (out_dir / "recordings").glob("*/*/*")]) \
        if (out_dir / "recordings").exists() else pd.DataFrame(columns=["camera_id"])
    stored = rec.groupby("camera_id").size().rename("stored").reset_index() if len(rec) else pd.DataFrame(columns=["camera_id", "stored"])
    t = pd.DataFrame([{"camera_id": c["camera_id"], "name": c["name"], "road": f"{c['edge']} → {c['junction']}",
                       "lanes": len(c["lanes"]),
                       "calibration": f"fit {c['calibration']['rms_m']:.2f} m" if c.get("calibration")
                       else ("error: " + c["calibration_error"] if c.get("calibration_error") else "missing")}
                      for c in cams])
    for extra in (stored, ing, obs):
        if len(extra):
            t = t.merge(extra, on="camera_id", how="left")
    st.dataframe(t, hide_index=True, width="stretch")
    if (out_dir / "map.png").exists():
        with st.expander("Map with junction ids and cameras (map.png)"):
            st.image(str(out_dir / "map.png"))


def _inbox(sim: dict, out_dir: Path) -> None:
    from netstudy.inbox import inbox_dir, waiting

    box = inbox_dir(sim, out_dir)
    w = waiting(sim, out_dir)
    st.markdown(f"`{box}` · {len(w)} waiting")
    if w:
        st.dataframe(pd.DataFrame([{"file": str(p.relative_to(box)), "MB": round(p.stat().st_size / 1e6, 1),
                                    "changed": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")}
                                   for p in w[:200]]), hide_index=True, width="stretch")
    rej = sorted((box / "rejected").glob("*.reason.txt")) if (box / "rejected").exists() else []
    if rej:
        st.markdown(f"**Refused ({len(rej)})** — fix and copy back into the inbox")
        st.dataframe(pd.DataFrame([{"file": p.name.removesuffix(".reason.txt"),
                                    "reason": p.read_text(encoding="utf-8").strip()} for p in rej[-100:]]),
                     hide_index=True, width="stretch")
    got = _q(out_dir / "traffic.db", "SELECT received_at, file, camera_id, start, seconds, status, reason FROM received "
                                     "ORDER BY received_at DESC LIMIT 50")
    if len(got):
        st.markdown("**Last picked up**")
        st.dataframe(got, hide_index=True, width="stretch")
    elif not w and not rej:
        st.caption("Nothing received yet.")


def _latest_video(out_dir: Path, inbox: Path, cid: str) -> Path | None:
    from netstudy.ingest import VIDEO

    found = [p for p in (out_dir / "recordings").glob(f"*/*/{cid}_*") if p.suffix.lower() in VIDEO] \
        if (out_dir / "recordings").exists() else []
    found += [p for p in inbox.rglob(f"{cid}*") if p.suffix.lower() in VIDEO] if inbox.exists() else []
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _calibration(sim: dict, out_dir: Path) -> None:
    import cv2
    import numpy as np

    from netstudy import osm
    from netstudy.inbox import inbox_dir
    from netstudy.realtown import HomographyCalib, ShapeZone

    cams = [c for c in _cams(out_dir) if c.get("calibration")]
    if not cams:
        st.caption("No calibrated camera yet.")
        return
    cid = st.selectbox("Camera", [c["camera_id"] for c in cams], key="onb_cal_cam",
                       format_func=lambda i: f"{i} · {next(c['name'] for c in cams if c['camera_id'] == i)}")
    cam = next(c for c in cams if c["camera_id"] == cid)
    video = _latest_video(out_dir, inbox_dir(sim, out_dir), cid)
    if video is None:
        st.caption(f"No recording of {cid} yet: the check draws on its latest frame.")
        return
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2))
    ok, img = cap.read()
    cap.release()
    if not ok:
        st.warning(f"Could not read a frame of {video.name}.")
        return
    calib = HomographyCalib()
    zone = ShapeZone(cam, osm.load_shapes(out_dir)[cam["edge"]])
    poly = [calib.to_pixel(cam, x, y) for x, y in zone.outline()]
    poly = np.array([p for p in poly if p], dtype=np.int32)
    over = img.copy()
    if len(poly) >= 3:
        cv2.fillPoly(over, [poly], (110, 118, 15))
        img = cv2.addWeighted(over, 0.28, img, 0.72, 0)
        cv2.polylines(img, [poly], True, (110, 118, 15), 2)
    ctr = osm.centre(out_dir)
    for p in cam["calibration"]["points"]:
        u, v = map(int, p["px"])
        g = (p["x"], p["y"]) if "x" in p else osm.to_xy(ctr, p["lat"], p["lon"])
        back = calib.to_pixel(cam, *g)
        cv2.circle(img, (u, v), 7, (0, 200, 255), 2)                     # where the installer clicked
        if back:
            cv2.drawMarker(img, (int(back[0]), int(back[1])), (40, 40, 220), cv2.MARKER_CROSS, 14, 2)   # the map point
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch",
             caption=f"{video.name}: the measured zone (teal, {zone.view_m:.0f} m up from the stop line, "
                     f"{len(cam['lanes'])} lanes); calibration points as clicked (circles) and where their map position "
                     f"falls (crosses). Fit {cam['calibration']['rms_m']:.2f} m. The zone should cover exactly the "
                     "approach's lanes; if it does not, correct the points in the town config and place the cameras again.")
