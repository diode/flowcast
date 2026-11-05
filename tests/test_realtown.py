"""A real town end to end, without SUMO, CARLA or a GPU: OpenStreetMap extract -> graph, a camera placed and
calibrated from point pairs -> recordings pushed into the inbox (and over HTTP) -> picked up -> read (a stand-in
detector for SAM 3.1) -> observations -> edge state and KPIs -> what the dashboard reads; the checklist on the way."""
import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import yaml

cv2 = pytest.importorskip("cv2")

FIXTURE = Path(__file__).parent / "fixtures" / "millbrook.osm"
FPS, SECONDS, W, H = 5, 60, 640, 360
SPEED_MS = 10.0                                   # the stand-in vehicles: 36 km/h towards the stop line


def write_config(d: Path, cameras=None) -> Path:
    cfg = {"name": "Millbrook", "out_dir": "data/millbrook", "sample_minutes": 5,
           "source": {"kind": "real", "inbox_quiet_s": 0},
           "network": {"source": "osm", "osm": {"file": str(FIXTURE)}, "view_m": 60, "lefthand": False},
           "ingest": {"min_samples": 3}, "cameras": cameras or []}
    f = d / "towns" / "millbrook.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return f


def ground(cam, shape, lane, m):
    """The road point `m` metres before the camera's stop line on `lane` (as ShapeZone measures it)."""
    from netstudy.realtown import LANE_W, _point_at

    p, u = _point_at(np.asarray(shape, dtype=float), cam["stop_s"] - m)
    off = cam["lateral_m"][0] + (lane + 0.5) * LANE_W
    return p + off * np.array([u[1], -u[0]])


class StandIn:
    """SAM 3.1's stand-in: reads the frame number painted into the frame and puts three cars where they are then."""

    def __init__(self, cam, shape):
        from netstudy.realtown import HomographyCalib

        self.cam, self.shape, self.calib = cam, shape, HomographyCalib()

    def detect(self, bgr):
        from netstudy.vision import Detection

        k = int(round(bgr[:, : W // 2].mean() / 20)) * 50 + int(round(bgr[:, W // 2:].mean() / 5))
        t = k / FPS
        out = []
        for i in range(3):
            m = (50 - SPEED_MS * t + 18 * i) % 54 + 2
            u, v = self.calib.to_pixel(self.cam, *ground(self.cam, self.shape, 0, m))
            out.append(Detection(u - 12, v - 16, u + 12, v, 0.9, "car"))
        return out


def video(path: Path):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for k in range(FPS * SECONDS):
        img = np.zeros((H, W, 3), np.uint8)
        img[:, : W // 2] = (k // 50) * 20
        img[:, W // 2:] = (k % 50) * 5
        wr.write(img)
    wr.release()


def test_real_town_end_to_end(tmp_path, monkeypatch):
    from netstudy import inbox, osm
    from netstudy.cli import load_sim_config
    from netstudy.observe import observe, status
    from netstudy.realtown import onboard, place_cameras

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    sim, out = load_sim_config(write_config(tmp_path))
    town = onboard(sim, out, log=lambda m: None)

    # the map: four junctions, five map edges, the residential street left out, names from the streets
    n = town["nodes"]
    assert (n.kind == "junction").sum() == 4 and (n.kind == "boundary").sum() == 5
    assert len(town["edges"]) == 17                                   # 8 two-way roads + one-way Park Road
    names = set(town["names"].values())
    assert {"High Street & Mill Lane", "High Street & Station Road", "Mill Lane & Park Road",
            "Park Road & Station Road"} <= names
    centre_j = next(k for k, v in town["names"].items() if v == "High Street & Mill Lane")
    assert int(n.set_index("node_id").at[centre_j, "signal"]) == 1     # the signals node 7 m away
    e = town["edges"]
    assert set(e[e.street == "Mill Lane"].speed_kmh) == {48.3}         # 30 mph
    assert not (e.street == "Orchard Close").any()

    # a camera on High Street eastbound into the centre, calibrated from four point pairs
    sim["cameras"] = [{"id": "cam01", "name": "High Street eastbound at Mill Lane",
                       "at": {"lat": 52.0, "lon": -0.0003}, "heading": 90}]
    cam = place_cameras(sim, out)[0]
    assert cam["junction"] == centre_j and cam["edge"].endswith(f"_{centre_j}") and cam["calibration"] is None
    shape = osm.load_shapes(out)[cam["edge"]]
    quad_px = [(120, 340), (520, 340), (390, 70), (250, 70)]
    quad_g = [ground(cam, shape, -0.5, 0), ground(cam, shape, 0.5, 0), ground(cam, shape, 0.5, 60),
              ground(cam, shape, -0.5, 60)]
    sim["cameras"][0]["calibration"] = {"image_size": [W, H], "points": [
        {"px": list(p), "x": float(g[0]), "y": float(g[1])} for p, g in zip(quad_px, quad_g)]}
    cam = place_cameras(sim, out)[0]
    assert cam["calibration"]["rms_m"] < 0.01

    steps = {s["step"]: s for s in status(sim, out, "towns/millbrook.yaml")}
    assert steps["Road map imported"]["done"] and steps["Cameras calibrated"]["done"]
    assert not steps["Recordings arriving"]["done"]

    # recordings pushed in: two copied (both name forms), one over HTTP, two refused
    box = inbox.inbox_dir(sim, out)
    box.mkdir(parents=True)
    video(box / "cam01_20261105T080000.mp4")
    (box / "cam01" / "2026-11-05").mkdir(parents=True)
    video(box / "cam01" / "2026-11-05" / "080100.mp4")
    video(box / "cam99_20261105T080000.mp4")                          # no such camera
    (box / "notes.mp4").write_bytes(b"not a video")
    monkeypatch.setenv("FLOWCAST_INGEST_TOKEN", "secret")
    th = threading.Thread(target=inbox.receive, args=(sim, out), kwargs={"host": "127.0.0.1", "port": 8799,
                                                                        "log": lambda m: None}, daemon=True)
    th.start()
    time.sleep(0.5)
    video(tmp_path / "push.mp4")
    body = (tmp_path / "push.mp4").read_bytes()
    req = urllib.request.Request("http://127.0.0.1:8799/recordings/cam01/20261105T080200.mp4", body, method="PUT",
                                 headers={"Authorization": "Bearer secret"})
    assert urllib.request.urlopen(req, timeout=10).status == 201
    bad = urllib.request.Request("http://127.0.0.1:8799/recordings/cam01/20261105T080300.mp4", body, method="PUT")
    with pytest.raises(urllib.error.HTTPError, match="401"):
        urllib.request.urlopen(bad, timeout=10)

    t = observe(sim, out, detector=StandIn(cam, shape), check_gpu=False, log=lambda m: None)
    assert t == {"stored": 3, "rejected": 2, "ingested": 3}
    reasons = {p.name: p.read_text(encoding="utf-8") for p in (box / "rejected").glob("*.reason.txt")}
    assert "not in cameras.json" in reasons["cam99_20261105T080000.mp4.reason.txt"]
    assert "name not understood" in reasons["notes.mp4.reason.txt"]
    assert (out / "recordings" / "2026-11-05" / "day" / "cam01_080200.mp4").exists()

    # what the camera measured: three cars on the approach, at the stand-in's speed
    import sqlite3

    import pandas as pd
    with sqlite3.connect(out / "traffic.db") as c:
        obs = pd.read_sql("SELECT * FROM observations", c)
        kpi = pd.read_sql("SELECT * FROM network_kpi", c)
        st_ = pd.read_sql("SELECT * FROM edge_state WHERE observed = 1", c)
    assert len(obs) >= 1 and set(obs.camera_id) == {"cam01"}
    assert obs.total.median() == 3
    assert obs.speed.median() == pytest.approx(SPEED_MS * 3.6, abs=3)
    assert len(kpi) >= 1 and set(st_.edge_id) == {cam["edge"]}

    # the dashboard reads the town: its map shapes, its names, its title
    from netstudy import dashboard_data as dd
    from netstudy.places import Places
    net = dd.Network("millbrook", root=out.parent)
    assert len(net.shapes()) == 17
    monkeypatch.setattr("netstudy.places.TOWNS", tmp_path / "towns")
    pl = Places("millbrook", net.nodes(), net.edges())
    assert pl.node(centre_j) == "High Street & Mill Lane"
    from netstudy.places import city
    assert city("millbrook")["real"] and city("millbrook")["name"] == "Millbrook"

    steps = {s["step"]: s for s in status(sim, out, "towns/millbrook.yaml")}
    assert all(steps[k]["done"] for k in ("Recordings arriving", "Recordings read (SAM 3.1)",
                                          "Observations and network state"))
    assert not steps["Graph model for Forecast"]["done"]                # one morning is not a week


def test_simulation_commands_refuse_a_real_town(tmp_path):
    from netstudy.cli import main

    f = write_config(tmp_path)
    for cmd in ("network", "record", "run"):
        args = [cmd, "--sim-config", str(f)] + (["--day", "2026-11-05"] if cmd == "record" else [])
        with pytest.raises(SystemExit, match="real town uses netstudy onboard"):
            main(args)


def test_jobs_run_a_command_in_the_background(tmp_path):
    from netstudy import jobs

    f = write_config(tmp_path)
    assert jobs.start(tmp_path, "status", ["status", "--sim-config", str(f)]) is None
    for _ in range(100):
        if not jobs.running(tmp_path, "status"):
            break
        time.sleep(0.2)
    s = jobs.state(tmp_path, "status")
    assert s["state"] == "done" and "Road map imported" in jobs.tail(tmp_path, "status")
