"""Can CARLA film a city, and how should its cameras be exposed? Builds the city's world from its OpenDRIVE file,
drives some cars round it, and per time of day (morning, noon, evening, night) photographs from the cameras
after the camera's auto-exposure has settled (docs/recording.md).

    CarlaUE4.exe -RenderOffScreen                      # CARLA server, in another window
    .venv\\Scripts\\python scripts\\check_carla_world.py --sim-config towns/ashford.yaml [--cameras cam06,cam02]
    .venv\\Scripts\\python scripts\\check_carla_world.py --sim-config towns/ashford.yaml --calibrate

--calibrate tries exposure settings per time of day and prints the one whose frames come closest to the
target brightness (carla_world.TARGET_GREY), for the city config's recording.exposure. Photos go to
<out_dir>/check/carla/<light>/<camera>_<edge>.png.
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netstudy.carla_world import (LIGHTS, TARGET_GREY, build_world, camera_blueprint, camera_pose, dress,  # noqa: E402
                                  exposure, headlights_on, pose_settings, weather)
from netstudy.cli import load_sim_config  # noqa: E402
from netstudy.network import load_cameras  # noqa: E402

SETTLE_S = 6.0                       # seconds of video before the frame that is judged


def candidates(light: str) -> list[dict]:
    """Exposure settings tried by --calibrate: compensation by day; at night the auto-exposure range."""
    if light == "night":                                   # one fixed exposure for every camera
        return [{"exposure_compensation": 0.0, "exposure_min_bright": ev, "exposure_max_bright": ev}
                for ev in (5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0)]
    return [{"exposure_compensation": c} for c in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]


def label(attrs: dict) -> str:
    if "exposure_min_bright" in attrs:
        lo, hi = attrs["exposure_min_bright"], attrs["exposure_max_bright"]
        return f"fixed {lo:+.1f} EV" if lo == hi else f"range {lo:+.1f}..{hi:+.1f} EV"
    return f"{attrs['exposure_compensation']:+.1f}"


def shoot(client, world, cams, pose, attrs: dict, fps: int, light: str, out: Path | None) -> float:
    """Mean grey of every camera's frame after SETTLE_S seconds with these exposure settings."""
    import cv2

    bp = camera_blueprint(world, pose, attrs)
    sensors, queues = [], {}
    for c in cams:
        q: queue.Queue = queue.Queue()
        s = world.spawn_actor(bp, camera_pose(c, pose))
        s.listen(q.put)
        sensors.append(s)
        queues[c["camera_id"]] = q
    last = {}
    for _ in range(int(SETTLE_S * fps)):
        if light == "night":
            headlights_on(client, world)
        world.tick()
        for cid, q in queues.items():
            last[cid] = q.get(timeout=60)
    means = []
    for c in cams:
        img = last[c["camera_id"]]
        bgr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
        means.append(float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()))
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{c['camera_id']}_{c['edge']}.png"), bgr)
    if out is not None:
        print("      per camera: " + "  ".join(f"{c['camera_id']} {m:.0f}" for c, m in zip(cams, means)))
    for s in sensors:
        s.stop()
        s.destroy()
    world.tick()
    return float(np.mean(means))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sim-config", required=True)
    p.add_argument("--cameras", help="comma-separated camera ids (default: all)")
    p.add_argument("--lights", default=",".join(LIGHTS), help="times of day to check")
    p.add_argument("--calibrate", action="store_true", help="search the exposure per time of day")
    p.add_argument("--vehicles", type=int, default=110)
    p.add_argument("--drive-s", type=float, default=30.0, help="seconds of driving before the photos")
    p.add_argument("--port", type=int, default=2000)
    args = p.parse_args()

    import carla
    import sumolib

    sim, out_dir = load_sim_config(args.sim_config)
    pose = pose_settings(sim)
    fps = int((sim.get("recording") or {}).get("fps", 5))
    cams = load_cameras(out_dir / "cameras.json")
    if args.cameras:
        cams = [c for c in cams if c["camera_id"] in args.cameras.split(",")]
    shots = out_dir / "check" / "carla"

    client = carla.Client("localhost", args.port)
    client.set_timeout(600.0)
    client.get_server_version()                                # waits until the server answers
    t0 = time.time()
    world = build_world(client, out_dir / "network" / "net.xodr")
    props = dress(world, sumolib.net.readNet(str(out_dir / "network" / "net.net.xml")), seed=int(sim.get("seed", 0)))
    print(f"World built in {time.time() - t0:.0f} s, {len(props)} roadside props")

    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / fps
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    bps = [b for b in world.get_blueprint_library().filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    spawned = []
    best = {}
    try:
        for i, sp in enumerate(world.get_map().get_spawn_points()[: args.vehicles]):
            v = world.try_spawn_actor(bps[i % len(bps)], sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                spawned.append(v)
        for _ in range(int(args.drive_s * fps)):
            world.tick()
        print(f"{len(spawned)} vehicles driving; judging the frame after {SETTLE_S:.0f} s of video")
        for light in args.lights.split(","):
            world.set_weather(weather(light))
            target = TARGET_GREY[light]
            if not args.calibrate:
                attrs = exposure(sim, light)
                grey = shoot(client, world, cams, pose, attrs, fps, light, shots / light)
                print(f"  {light:8} exposure {label(attrs)}: grey {grey:.0f} (target {target})")
                continue
            tried = [(a, shoot(client, world, cams, pose, a, fps, light, None)) for a in candidates(light)]
            attrs, _ = min(tried, key=lambda t: abs(t[1] - target))
            grey = shoot(client, world, cams, pose, attrs, fps, light, shots / light)
            best[light] = attrs
            print(f"  {light:8} " + "  ".join(f"{label(a)}:{g:.0f}" for a, g in tried)
                  + f"  -> {label(attrs)}, grey {grey:.0f} (target {target})")
    finally:
        client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in spawned])
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None
        world.apply_settings(settings)
    if best:
        print("recording:\n  exposure:")
        for k, v in best.items():
            print(f"    {k}: {{" + ", ".join(f"{a}: {b}" for a, b in v.items()) + "}")
    print(f"Photos in {shots}")


if __name__ == "__main__":
    main()
