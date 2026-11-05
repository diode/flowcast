"""Does each camera's calibration fit its video? Draws the camera's measured zone (its approach lanes, view_m
before the stop line), the counting line and the speed lines on a frame of its recording, and SAM 3.1's
detections with where each lands on the road. For review by eye.

    .venv\\Scripts\\python scripts\\check_calibration.py --sim-config towns/ashford.yaml --day 2026-10-07 [--cameras cam06]

Writes <out_dir>/check/calib/<camera>.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netstudy.calib import Calib, Zone  # noqa: E402
from netstudy.cli import load_sim_config  # noqa: E402
from netstudy.network import load_cameras  # noqa: E402
from netstudy.record import measure_settings  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sim-config", required=True)
    p.add_argument("--day", required=True)
    p.add_argument("--cameras", help="comma-separated camera ids (default: all)")
    p.add_argument("--second", type=float, default=30.0, help="which second of the first segment to draw on")
    p.add_argument("--no-detect", action="store_true")
    args = p.parse_args()

    import cv2
    import numpy as np
    import sumolib

    sim, out_dir = load_sim_config(args.sim_config)
    net = sumolib.net.readNet(str(out_dir / "network" / "net.net.xml"))
    cams = load_cameras(out_dir / "cameras.json")
    if args.cameras:
        cams = [c for c in cams if c["camera_id"] in args.cameras.split(",")]
    calib, ms, view_m = Calib(sim), measure_settings(sim), float(sim["network"].get("view_m", 60))
    det = None
    if not args.no_detect:
        from netstudy.vision import Sam3Detector
        det = Sam3Detector()
    out = out_dir / "check" / "calib"
    out.mkdir(parents=True, exist_ok=True)
    for c in cams:
        seg = sorted((out_dir / "recordings" / args.day).glob(f"*/{c['camera_id']}_*.mp4"))
        if not seg:
            print(f"{c['camera_id']}: no recording")
            continue
        cap = cv2.VideoCapture(str(seg[0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.second * (cap.get(cv2.CAP_PROP_FPS) or 5)))
        ok, img = cap.read()
        if not ok:
            continue
        zone = Zone(c, net, view_m)
        pts = [calib.to_pixel(c, x, y) for x, y in zone.outline()]
        pts = np.array([q for q in pts if q is not None], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.polylines(img, [pts], True, (0, 200, 255), 2)
        lane0, lanen = zone.lanes[0], zone.lanes[-1]
        for metres, colour in ((ms["line_m"], (0, 0, 255)), *((m, (255, 0, 255)) for m in ms["speed_lines_m"])):
            from sumolib.geomhelper import positionAtShapeOffset
            a = positionAtShapeOffset(lane0.getShape(), lane0.getLength() - metres)
            b = positionAtShapeOffset(lanen.getShape(), lanen.getLength() - metres)
            pa, pb = calib.to_pixel(c, *a), calib.to_pixel(c, *b)
            if pa and pb:
                cv2.line(img, tuple(map(int, pa)), tuple(map(int, pb)), colour, 2)
        n_in = 0
        for d in det.detect(img) if det else []:
            g = calib.to_ground(c, *d.bottom_center)
            where = zone.locate(*g) if g else None
            n_in += where is not None
            colour = (0, 255, 0) if where else (160, 160, 160)
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), colour, 2)
            text = f"{d.category} lane {where[0]} {where[1]:.0f} m" if where else f"{d.category} (not on approach)"
            cv2.putText(img, text, (int(d.x1), int(d.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{c['camera_id']}.png"), img)
        print(f"{c['camera_id']} {c['edge']:>6}: zone drawn, {n_in} vehicles on the approach")
    print(f"Images in {out}")


if __name__ == "__main__":
    main()
