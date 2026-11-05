"""Still images from recorded video: any recording in the recordings layout, CARLA's or a real camera's.

  recordings/<date>/<period>/<camera>_<HHMMSS>.mp4  ->  frames/<date>/<period>/<camera>_<HHMMSS>.jpg

One image every `every_s` seconds on the clock (default 300: the frame at each 5-minute slot, the one ingest
reads); the file name is the frame's own time.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


def extract(rec_dir: Path, out_dir: Path, every_s: float = 300, day: str | None = None) -> int:
    import cv2

    n = 0
    for mp4 in sorted(Path(rec_dir).glob(f"{day or '*'}/*/*.mp4")):
        cam, hhmmss = mp4.stem.rsplit("_", 1)
        start = datetime.strptime(f"{mp4.parent.parent.name} {hhmmss}", "%Y-%m-%d %H%M%S")
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        target = out_dir / mp4.parent.parent.name / mp4.parent.name
        k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = start + timedelta(seconds=k / fps)
            secs = (t - t.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            if abs(secs - round(secs / every_s) * every_s) < 0.5 / fps:
                target.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target / f"{cam}_{t:%H%M%S}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
            k += 1
        cap.release()
    return n
