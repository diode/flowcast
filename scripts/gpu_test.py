"""Standalone GPU tests, one card at a time (docs/architecture-next.md 2.3): is each card stable on its own?

  .venv\\Scripts\\python scripts\\gpu_test.py sam --minutes 10        # SAM 3.1 on the vision GPU only
  .venv\\Scripts\\python scripts\\gpu_test.py glimmer --minutes 10    # Glimmer (llama-server) on its GPU only
  .venv\\Scripts\\python scripts\\gpu_test.py carla --minutes 10      # CARLA off screen on the vision GPU only

Before a test starts the other card must be idle (under 40 W); every 30 s it is checked again and the
test stops if it is busy. Both cards are logged every second to data/gpu_check/<test>_<time>.csv (nvidia-smi). A
card that disappears ends the test, and the last readings are kept. Summary -> data/gpu_check/results.json.
GPU numbers are nvidia-smi's (PCI order); which card runs what comes from workspace.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gpu_check"
FIELDS = ("timestamp,index,name,pstate,utilization.gpu,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,"
          "pcie.link.gen.current,pcie.link.width.current,memory.used,clocks_throttle_reasons.active")
from netstudy import workspace  # noqa: E402

CARD = {"sam": workspace.vision_gpu(), "carla": workspace.vision_gpu(), "glimmer": workspace.glimmer_gpu()}
IDLE_W = 40.0


def read() -> dict[int, dict]:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,power.draw,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    out = {}
    for line in r.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            out[int(p[0])] = {"name": p[1], "util": float(p[2]), "watts": float(p[3]), "mib": float(p[4])}
        except (ValueError, IndexError):
            pass
    return out


def other_idle(card: int) -> tuple[bool, str]:
    others = {i: g for i, g in read().items() if i != card}
    if not others:
        return True, "no other GPU"
    busy = {i: g for i, g in others.items() if g["watts"] >= IDLE_W}   # power draw, not utilization: the desktop
    if busy:                                                            # alone shows ~10% use
        i, g = next(iter(busy.items()))
        return False, f"GPU{i} {g['util']:.0f}% {g['watts']:.0f} W"
    return True, "other GPUs idle"


def env_for(card: int) -> dict:
    e = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(card))
    return e


SAM_LOOP = r"""
import sys, time, glob, cv2, torch
from netstudy.vision import Sam3Detector
print("SAM 3.1 on", torch.cuda.get_device_name(0), flush=True)
clips = sorted(glob.glob(r"{root}\data\ashford\recordings\2026-10-05\*\cam0*.mp4"))[:8]
frames = []
for c in clips:
    cap = cv2.VideoCapture(c)
    while len(frames) < 60:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
det = Sam3Detector(device="cuda")
end, n = time.time() + {seconds}, 0
while time.time() < end:
    for f in frames:
        det.detect(f); n += 1
    print("frames", n, flush=True)
"""


def start(test: str, seconds: float) -> subprocess.Popen:
    card = CARD[test]
    log = open(OUT / f"{test}_load.log", "w")
    if test == "sam":
        cmd = [sys.executable, "-c", SAM_LOOP.format(root=ROOT, seconds=seconds)]
    elif test == "glimmer":
        cmd = [str(ROOT / "tools" / "llama.cpp" / "llama-server.exe"), "-m", str(ROOT / "models" /
               "Muse-Glimmer-30B-Q3_K_M.gguf"), "--jinja", "-ngl", "99", "-c", "8192", "--port", "8081",
               "--device", "CUDA0"]                                     # CUDA0 = the only visible card
    else:
        cmd = [str(ROOT / "tools" / "CARLA_0.9.16" / "CarlaUE4.exe"), "-RenderOffScreen", f"-graphicsadapter={dxgi()}",
               "-carla-rpc-port=2010"]
    return subprocess.Popen(cmd, env=env_for(card), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)


def dxgi() -> int:
    """CARLA's -graphicsadapter: the workspace CARLA card's index in Windows' DXGI list (it can change when a card
    is re-enabled)."""
    r = subprocess.run([sys.executable, str(Path(__file__).with_name("dxgi_list.py"))], capture_output=True, text=True)
    hits = [int(line.split()[0]) for line in r.stdout.splitlines() if (workspace.carla_gpu() or "") in line]
    if len(hits) != 1:
        raise SystemExit(f"DXGI: expected one adapter named {workspace.carla_gpu()!r} (workspace.yaml carla_gpu), "
                         f"got {r.stdout!r}")
    return hits[0]


def glimmer_load(stop_at: float) -> int:
    """Keep the server generating: long answers back to back on port 8081."""
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    n = 0
    body = json.dumps({"messages": [{"role": "user", "content": "Describe how traffic signals at a busy junction are "
                                     "timed, in detail, for an engineer."}], "max_tokens": 600, "temperature": 0.7}).encode()
    while time.time() < stop_at:
        req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", body, {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=300).read()
            n += 1
        except Exception:
            break
    return n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("test", choices=list(CARD))
    p.add_argument("--minutes", type=float, default=10)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    card = CARD[a.test]
    ok, why = other_idle(card)
    if not ok:
        raise SystemExit(f"not started: the other card is busy ({why})")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = OUT / f"{a.test}_{stamp}.csv"
    logger = subprocess.Popen(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv", "-l", "1", "-f", str(csv)])
    t0 = time.time()
    stop_at = t0 + 60 * a.minutes
    proc = start(a.test, 60 * a.minutes + (120 if a.test == "sam" else 0))
    outcome, n_load = "passed", None
    try:
        if a.test == "glimmer":
            import threading
            box = {}
            th = threading.Thread(target=lambda: box.update(n=glimmer_load(stop_at)), daemon=True)
            th.start()
        while time.time() < stop_at:
            time.sleep(30)
            seen = read()
            if card not in seen:
                outcome = f"card lost after {time.time() - t0:.0f} s"
                break
            ok, why = other_idle(card)
            if not ok:
                outcome = f"stopped: the other card got busy ({why})"
                break
            if a.test == "sam" and proc.poll() is not None:
                outcome = f"load ended early (exit {proc.returncode}), see {OUT / 'sam_load.log'}"
                break
            g = seen[card]
            print(f"{time.time() - t0:5.0f}s  GPU{card} {g['util']:.0f}% {g['watts']:.0f} W {g['mib']:.0f} MiB  | {why}",
                  flush=True)
        if a.test == "glimmer":
            th.join(timeout=320)                 # the answer being generated when time is up
            n_load = box.get("n")
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if a.test == "carla":
            subprocess.run(["taskkill", "/F", "/IM", "CarlaUE4-Win64-Shipping.exe"], capture_output=True)
        time.sleep(5)
        logger.terminate()
    import csv as _csv
    rows = list(_csv.DictReader(open(csv, encoding="utf-8"), skipinitialspace=True))
    mine = [r for r in rows if r.get("index") == str(card)]
    watts = [float(r["power.draw [W]"].split()[0]) for r in mine if r.get("power.draw [W]", "").split()[:1]
             and r["power.draw [W]"].split()[0].replace(".", "", 1).isdigit()]
    temps = [float(r["temperature.gpu"]) for r in mine if r.get("temperature.gpu", "").isdigit()]
    other = [float(r["power.draw [W]"].split()[0]) for r in rows if r.get("index") == str(1 - card)
             and r.get("power.draw [W]", "").split()[:1] and r["power.draw [W]"].split()[0].replace(".", "", 1).isdigit()]
    res = {"test": a.test, "card": f"GPU{card} {NAME[card]}", "started": stamp, "minutes": round((time.time() - t0) / 60, 1),
           "outcome": outcome, "watts_max": max(watts, default=None),
           "watts_mean": round(sum(watts) / len(watts), 1) if watts else None, "temp_max": max(temps, default=None),
           "other_card_watts_max": max(other, default=None), "answers": n_load, "log": str(csv)}
    allr = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else []
    (OUT / "results.json").write_text(json.dumps(allr + [res], indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
