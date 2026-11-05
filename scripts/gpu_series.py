"""GPU test series, one card at a time, with a cool-down before each test (docs/gpu-tests.md, part 2).

  .venv\\Scripts\\python scripts\\gpu_series.py                       # the whole series
  .venv\\Scripts\\python scripts\\gpu_series.py --only sam_clean      # one test

Tests (3 minutes of load each):
  sam_kill    vision GPU: SAM 3.1 in a loop, killed at full load (as an interrupted job)
  sam_clean   vision GPU: SAM 3.1 in a loop that finishes its frame, frees the model and exits by itself
  record      vision GPU: a real CARLA recording of Ashford (20 cameras, 1280x720, 5 fps, 2 simulated minutes) into a
              scratch folder; CARLA asked to close first, killed only if it does not
  glimmer_q   Glimmer GPU: Glimmer answering test questions with 10 s pauses (real Predict use), the server stopped idle
Cool-down: at least 2 minutes and until the card is below 50 °C (at most 10 minutes). The other card must stay
under 40 W. Each test: nvidia-smi log every second, the stop time, and any nvlddmkm / WHEA / Kernel-Power event
from its start to 60 s after it ended -> data/gpu_check/series.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gpu_test import OUT, ROOT, dxgi, env_for, other_idle, read  # noqa: E402

LOAD_S = 180
COOL_MIN_S, COOL_MAX_S, COOL_C = 120, 600, 50
FIELDS = ("timestamp,index,pstate,utilization.gpu,power.draw,temperature.gpu,clocks.sm,pcie.link.gen.current,"
          "pcie.link.width.current,memory.used")
SERIES = ["sam_kill", "sam_clean", "sam_kill", "sam_clean", "record", "glimmer_q"]
from netstudy import workspace  # noqa: E402

V, G = workspace.vision_gpu(), workspace.glimmer_gpu()
CARD = {"sam_kill": V, "sam_clean": V, "record": V, "glimmer_q": G, "glimmer_kill": G, "glimmer_clean": G}
MINIMAL = ["sam_kill", "sam_clean", "glimmer_kill", "glimmer_clean"]
SCRATCH = Path(__file__).resolve().parents[1] / "data" / "gpu_check" / "rec_test"

SAM = r"""
import glob, time, cv2, torch
from netstudy.vision import Sam3Detector
print("SAM 3.1 on", torch.cuda.get_device_name(0), flush=True)
frames = []
for c in sorted(glob.glob(r"{root}\data\ashford\recordings\2026-10-05\*\cam0*.mp4"))[:8]:
    cap = cv2.VideoCapture(c)
    while len(frames) < 60:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
det = Sam3Detector(device="cuda")
print("loaded", flush=True)
end, n = time.time() + {seconds}, 0
while time.time() < end:
    det.detect(frames[n % len(frames)]); n += 1
print("frames", n, flush=True)
del det                                   # clean end: free the model, let the card come down, then exit
torch.cuda.synchronize(); torch.cuda.empty_cache()
time.sleep(3)
print("done", flush=True)
"""

QUESTIONS = ["Elm Park to Ashford Centre jammed tomorrow 8am for an hour",
             "What if the road from Ashford Centre to Mill Yard is closed at 17:30 tomorrow for 60 minutes?",
             "How busy is the town usually on Wednesday at 8am?",
             "Northgate to Elm Park busy for 45 minutes tomorrow at 17:00 - what should we do?",
             "What's the way round if Mill Yard to Northgate is closed?"]


def temp(card: int) -> float | None:
    r = subprocess.run(["nvidia-smi", "-i", str(card), "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def cool_down(card: int, log):
    t0 = time.time()
    while True:
        t = temp(card)
        waited = time.time() - t0
        if waited >= COOL_MIN_S and (t is None or t < COOL_C) or waited >= COOL_MAX_S:
            log(f"  cool-down {waited:.0f} s, GPU{card} {t} °C")
            return round(waited), t
        time.sleep(10)


def events_since(t0: datetime) -> list[str]:
    ps = ("Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'%s'} -ErrorAction SilentlyContinue | "
          "Where-Object { $_.ProviderName -match 'nvlddmkm|WHEA|Kernel-Power' } | ForEach-Object { '{0} {1} {2} {3}' -f "
          "$_.TimeCreated.ToString('HH:mm:ss.fff'), $_.ProviderName, $_.Id, (($_.Properties | Select-Object -First 2 | "
          "ForEach-Object { $_.Value }) -join ' | ') }") % t0.strftime("%Y-%m-%d %H:%M:%S")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    return [x for x in r.stdout.strip().splitlines() if x]


def run_sam(kill: bool, log) -> dict:
    lf = open(OUT / "series_sam.log", "w")
    secs = LOAD_S + (120 if kill else 0)            # a killed run is still busy when it is killed
    p = subprocess.Popen([sys.executable, "-c", SAM.format(root=ROOT, seconds=secs)], env=env_for(1),
                         stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
    t_load = time.time()
    while "loaded" not in (OUT / "series_sam.log").read_text(errors="replace") and p.poll() is None:
        time.sleep(1)
    t_load = time.time()
    if kill:
        time.sleep(LOAD_S)
        g = read().get(1, {})
        p.kill()                                                    # TerminateProcess: at full load
        stop = datetime.now()
        p.wait()
        return {"stop": stop.strftime("%H:%M:%S.%f")[:-3], "how": "killed", "watts_at_stop": g.get("watts")}
    p.wait(timeout=LOAD_S + 300)
    return {"stop": datetime.now().strftime("%H:%M:%S.%f")[:-3], "how": f"exited itself (code {p.returncode})",
            "load_s": round(time.time() - t_load)}


def run_record(log) -> dict:
    """A real recording: CARLA on the vision GPU, Ashford's world, 20 cameras, 08:00-08:02 of 6 Oct, into scratch."""
    from netstudy.cli import load_sim_config
    from netstudy.network import load_cameras
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "network").mkdir(parents=True)
    src = ROOT / "data" / "ashford"
    for f in (src / "network").iterdir():
        if f.is_file():
            shutil.copy(f, SCRATCH / "network" / f.name)
    shutil.copy(src / "cameras.json", SCRATCH / "cameras.json")
    adapter = dxgi()
    carla = subprocess.Popen([str(ROOT / "tools" / "CARLA_0.9.16" / "CarlaUE4.exe"), "-RenderOffScreen",
                              f"-graphicsadapter={adapter}", "-carla-rpc-port=2010"], cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import socket
    t0 = time.time()
    while time.time() - t0 < 300:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", 2010)) == 0:
                break
        time.sleep(2)
    time.sleep(15)
    log(f"  CARLA up on DXGI adapter {adapter}")
    code = (f"from datetime import date; from pathlib import Path; from netstudy.cli import load_sim_config; "
            f"from netstudy.network import load_cameras; from netstudy.record import record_day; "
            f"sim, _ = load_sim_config(r'{ROOT / 'towns' / 'ashford.yaml'}'); out = Path(r'{SCRATCH}'); "
            f"s = record_day(date(2026, 10, 6), sim, out, load_cameras(out / 'cameras.json'), "
            f"[{{'period': 'morning', 'start': '08:00', 'end': '08:02'}}], port=2010); print(s)")
    t_rec = time.time()
    r = subprocess.run([sys.executable, "-c", code], env=env_for(1), cwd=ROOT, capture_output=True, text=True,
                       timeout=3600)
    rec_s = round(time.time() - t_rec)
    (OUT / "series_record.log").write_text(r.stdout[-4000:] + "\n" + r.stderr[-4000:])
    g = read().get(1, {})
    subprocess.run(["taskkill", "/IM", "CarlaUE4-Win64-Shipping.exe"], capture_output=True)    # ask it to close
    try:
        carla.wait(30)
        how = "closed when asked"
    except subprocess.TimeoutExpired:
        how = "killed (did not close when asked)"
    for name in ("CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe"):
        subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)
    stop = datetime.now()
    segs = len(list((SCRATCH / "recordings").rglob("*.mp4"))) if (SCRATCH / "recordings").exists() else 0
    return {"stop": stop.strftime("%H:%M:%S.%f")[:-3], "how": how, "record_s": rec_s, "segments": segs,
            "record_exit": r.returncode, "watts_at_stop": g.get("watts")}


def run_glimmer(log) -> dict:
    import urllib.request
    srv = subprocess.Popen([str(ROOT / "tools" / "llama.cpp" / "llama-server.exe"), "-m",
                            str(ROOT / "models" / "Muse-Glimmer-30B-Q3_K_M.gguf"), "--jinja", "-ngl", "99", "-c", "8192",
                            "--port", "8081", "--device", "CUDA0"], env=env_for(G), cwd=ROOT,
                           stdout=open(OUT / "series_glimmer.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    end, n = time.time() + LOAD_S, 0
    while time.time() < end:
        body = json.dumps({"messages": [{"role": "user", "content": QUESTIONS[n % len(QUESTIONS)]}],
                           "max_tokens": 300, "temperature": 0}).encode()
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", body,
                                                      {"Content-Type": "application/json"}), timeout=300).read()
        n += 1
        time.sleep(10)                                              # a person reading the answer
    time.sleep(5)                                                   # idle before it is stopped
    g = read().get(0, {})
    srv.terminate()
    srv.wait(30)
    return {"stop": datetime.now().strftime("%H:%M:%S.%f")[:-3], "how": "stopped while idle", "questions": n,
            "watts_at_stop": g.get("watts")}


def run_glimmer_stop(kill: bool, log) -> dict:
    """Glimmer generating non-stop for LOAD_S, then the server killed mid-answer, or (clean) the answer finished,
    the server idle for 5 s and stopped."""
    import threading
    import urllib.request
    srv = subprocess.Popen([str(ROOT / "tools" / "llama.cpp" / "llama-server.exe"), "-m",
                            str(ROOT / "models" / "Muse-Glimmer-30B-Q3_K_M.gguf"), "--jinja", "-ngl", "99", "-c", "8192",
                            "--port", "8081", "--device", "CUDA0"], env=env_for(G), cwd=ROOT,
                           stdout=open(OUT / "series_glimmer.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    end = time.time() + LOAD_S
    body = json.dumps({"messages": [{"role": "user", "content": "Explain traffic signal timing in detail."}],
                       "max_tokens": 400, "temperature": 0.7}).encode()
    box = {"n": 0}

    def gen():
        while time.time() < end:
            try:
                urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", body,
                                                              {"Content-Type": "application/json"}), timeout=300).read()
                box["n"] += 1
            except Exception:
                return
    th = threading.Thread(target=gen, daemon=True)
    th.start()
    time.sleep(LOAD_S)
    g = read().get(0, {})
    if kill:
        srv.kill()                                                  # mid-answer, at full load
        how = "killed mid-answer"
    else:
        th.join(timeout=120)                                        # the last answer finishes
        time.sleep(5)
        g = read().get(0, {})
        srv.terminate()
        how = "stopped while idle"
    stop = datetime.now()
    srv.wait(30)
    return {"stop": stop.strftime("%H:%M:%S.%f")[:-3], "how": how, "answers": box["n"], "watts_at_stop": g.get("watts")}


def main(argv=None):
    global LOAD_S, COOL_MIN_S
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=sorted(CARD))
    p.add_argument("--minimal", action="store_true", help="sam_kill, sam_clean, glimmer_kill, glimmer_clean")
    p.add_argument("--load-s", type=int, default=LOAD_S, help="seconds of load per test (default 180)")
    p.add_argument("--cool-min-s", type=int, default=COOL_MIN_S, help="least cool-down before a test (default 120)")
    p.add_argument("--event-wait-s", type=int, default=60, help="wait after a stop for driver events (default 60)")
    a = p.parse_args(argv)
    LOAD_S, COOL_MIN_S = a.load_s, a.cool_min_s
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    results = []
    for i, test in enumerate([a.only] if a.only else MINIMAL if a.minimal else SERIES, start=1):
        card = CARD[test]
        log(f"{i}. {test} on GPU{card}")
        waited, t_start = cool_down(card, log)
        ok, why = other_idle(card)
        if not ok:
            log(f"  skipped: the other card is busy ({why})")
            results.append({"test": test, "outcome": f"skipped ({why})"})
            continue
        t0 = datetime.now()
        csv = OUT / f"series_{test}_{t0:%H%M%S}.csv"
        logger = subprocess.Popen(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv", "-l", "1", "-f", str(csv)])
        try:
            if test.startswith("sam"):
                r = run_sam(test == "sam_kill", log)
            elif test == "record":
                r = run_record(log)
            elif test in ("glimmer_kill", "glimmer_clean"):
                r = run_glimmer_stop(test == "glimmer_kill", log)
            else:
                r = run_glimmer(log)
            outcome = "ran"
        except Exception as e:                                      # noqa: BLE001
            r, outcome = {}, f"error: {e}"
        time.sleep(a.event_wait_s)                                  # events can come a little after the stop
        logger.terminate()
        lost = card not in read()
        ev = events_since(t0)
        res = {"test": test, "card": card, "start": f"{t0:%H:%M:%S}", "cooled_s": waited, "start_temp": t_start,
               **r, "outcome": "CARD LOST" if lost else outcome, "events": ev, "log": csv.name}
        log(f"  {json.dumps(res)}")
        results.append(res)
        if lost:
            log("  stopping the series: the card is gone (Device Manager: disable / enable)")
            break
    prev = json.loads((OUT / "series.json").read_text()) if (OUT / "series.json").exists() else []
    (OUT / "series.json").write_text(json.dumps(prev + results, indent=2))


if __name__ == "__main__":
    main()
