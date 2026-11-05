"""Two-GPU check for a workspace with separate cards: Glimmer (llama-server) on the workspace's Traffic Assistant
GPU and SAM 3.1 on its vision GPU (workspace.yaml), alone and at the same time.

GPU numbers are nvidia-smi's (PCI order). CUDA_DEVICE_ORDER is set to PCI_BUS_ID before torch or llama-server
starts; without it torch numbers the cards fastest first.

    .venv\\Scripts\\python scripts\\check_gpus.py                  # all phases
    .venv\\Scripts\\python scripts\\check_gpus.py --phase sam      # sam | glimmer | both

Results go to data/ashford/check/gpus.json (written after every phase, so a lost GPU keeps what was measured).
"""
from __future__ import annotations

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = sorted((ROOT / "data" / "ashford" / "recordings" / "2026-10-05").glob("*/cam0[1-4]_*00.mp4"))[:8]
OUT = ROOT / "data" / "ashford" / "check" / "gpus.json"
SERVER = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
MODEL = ROOT / "models" / "Muse-Glimmer-30B-Q3_K_M.gguf"
PORT = 8080
from netstudy import workspace  # noqa: E402

SAM_GPU, GLIMMER_GPU = workspace.vision_gpu(), workspace.glimmer_gpu()
QUESTION = "Elm Park to Ashford Centre jammed tomorrow 8am for an hour"
TOOLS = [
    {"type": "function", "function": {
        "name": "find_road", "description": "Find the camera road (edge id) between two places.",
        "parameters": {"type": "object", "properties": {"from_place": {"type": "string"}, "to_place": {"type": "string"}},
                       "required": ["from_place", "to_place"]}}},
    {"type": "function", "function": {
        "name": "set_road", "description": "Set a road's state for a period, for the graph model's road change.",
        "parameters": {"type": "object", "properties": {
            "edge": {"type": "string"}, "state": {"type": "string", "enum": ["closed", "jammed", "slow"]},
            "minutes": {"type": "integer"}, "date": {"type": "string"}, "start": {"type": "string"}},
            "required": ["edge", "state", "minutes", "date", "start"]}}},
]


def smi() -> dict[int, int]:
    """MiB used per GPU, as nvidia-smi reports it."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}


def frames(n: int):
    import cv2

    got = []
    for clip in CLIPS:
        cap = cv2.VideoCapture(str(clip))
        while len(got) < n:
            ok, f = cap.read()
            if not ok:
                break
            got.append(f)
        cap.release()
        if len(got) >= n:
            break
    return got


class Sam:
    def __init__(self):
        import torch

        torch.cuda.set_device(SAM_GPU)
        from netstudy.vision import Sam3Detector

        print("SAM 3.1 on", torch.cuda.get_device_name(SAM_GPU), flush=True)
        t = time.perf_counter()
        self.det = Sam3Detector(device="cuda")
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t
        self.torch = torch

    def run(self, imgs, stop: threading.Event | None = None) -> dict:
        torch = self.torch
        torch.cuda.set_device(SAM_GPU)                        # per thread: a new thread starts on device 0
        for f in imgs[:3]:                                    # warm-up
            self.det.detect(f)
        torch.cuda.reset_peak_memory_stats()
        n, boxes, t = 0, 0, time.perf_counter()
        while True:
            for f in imgs:
                boxes += len(self.det.detect(f))
                n += 1
            if stop is None or stop.is_set():
                break
        s = time.perf_counter() - t
        return {"frames": n, "seconds": round(s, 1), "fps": round(n / s, 2), "boxes_per_frame": round(boxes / n, 1),
                "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20)}


class Glimmer:
    def __init__(self):
        self.log = open(OUT.parent / "llama-server.log", "w")
        t = time.perf_counter()
        self.proc = subprocess.Popen([str(SERVER), "-m", str(MODEL), "--jinja", "-ngl", "99", "-c", "8192",
                                      "--port", str(PORT), "--device", f"CUDA{GLIMMER_GPU}"],
                                     stdout=self.log, stderr=subprocess.STDOUT, env=os.environ.copy())
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server stopped (code {self.proc.returncode}), see {self.log.name}")
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(0.5)
        self.load_s = time.perf_counter() - t

    def ask(self) -> dict:
        body = {"messages": [
            {"role": "system", "content": "You answer questions about Ashford's road network. Today is 2026-11-03. "
                                          "Use the tools; never make up numbers."},
            {"role": "user", "content": QUESTION}], "tools": TOOLS, "temperature": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.perf_counter()
        r = json.load(urllib.request.urlopen(req, timeout=300))
        wall = time.perf_counter() - t
        tm, msg = r.get("timings", {}), r["choices"][0]["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in msg.get("tool_calls") or []]
        return {"wall_s": round(wall, 2), "prompt_tok_s": round(tm.get("prompt_per_second", 0), 1),
                "gen_tok_s": round(tm.get("predicted_per_second", 0), 1), "gen_tokens": tm.get("predicted_n"),
                "tool_calls": calls}

    def stop(self):
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()


def summary(asks: list[dict]) -> dict:
    return {"gen_tok_s": round(sum(a["gen_tok_s"] for a in asks) / len(asks), 1),
            "prompt_tok_s": round(sum(a["prompt_tok_s"] for a in asks) / len(asks), 1),
            "first_tool": sorted({a["tool_calls"][0][0] if a["tool_calls"] else "-" for a in asks}), "asks": asks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sam", "glimmer", "both", "all"], default="all")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--asks", type=int, default=5)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["idle_mib"] = smi()

    def save(key, value):
        res[key] = value
        OUT.write_text(json.dumps(res, indent=2))
        print(key, json.dumps({k: v for k, v in value.items() if k != "asks"}))

    imgs = frames(a.frames)
    print(f"{len(imgs)} frames {imgs[0].shape[1]}x{imgs[0].shape[0]} from {len(CLIPS)} clips")
    if a.phase in ("sam", "all"):
        sam = Sam()
        save("sam_alone", {"gpu": SAM_GPU, "load_s": round(sam.load_s, 1), **sam.run(imgs), "smi_mib": smi()})
        del sam
        import torch
        torch.cuda.empty_cache()
    if a.phase in ("glimmer", "all"):
        g = Glimmer()
        try:
            save("glimmer_alone", {"gpu": GLIMMER_GPU, "load_s": round(g.load_s, 1), "smi_mib": smi(),
                                   **summary([g.ask() for _ in range(a.asks)])})
        finally:
            g.stop()
    if a.phase in ("both", "all"):
        g = Glimmer()
        try:
            sam = Sam()
            stop, out = threading.Event(), {}
            th = threading.Thread(target=lambda: out.update(sam.run(imgs, stop)))
            th.start()
            time.sleep(5)
            asks = [g.ask() for _ in range(a.asks)]
            mib = smi()
            stop.set()
            th.join()
            if not out:
                raise SystemExit("SAM 3.1 stopped while Glimmer was answering (see the thread's error above)")
            save("both", {"glimmer": {"load_s": round(g.load_s, 1), **summary(asks)}, "sam": out, "smi_mib": mib})
        finally:
            g.stop()


if __name__ == "__main__":
    main()
