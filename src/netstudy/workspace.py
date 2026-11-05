"""The workspace machine: which GPU runs what, on whatever machine FlowCast is set up on.

Nothing about one particular machine is built into the code. Settings, highest first:
  1. environment variables  FLOWCAST_VISION_GPU, FLOWCAST_GLIMMER_GPU, FLOWCAST_CARLA_GPU, FLOWCAST_ONE_GPU_AT_A_TIME
  2. workspace.yaml in the project root (this machine's own file, not shared; see workspace.example.yaml)
  3. defaults: one GPU, number 0, for everything; no limit on GPUs under load together

  vision_gpu()      SAM 3.1 (ingest), nvidia-smi / PCI numbering
  glimmer_gpu()     the local Traffic Assistant (llama-server)
  carla_gpu()       sample towns only: the card CARLA renders on, by (part of) its name; None = the vision GPU's
  one_at_a_time()   True when the power supply cannot carry two GPUs under heavy load: ingest waits while another
                    card is busy, and Glimmer will not start while one is
GPU numbers are nvidia-smi's (PCI bus order); FlowCast sets CUDA_DEVICE_ORDER=PCI_BUS_ID so torch and llama.cpp agree.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "workspace.yaml"
DEFAULTS = {"vision_gpu": 0, "glimmer_gpu": 0, "carla_gpu": None, "one_gpu_at_a_time": False}
ENV = {"vision_gpu": "FLOWCAST_VISION_GPU", "glimmer_gpu": "FLOWCAST_GLIMMER_GPU", "carla_gpu": "FLOWCAST_CARLA_GPU",
       "one_gpu_at_a_time": "FLOWCAST_ONE_GPU_AT_A_TIME"}


@lru_cache(maxsize=1)
def _file() -> dict:
    if not FILE.exists():
        return {}
    import yaml
    return yaml.safe_load(FILE.read_text(encoding="utf-8")) or {}


def setting(key: str):
    env = os.environ.get(ENV[key])
    if env not in (None, ""):
        if key == "one_gpu_at_a_time":
            return env.strip().lower() in ("1", "true", "yes", "on")
        return env if key == "carla_gpu" else int(env)
    return _file().get(key, DEFAULTS[key])


def vision_gpu() -> int:
    return int(setting("vision_gpu"))


def glimmer_gpu() -> int:
    return int(setting("glimmer_gpu"))


def carla_gpu() -> str | None:
    v = setting("carla_gpu")
    return str(v) if v else None


def one_at_a_time() -> bool:
    return bool(setting("one_gpu_at_a_time"))


def describe() -> str:
    src = "workspace.yaml" if FILE.exists() else "defaults (no workspace.yaml)"
    return (f"vision GPU {vision_gpu()}, Traffic Assistant GPU {glimmer_gpu()}, CARLA card "
            f"{carla_gpu() or 'the vision GPU'}, one GPU under load at a time: {one_at_a_time()}; from {src}")
