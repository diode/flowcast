"""The Traffic Assistant's model (architecture-next 4.4): Glimmer on this computer (llama-server on the
workspace's Traffic Assistant GPU, glimmer_server) or Muse Spark through the Meta Model API (OpenAI-style chat completions; questions and tool
results leave the site). The choice is kept in data/assistant_backend.json, for every page and session.

  current()     "glimmer" | "muse_spark"
  choose(b)     switch: Muse Spark stops the Glimmer server this computer started (frees its GPU); Glimmer starts it
  status()      "off" | "loading" | "ready" (Glimmer, as glimmer_server) | "ready" | "no key" (Muse Spark)

The API key: MODEL_API_KEY in the environment (the dashboard also copies it from .streamlit/secrets.toml).
MUSE_SPARK_URL and MUSE_SPARK_MODEL override the endpoint and the model id.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data" / "assistant_backend.json"
CHOICES = {"muse_spark": "Muse Spark · Meta Model API (production)",
           "glimmer": "Glimmer · local, development (low cost)"}
DEFAULT = "glimmer"
KEY_ENV = "MODEL_API_KEY"
API_URL = os.environ.get("MUSE_SPARK_URL", "https://api.meta.ai/v1")
API_MODEL = os.environ.get("MUSE_SPARK_MODEL", "muse-spark-1.1")


def current() -> str:
    try:
        b = json.loads(STATE.read_text(encoding="utf-8"))["backend"]
    except (OSError, ValueError, KeyError):
        return DEFAULT
    return b if b in CHOICES else DEFAULT


def api_key() -> str | None:
    return os.environ.get(KEY_ENV) or None


def choose(backend: str) -> str | None:
    """Keep the choice and switch the local server to match; a message when Glimmer could not start (another GPU busy)."""
    from . import glimmer_server as gs

    if backend not in CHOICES:
        raise ValueError(f"unknown assistant backend {backend!r}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"backend": backend}), encoding="utf-8")
    if backend == "muse_spark":
        gs.stop()                                       # only a server this module started; its GPU is free again
        return None
    if gs.status() == "off":
        try:
            gs.start()
        except SystemExit as e:                         # refused: another GPU under load (workspace.one_at_a_time)
            return str(e)
    return None


def status(backend: str | None = None) -> str:
    from . import glimmer_server as gs

    if (backend or current()) == "muse_spark":
        return "ready" if api_key() else "no key"
    s = gs.status()
    return "ready" if s == "running elsewhere" else s
