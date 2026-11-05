"""Where this project's external tools live (docs/setup.md). Everything is inside the project folder:

  tools/CARLA_0.9.16/      CARLA server (CarlaUE4.exe) and its SUMO co-simulation bridge (Co-Simulation/Sumo)
  .venv                    Python, SUMO (eclipse-sumo), CARLA's Python client, PyTorch

CARLA_ROOT (environment) overrides the CARLA folder; ffmpeg comes from PATH unless recording.ffmpeg is set.
`scripts/setup_tools.ps1` downloads and checks them.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
CARLA_VERSION = "0.9.16"


def _carla_ok(root: Path) -> bool:
    return (root / "CarlaUE4.exe").exists() and (root / "Co-Simulation" / "Sumo").exists()


def carla_root() -> Path:
    own = TOOLS / f"CARLA_{CARLA_VERSION}"
    root = Path(os.environ.get("CARLA_ROOT") or own)
    if not _carla_ok(root) and root != own and _carla_ok(own):      # a stale CARLA_ROOT (e.g. an old project folder)
        import warnings
        warnings.warn(f"CARLA_ROOT={root} has no CARLA {CARLA_VERSION}; using the project's {own}")
        root = own
    if not _carla_ok(root):
        raise SystemExit(f"CARLA {CARLA_VERSION} not found in {root} (needs CarlaUE4.exe and Co-Simulation\\Sumo): "
                         "run scripts\\setup_tools.ps1, or set CARLA_ROOT")
    return root


def carla_server() -> Path:
    return carla_root() / "CarlaUE4.exe"


def ffmpeg(configured: str | None = None) -> str:
    exe = configured or shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not found: put it on PATH or set recording.ffmpeg in the city config")
    return exe
