"""The sample towns' results, as sample-data/<town>.zip (downloaded from Google Drive, see README Setup).

A fresh clone has no data/ (it is ignored: recordings and runs are many GB). ensure() unpacks each town's
results (traffic.db, labels.db, network, graph model) into data/<town>/ once, so the dashboard and the tests
work straight after cloning. A town already in data/ is left alone.

  .venv\\Scripts\\python -m netstudy.sample_data          # unpack by hand (the dashboard and pytest do it too)
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLES = ROOT / "sample-data"


def ensure(root: Path = ROOT) -> list[str]:
    """Unpack every sample town that is not in data/ yet; returns the towns unpacked."""
    done = []
    for z in sorted((root / "sample-data").glob("*.zip")):
        if (root / "data" / z.stem / "traffic.db").exists():
            continue
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        done.append(z.stem)
    return done


if __name__ == "__main__":
    print("unpacked:", ", ".join(ensure()) or "nothing (already in data/)")
