"""source.render_cameras: the rendered day films only some cameras (picked by the town's seed); the rest direct."""
from pathlib import Path

import pytest
import yaml

from netstudy.pipeline import render_cameras

TOWNS = Path(__file__).resolve().parents[1] / "towns"
CAMS = [{"camera_id": f"cam{i:02d}", "edge": f"e{i}"} for i in range(1, 33)]


def test_not_set_films_every_camera():
    assert render_cameras({"source": {}}, CAMS) is None


def test_random_pick_is_fixed_by_the_seed():
    sim = {"seed": 22, "source": {"render_cameras": 3}}
    a, b = render_cameras(sim, CAMS), render_cameras(sim, CAMS)
    assert a == b and len(set(a)) == 3 and set(a) <= {c["camera_id"] for c in CAMS}
    assert render_cameras({**sim, "seed": 23}, CAMS) != a


def test_list_and_refusals():
    assert render_cameras({"source": {"render_cameras": ["cam01", "cam05"]}}, CAMS) == ["cam01", "cam05"]
    with pytest.raises(SystemExit, match="no camera"):
        render_cameras({"source": {"render_cameras": ["cam99"]}}, CAMS)
    with pytest.raises(SystemExit, match="expected 1..32"):
        render_cameras({"source": {"render_cameras": 0}}, CAMS)


@pytest.mark.parametrize("town", ["brookhaven", "clearwater"])
def test_pending_towns_render_three_cameras(town):
    sim = yaml.safe_load((TOWNS / f"{town}.yaml").read_text(encoding="utf-8"))
    assert sim["source"]["render_days"] == 1 and sim["source"]["render_cameras"] == 3
