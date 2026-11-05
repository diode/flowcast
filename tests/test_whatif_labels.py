from pathlib import Path

import pytest

from netstudy.whatif_labels import case_id, make


def test_case_id():
    assert case_id({"edge": "J1_C", "level": "closed", "start": "08:00", "minutes": 60}) == "J1_C_closed_0800_60"


def test_refused_in_a_real_town(tmp_path: Path):
    with pytest.raises(SystemExit, match="simulation only"):
        make({"source": {"kind": "real"}}, tmp_path, [])
