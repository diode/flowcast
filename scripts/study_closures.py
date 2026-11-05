"""Closure study for the city overview's "What our studies found": every road between two junctions closed for an
hour at each study period of a working day, on the graph model -> traffic.db study_closures. The overview ranks the
most critical roads and the quietest period for roadworks from it.

  .venv\\Scripts\\python scripts\\study_closures.py --sim-config towns/ashford.yaml

CPU only, about 1-4 s per closure (Clearwater: 42 roads x 4 periods, about 12 minutes).
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import date, datetime

import pandas as pd

from netstudy.assist import Tools, windows
from netstudy.cli import load_sim_config

DAY = date(2026, 11, 4)          # a working day (Wednesday)
MINUTES = 60


def period_times(sim: dict) -> dict[str, str]:
    """Each study period's time: an hour into the window (its busy part), the window's start for a short one."""
    out = {}
    for w in windows(sim):
        a, b = (int(x[:2]) * 60 + int(x[3:]) for x in (w["start"], w["end"]))
        m = a + 60 if b - a >= 90 else a
        out[w["period"]] = f"{m // 60:02d}:{m % 60:02d}"
    return out or {"morning": "09:00"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="towns/ashford.yaml")
    args = p.parse_args(argv)
    sim, out_dir = load_sim_config(args.sim_config)
    tools = Tools.load(out_dir.name, sim, out_dir)
    roads = [e for e in tools.edges.edge_id if not e.startswith(("in_", "out_"))]
    times = period_times(sim)
    rows, t0 = [], time.time()
    for period, hhmm in times.items():
        for e in roads:
            r = tools.model.road_change(e, "closed", DAY, hhmm, MINUTES)
            j = r["junctions"]
            g = j.groupby("junction").agg(x=("extra_delay_veh_h", "sum"), a=("affected", "any"))
            rows.append({"edge": e, "road": tools.places.road(e), "period": period, "time": hhmm,
                         "extra_delay_veh_h": round(float(g.x[g.a].clip(lower=0).sum()), 2),
                         "max_queue_veh": round(r["summary"]["max_queue_veh"], 1),
                         "roads_affected": len(r["summary"]["affected_edges"])})
        print(f"{period} {hhmm}: {len(roads)} closures, {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows).assign(day=DAY.isoformat(), minutes=MINUTES,
                                   studied_at=datetime.now().isoformat(timespec="seconds"))
    with sqlite3.connect(out_dir / "traffic.db") as c:
        df.to_sql("study_closures", c, if_exists="replace", index=False)
    top = df.sort_values("extra_delay_veh_h", ascending=False).head(3)
    print(top[["road", "period", "extra_delay_veh_h"]].to_string(index=False))


if __name__ == "__main__":
    main()
