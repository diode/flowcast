"""Phase 2 step 3: lagged correlation between connected edges over the 28-day run (docs/phase2.md).

  .venv\\Scripts\\python scripts\\check_influence.py [--sim-config sim/maple_falls.yaml]

Writes edge_influence (one row per connected pair) to traffic.db and prints it with the control:
the correlation between edges that share no junction.
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from netstudy.cli import load_sim_config
from netstudy.detect import counts
from netstudy.graph import load_camera_series
from netstudy.influence import influence


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="sim/maple_falls.yaml")
    args = p.parse_args(argv)
    _, out_dir = load_sim_config(args.sim_config)
    db = out_dir / "traffic.db"
    edges, obs = load_camera_series(db, out_dir / "sumo_series.csv")
    table, control = influence(counts(obs, 1), edges)
    with sqlite3.connect(db) as conn:
        table.to_sql("edge_influence", conn, index=False, if_exists="replace")
    lo, hi = control.median(), control.quantile(0.9)
    with pd.option_context("display.width", 200):
        print(f"control (edges sharing no junction, lag 0): median r {lo:.2f}, 90% below {hi:.2f}, {len(control)} pairs")
        print(f"peak lag: {table.lag_min.value_counts().to_dict()} (minutes: pairs); "
              f"pairs with r0 above the control's 90%: peak at 0 for "
              f"{(table[table.r0 > hi].lag_min == 0).mean():.0%}")
        print("\nby kind (r0):")
        print(table.groupby("kind").r0.describe()[["count", "min", "50%", "max"]].round(2).to_string())
        print("\nroad -> road through a junction:")
        print(table[table.kind == "road->road"].to_string(index=False))


if __name__ == "__main__":
    main()
