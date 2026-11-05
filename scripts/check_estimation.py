"""Estimation of hidden edges: every scenario x method (docs/estimation.md).

  .venv\\Scripts\\python scripts\\check_estimation.py [--sim-config sim/maple_falls.yaml]

Runs each scenario in state.scenarios, writes estimate_eval, prints per-scenario summaries and the
per-edge table. edge_state is left as with every camera visible.
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from netstudy.cli import load_sim_config
from netstudy.estimate import evaluate
from netstudy.graph import build_state, load_camera_series, save_evaluation


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="sim/maple_falls.yaml")
    args = p.parse_args(argv)
    sim, out_dir = load_sim_config(args.sim_config)
    db = out_dir / "traffic.db"
    edges, obs = load_camera_series(db, out_dir / "sumo_series.csv")
    wide = {c: obs.pivot(index="ts", columns="edge_id", values=c) for c in ("flow", "load_ratio")}
    with sqlite3.connect(db) as conn:                  # old layout (before flow_wape_1h)
        conn.execute("DROP TABLE IF EXISTS estimate_eval")
    frames = []
    for name, hidden in sim["state"]["scenarios"].items():
        ev, _, _ = evaluate(wide["flow"], wide["load_ratio"], edges, hidden, name)
        save_evaluation(db, ev)
        frames.append(ev)
    ev = pd.concat(frames, ignore_index=True)
    build_state(db, out_dir / "sumo_series.csv")       # back to every camera visible

    with pd.option_context("display.width", 200):
        print("Mean over hidden edges, on 15-minute means (flow_wape: error / total flow; flow_wape_1h: per hour;\n"
              "recall: congested 15-minute slots found, only for edges with enough of them)")
        s = ev.groupby(["scenario", "method"], sort=False).agg(
            edges=("edge_id", "count"), determined=("determined", "sum"), flow_mae=("flow_mae", "mean"),
            flow_wape=("flow_wape", "mean"), flow_wape_1h=("flow_wape_1h", "mean"), load_mae=("load_mae", "mean"),
            level_acc=("level_acc", "mean"), congested_slots=("congested_slots", "sum"),
            congested_recall=("congested_recall", "mean"))
        print(s.round(3).to_string())
        print("\nM1 per hidden edge")
        m1 = ev[ev.method == "M1"].drop(columns="method")
        print(m1.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
