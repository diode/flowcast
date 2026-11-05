"""Phase 3: network forecast vs v1 on the same cameras (docs/phase3.md).

  .venv\\Scripts\\python scripts\\check_forecast.py [--sim-config sim/maple_falls.yaml] [--workers 4] [--skip-incidents]

Main test: train on all but the last 7 days, test on them (v1's hold-out). Incident test: each
closure run of phase 2, forecast by models trained without its week, scored on the closed road and
its footprint from the closure's start to 60 minutes after its end. Writes forecast_eval to
traffic.db.
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from netstudy.cli import load_sim_config
from netstudy.forecast import HORIZONS, TARGETS, camera_series, fit_all, neighbours, predictions, scores, weeks, wide
from netstudy.graph import load_camera_series

AFTER_MIN = 60


def summary(ev: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = ev.groupby(by + ["model"]).agg(abs_err=("abs_err", "sum"), actual=("actual_sum", "sum"))
    return (g.abs_err / g.actual).unstack("model")[["v1", "network", "persistence", "same_time_yesterday"]]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="sim/maple_falls.yaml")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-incidents", action="store_true")
    args = p.parse_args(argv)
    _, out_dir = load_sim_config(args.sim_config)
    db, idb = out_dir / "traffic.db", out_dir / "incidents" / "incidents.db"
    edges, obs = load_camera_series(db, out_dir / "sumo_series.csv")
    kind = dict(zip(edges.edge_id, edges.kind))
    w, nb = wide(obs), neighbours(edges)
    series = {e: camera_series(e, w, nb[e]) for e in nb}
    index = next(iter(series.values())).index
    cutoff = index.max() - pd.Timedelta(days=7)

    rows = []

    def add(test, edge, target, h, p_):
        for r in scores(p_):
            rows.append({"test": test, "target": target, "minutes_ahead": 5 * h, "edge_id": edge,
                         "kind": kind[edge], **r, "wape": r["abs_err"] / r["actual_sum"] if r["actual_sum"] else None})

    print(f"main test: train until {cutoff}, test the 7 days after")
    holdout = fit_all(series, args.workers, train_until=cutoff)
    for e, s in series.items():
        for target in TARGETS:
            for h in HORIZONS:
                pr = predictions(holdout[e], s, target, h)
                add("holdout", e, target, h, pr[pr.index - pd.Timedelta(minutes=5 * h) > cutoff])

    if not args.skip_incidents:
        with sqlite3.connect(idb) as conn:
            incs = pd.read_sql("SELECT * FROM incident_list WHERE kind = 'closure'", conn)
            fp = pd.read_sql("SELECT incident_id, edge_id FROM incident_footprint WHERE kind = 'closure'", conn)
            runs = pd.read_sql("SELECT * FROM run_cameras", conn).merge(edges[["edge_id", "camera_id"]], on="camera_id")
        folds = weeks(index)
        models = {}
        for i, (lo, hi) in enumerate(folds):
            if hi >= index.max():
                models[i] = holdout               # last week: the hold-out models never saw it
            else:
                print(f"incident test: training without {lo:%d %b} - {hi:%d %b}")
                models[i] = fit_all(series, args.workers, train_until=None, hide=(lo, hi))
        for r in incs.itertuples():
            start, end = pd.Timestamp(r.start), pd.Timestamp(r.end)
            fold = next(i for i, (lo, hi) in enumerate(folds) if lo < start <= hi)
            rw = wide(runs[runs.incident_id == r.incident_id])
            day_w = {v: w[v].copy() for v in w}
            for v in day_w:
                day_w[v].loc[rw[v].index] = rw[v]               # that day as it went with the closure
            affected = sorted(set(fp.edge_id[fp.incident_id == r.incident_id]) | {r.edge_id})
            for e in affected:
                s = camera_series(e, day_w, nb[e])
                for target in TARGETS:
                    for h in HORIZONS:
                        pr = predictions(models[fold][e], s, target, h)
                        add("incident", e, target, h,
                            pr[(pr.index > start) & (pr.index <= end + pd.Timedelta(minutes=AFTER_MIN))])

    ev = pd.DataFrame(rows)
    with sqlite3.connect(db) as conn:
        ev.drop(columns=["abs_err", "actual_sum"]).to_sql("forecast_eval", conn, index=False, if_exists="replace")
    with pd.option_context("display.width", 200):
        for test in ev.test.unique():
            t = ev[ev.test == test]
            print(f"\n{test}: WAPE (total error / total actual, all cameras pooled)")
            print(summary(t, ["target", "minutes_ahead"]).round(3).to_string())
            print(f"\n{test}, 60 minutes ahead, by edge kind")
            print(summary(t[t.minutes_ahead == 60], ["target", "kind"]).round(3).to_string())
        h = ev[ev.test == "holdout"].pivot_table(index=["target", "minutes_ahead", "edge_id"], columns="model",
                                                 values="wape")
        better = (h.network < h.v1).groupby(level=[0, 1]).mean()
        print("\nholdout: share of cameras where the network model beats v1\n" + better.round(2).to_string())


if __name__ == "__main__":
    main()
