"""Phase 2 step 2: detect incidents from the cameras and score against the footprints (docs/phase2.md).

  .venv\\Scripts\\python scripts\\check_propagation.py [--sim-config sim/maple_falls.yaml] [--min-missing 5 10 15 20]

Needs `netstudy incidents` first. For each `min_missing` (a road counts as dead once that many vehicles
were expected while it stayed empty): closures found, origin right, spread found (against incident_footprint), detection delay; false events per
day (placebo runs and incident-free days of the 28-day run); the 28-day run's random incidents found.
Writes detect_eval (one row per setting), detect_events and detect_spread (at the default) to
incidents.db.
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from netstudy.cli import load_sim_config
from netstudy.detect import MIN_MISSING, Normal, counts, detect
from netstudy.graph import load_camera_series



def day_of(ts: pd.Series) -> pd.Series:
    return (pd.to_datetime(ts) - pd.Timedelta(seconds=1)).dt.date


def on_day(d: pd.DataFrame, day) -> pd.DataFrame:
    return d[(d.index - pd.Timedelta(seconds=1)).date == day]


def random_incidents(db) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        d = pd.read_sql("SELECT edge_id, ts FROM edge_truth WHERE incident = 1 ORDER BY edge_id, ts", conn)
    d["ts"] = pd.to_datetime(d.ts)
    d["new"] = (d.ts.diff() > pd.Timedelta(minutes=10)) | (d.edge_id != d.edge_id.shift())
    g = d.groupby(d.new.cumsum()).agg(edge_id=("edge_id", "first"), start=("ts", "min"), end=("ts", "max"))
    g["start"] -= pd.Timedelta(minutes=5)
    g["day"] = g.start.dt.date
    return g.reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="sim/maple_falls.yaml")
    p.add_argument("--min-missing", type=float, nargs="+", default=[5.0, 10.0, 15.0, 20.0])
    args = p.parse_args(argv)
    _, out_dir = load_sim_config(args.sim_config)
    db, idb = out_dir / "traffic.db", out_dir / "incidents" / "incidents.db"
    edges, obs = load_camera_series(db, out_dir / "sumo_series.csv")
    base, base5 = counts(obs), counts(obs, 1)
    with sqlite3.connect(idb) as conn:
        incs = pd.read_sql("SELECT * FROM incident_list", conn)
        fp = pd.read_sql("SELECT * FROM incident_footprint WHERE kind = 'closure' AND hop > 0", conn)
        runs = pd.read_sql("SELECT * FROM run_cameras", conn).merge(edges[["edge_id", "camera_id"]], on="camera_id")
    rnd = random_incidents(db)
    inc_days = set(rnd.day)

    normals = {}

    def test(day, c):
        if day not in normals:
            normals[day] = Normal(base, edges, skip_day=day)          # from the other days
        return on_day(c, day), normals[day]          # 5-minute counts of the day

    # Days to test: every closure / placebo run, and every day of the 28-day run.
    tests = []                                   # (label, kind, day, counts, normal)
    for r in incs.itertuples():
        day = pd.Timestamp(r.day).date()
        tests.append((r.incident_id, r.kind, day, *test(day, counts(runs[runs.incident_id == r.incident_id], 1))))
    for day in sorted(set(day_of(pd.Series(base.index.astype(str))))):
        tests.append((f"day {day}", "random" if day in inc_days else "clean", day, *test(day, base5)))

    rows, kept = [], None
    for min_missing in args.min_missing:
        found = []
        for label, kind, day, c, normal in tests:
            ev, sp = detect(c, normal, min_missing)
            ev["label"], ev["kind"] = label, kind
            sp["label"] = label
            found.append((label, kind, day, ev, sp))
        ev_all = pd.concat([f[3] for f in found], ignore_index=True)
        sp_all = pd.concat([f[4] for f in found], ignore_index=True)
        if min_missing == MIN_MISSING:
            kept, kept_sp = ev_all, sp_all
        res = {"min_missing": min_missing}
        # Closures: found = an event on the closed road starting during the closure
        cl = incs[incs.kind == "closure"]
        hit, first_right, delays, tp, n_det, n_fp, lag = 0, 0, [], 0, 0, 0, []
        for r in cl.itertuples():
            s, e = pd.Timestamp(r.start), pd.Timestamp(r.end)
            ev = ev_all[(ev_all.label == r.incident_id) & (ev_all.known > s) & (ev_all.start <= e)]
            own = ev[ev.origin == r.edge_id]
            if len(own):
                hit += 1
                delays.append((own.known.min() - s).total_seconds() / 60)
                first_right += int(ev.sort_values(["known", "start"]).origin.iloc[0] == r.edge_id)
                sp = sp_all[sp_all.event_id.isin(own.event_id) & (sp_all.label == r.incident_id)]
                det = set(sp.edge_id)
                truth_fp = fp[fp.incident_id == r.incident_id]
                truth = set(truth_fp.edge_id)
                both = sp.groupby("edge_id")["first"].min().to_frame().join(
                    truth_fp.set_index("edge_id").first_ts, how="inner").dropna()
                lag += ((pd.to_datetime(both["first"]) - pd.to_datetime(both.first_ts)).dt.total_seconds() / 60).tolist()
                tp, n_det, n_fp = tp + len(det & truth), n_det + len(det), n_fp + len(truth)
        res.update(closures=len(cl), found=hit, origin_first=first_right,
                   known_after_min=pd.Series(delays).median() if delays else None,
                   spread_precision=tp / n_det if n_det else None, spread_recall=tp / n_fp if n_fp else None,
                   spread_lag_min=pd.Series(lag).median() if lag else None)
        # False events: placebo runs and clean days of the 28-day run, 06-22h
        for kind in ("placebo", "clean"):
            n_days = sum(1 for f in found if f[1] == kind)
            res[f"false_per_day_{kind}"] = sum(len(f[3]) for f in found if f[1] == kind) / n_days
        # Random incidents of the 28-day run
        got = 0
        for r in rnd.itertuples():
            ev = ev_all[(ev_all.label == f"day {r.day}") & (ev_all.origin == r.edge_id)
                        & (ev_all.start > r.start) & (ev_all.start <= r.end + pd.Timedelta(minutes=15))]
            got += int(len(ev) > 0)
        res.update(random=len(rnd), random_found=got)
        rows.append(res)

    out = pd.DataFrame(rows)
    with sqlite3.connect(idb) as conn:
        out.to_sql("detect_eval", conn, index=False, if_exists="replace")
        if kept is not None:
            kept.assign(start=kept.start.astype(str), end=kept.end.astype(str), known=kept.known.astype(str)) \
                .to_sql("detect_events", conn, index=False, if_exists="replace")
            kept_sp.assign(first=kept_sp["first"].astype(str)) \
                .to_sql("detect_spread", conn, index=False, if_exists="replace")
    with pd.option_context("display.width", 220):
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
