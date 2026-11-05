"""Phase 1b done-when checks (docs/phase1b.md, section 6).

  .venv\\Scripts\\python scripts\\check_state.py [--sim-config sim/maple_falls.yaml]

  1. completeness: every edge has a row in every slot, in edge_truth and edge_state
  2. truth flow (whole edge) vs camera loop flow, per edge over all days
  3. junctions: vehicles in = vehicles out, per hour
  4. incidents: lower speed_index and more congestion than the same hour on incident-free days
  5. coverage: what share of the network's traffic the cameras see
Runs on any network with a stored simulation (sim/maple_falls.yaml).
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from netstudy.cli import load_sim_config


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="sim/maple_falls.yaml")
    args = p.parse_args(argv)
    _, out_dir = load_sim_config(args.sim_config)
    with sqlite3.connect(out_dir / "traffic.db") as conn:
        edges = pd.read_sql("SELECT * FROM edges", conn)
        nodes = pd.read_sql("SELECT * FROM nodes", conn)
        truth = pd.read_sql("SELECT edge_id, ts, flow, veh_h FROM edge_truth", conn)
        state = pd.read_sql("SELECT edge_id, ts, observed, flow FROM edge_state", conn)
        kpi = pd.read_sql("SELECT * FROM network_kpi", conn)

    print("1. Completeness")
    n_slots = truth.ts.nunique()
    for name, t in (("edge_truth", truth), ("edge_state", state)):
        print(f"   {name}: {len(t)} rows = {t.edge_id.nunique()} edges x {t.ts.nunique()} slots "
              f"(expected {len(edges)} x {n_slots}), duplicates {t.duplicated(['edge_id', 'ts']).sum()}")
    print(f"   observed in edge_state: {state.observed.mean():.0%}")

    print("\n2. Truth flow vs camera loop flow (all days, camera edges)")
    cam_edges = edges.edge_id[edges.camera_id.notna()]
    f = truth[truth.edge_id.isin(cam_edges)].groupby("edge_id").flow.sum().to_frame("truth") \
        .join(state[state.edge_id.isin(cam_edges)].groupby("edge_id").flow.sum().rename("camera"))
    f["ratio"] = f.truth / f.camera
    print(f"   ratio min {f.ratio.min():.3f} ({f.ratio.idxmin()}), max {f.ratio.max():.3f} ({f.ratio.idxmax()}), "
          f"all edges {f.truth.sum() / f.camera.sum():.4f}")

    print("\n3. Junctions: vehicles in vs out per hour")
    t = truth.merge(edges[["edge_id", "from_node", "to_node"]], on="edge_id")
    t["hour"] = (pd.to_datetime(t.ts) - pd.Timedelta(minutes=5)).dt.floor("h")
    into = t.groupby(["to_node", "hour"]).flow.sum()       # leaving an edge = entering its end junction
    out = t.groupby(["from_node", "hour"]).flow.sum()      # ~ entering the edges that start there
    rows = []
    for j in nodes.node_id[nodes.kind == "junction"]:
        d = pd.DataFrame({"in": into[j], "out": out[j]}).fillna(0)
        gap = (d["in"] - d["out"]).abs() / d["in"].clip(lower=1)
        busy = d["in"] >= 100
        rows.append({"junction": j, "veh_in_per_day": d["in"].sum() / d.index.normalize().nunique(),
                     "in/out_all": d["in"].sum() / d["out"].sum(), "median_hourly_gap": gap[busy].median(),
                     "p95_hourly_gap": gap[busy].quantile(0.95)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("   (out counts vehicles leaving the next edge, so part of each hour's in shows up in the next hour)")

    print("\n4. Incidents vs the same hour on incident-free days (same day type)")
    k = kpi.copy()
    start = pd.to_datetime(k.ts) - pd.Timedelta(minutes=5)
    k["hour"], k["weekend"], k["day"] = start.dt.hour, start.dt.weekday >= 5, start.dt.normalize()
    inc_days = set(k.day[k.incident_edges > 0])
    base = k[~k.day.isin(inc_days)].groupby(["weekend", "hour"])[["speed_index", "share_congested", "delay_veh_h"]].mean()
    inc = k[k.incident_edges > 0].join(base, on=["weekend", "hour"], rsuffix="_base")
    print(f"   incident slots: {len(inc)} on {len(inc_days)} days; incident-free days: {k.day.nunique() - len(inc_days)}")
    for col, worse_if_lower in (("speed_index", True), ("share_congested", False), ("delay_veh_h", False)):
        worse = inc[col] < inc[col + "_base"] if worse_if_lower else inc[col] > inc[col + "_base"]
        print(f"   {col:16s} incident {inc[col].mean():.3f}  vs  baseline {inc[col + '_base'].mean():.3f}  "
              f"(worse in {worse.mean():.0%} of slots)")

    print("\n   On the incident edge itself vs the same edge, hour and day type on days without an incident there")
    with sqlite3.connect(out_dir / "traffic.db") as conn:
        e = pd.read_sql("SELECT t.edge_id, t.ts, t.speed / f.free_kmh AS rel, t.time_loss_s, t.incident "
                        "FROM edge_truth t JOIN edge_free_flow f USING (edge_id)", conn)
    start = pd.to_datetime(e.ts) - pd.Timedelta(minutes=5)
    e["hour"], e["weekend"], e["day"] = start.dt.hour, start.dt.weekday >= 5, start.dt.normalize()
    hit = e[e.incident == 1]
    clean = e.merge(hit[["edge_id", "day"]].drop_duplicates(), how="left", indicator=True)
    clean = clean[clean._merge == "left_only"]
    base = clean.groupby(["edge_id", "weekend", "hour"])[["rel", "time_loss_s"]].mean()
    j = hit.join(base, on=["edge_id", "weekend", "hour"], rsuffix="_base")
    print(f"   {len(j)} edge-slots: speed/free flow {j.rel.mean():.2f} vs {j.rel_base.mean():.2f} "
          f"(lower in {(j.rel < j.rel_base).mean():.0%}); time loss {j.time_loss_s.mean() / 60:.1f} vs "
          f"{j.time_loss_s_base.mean() / 60:.1f} veh-min per slot (higher in {(j.time_loss_s > j.time_loss_s_base).mean():.0%})")

    print("\n5. Coverage: share of the network's traffic on camera edges (all days)")
    t = truth.merge(edges[["edge_id", "length_m", "camera_id"]], on="edge_id")
    t["veh_km"] = t.flow * t.length_m / 1000
    cam = t.camera_id.notna()
    on_cam = edges.camera_id.notna()
    print(f"   edges {on_cam.sum()} of {len(edges)}, length {edges.length_m[on_cam].sum() / edges.length_m.sum():.0%}, "
          f"vehicle-km {t.veh_km[cam].sum() / t.veh_km.sum():.0%}, vehicle-hours {t.veh_h[cam].sum() / t.veh_h.sum():.0%}")


if __name__ == "__main__":
    main()
