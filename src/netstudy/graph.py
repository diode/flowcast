"""The road network as a directed graph, and the network state the system knows from its cameras.

Graph (from the SUMO network + cameras.json):
  nodes  junctions and boundary points (where a neighbourhood road meets the edge of town)
  edges  one row per road direction, with its camera (if any)

State builder: every slot, each edge gets its camera's measures (`observed = 1`). Edges whose camera
is hidden get estimates from the other cameras (`observed = 0`, method `M1`, see estimate.py); edges
with no camera at all stay empty for now.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .network import ensure_sumo
from .simulate import car_lanes

JAM_M_PER_PCU = 6.0          # queue space: one PCU per 6 m of lane (same rule as the cameras)

STATE_COLUMNS = ["edge_id", "ts", "total", "pcu", "occupancy", "load_ratio", "level", "flow", "speed",
                 "wait_s", "incident", "observed"]


def is_boundary(node) -> bool:
    """Where a road leaves the network: Maple Falls' F_* points, or a SUMO dead end (Westbrook, v1's grid)."""
    return node.getID().startswith("F_") or node.getType() == "dead_end"


def edge_kind(edge) -> str:
    """in: from a boundary point; out: to one; road: between junctions."""
    return "in" if is_boundary(edge.getFromNode()) else "out" if is_boundary(edge.getToNode()) else "road"


def build_graph(net_file: Path, cameras: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_sumo()
    import sumolib

    net = sumolib.net.readNet(str(net_file))
    nodes = pd.DataFrame([{
        "node_id": n.getID(),
        "kind": "boundary" if is_boundary(n) else "junction",
        "x": round(n.getCoord()[0], 1), "y": round(n.getCoord()[1], 1),
        "signal": int(n.getType() == "traffic_light"),
    } for n in net.getNodes()])
    cam_of = {c["edge"]: c["camera_id"] for c in cameras}
    edges = pd.DataFrame([{
        "edge_id": e.getID(),
        "from_node": e.getFromNode().getID(), "to_node": e.getToNode().getID(),
        "kind": edge_kind(e),
        "length_m": round(e.getLength(), 1), "lanes": car_lanes(e),
        "speed_kmh": round(e.getSpeed() * 3.6, 1),
        "capacity_pcu": round(car_lanes(e) * e.getLength() / JAM_M_PER_PCU, 1),
        "camera_id": cam_of.get(e.getID()),
    } for e in net.getEdges()])
    return nodes, edges


def save_graph(db_path: Path, json_path: Path, nodes: pd.DataFrame, edges: pd.DataFrame):
    with sqlite3.connect(db_path) as conn:
        nodes.to_sql("nodes", conn, if_exists="replace", index=False)
        edges.to_sql("edges", conn, if_exists="replace", index=False)
    json_path.write_text(json.dumps({"nodes": nodes.to_dict("records"), "edges": edges.to_dict("records")},
                                    indent=2), encoding="utf-8")


def load_edges(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT * FROM edges", conn)
    except Exception:
        raise SystemExit("no `edges` table; run `netstudy network` first")


def state_table(conn: sqlite3.Connection) -> str:
    """The per-edge state readers use: `edge_view` (from the cameras, build_view) for a camera city,
    `edge_truth` only for the old simulation-only study (Maple Falls)."""
    has = lambda t: conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (t,)).fetchone()  # noqa: E731
    return "edge_view" if has("edge_view") or not has("edge_truth") else "edge_truth"


def load_camera_series(db_path: Path, series_csv: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(edges, one row per camera edge per slot: camera measures; flow and speed from vision's observations,
    or for the old simulation-only study the loop flow from sumo_series.csv)."""
    with sqlite3.connect(db_path) as conn:
        edges = load_edges(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(observations)")}
        extra = [c for c in ("flow", "speed") if c in cols]
        obs = pd.read_sql(f"SELECT camera_id, ts, total, pcu, occupancy, load_ratio, level{''.join(', ' + c for c in extra)} "
                          "FROM observations", conn)
    if "flow" not in obs and series_csv is not None and series_csv.exists():
        flow = pd.read_csv(series_csv, usecols=["camera_id", "ts", "flow"])
        obs = obs.merge(flow, on=["camera_id", "ts"], how="left")
    for c in ("flow", "speed"):
        if c not in obs:
            obs[c] = None
    obs = obs.merge(edges[["edge_id", "camera_id"]], on="camera_id")
    return edges, obs


def build_state(db_path: Path, series_csv: Path | None = None, hidden: list[str] | None = None,
                scenario: str | None = None, view_m: float = 60) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """edge_state: one row per edge per slot. Camera edges copy their camera's measures (and the
    loop flow when there is one). Edges whose camera is `hidden` get the M1 estimate (flow, load,
    level, pcu); edges with no camera at all stay empty.
    Returns (edge_state, estimate_eval rows for the hidden edges or None)."""
    edges, obs = load_camera_series(db_path, series_csv)
    hidden = list(hidden or [])
    grid = edges[["edge_id"]].merge(pd.DataFrame({"ts": sorted(obs.ts.unique())}), how="cross")
    seen = obs[~obs.edge_id.isin(hidden)].drop(columns="camera_id")
    state = grid.merge(seen, on=["edge_id", "ts"], how="left")
    state["observed"] = state.edge_id.isin(seen.edge_id.unique()).astype(int)
    state["method"] = np.where(state.observed == 1, "camera", None)
    for col in ("wait_s", "incident"):                 # cameras don't measure these (yet)
        state[col] = None
    evaluation = None
    if hidden:
        from .estimate import evaluate, level_of

        wide = {c: obs.pivot(index="ts", columns="edge_id", values=c) for c in ("flow", "load_ratio")}
        evaluation, f_est, l_est = evaluate(wide["flow"], wide["load_ratio"], edges, hidden, scenario or "custom")
        est = f_est["M1"].stack().rename("flow").to_frame().join(l_est["M1"].stack().rename("load_ratio"))
        est = est.rename_axis(["ts", "edge_id"]).reset_index()
        e = edges.set_index("edge_id")
        cap = (e.lanes * np.minimum(view_m, e.length_m) / JAM_M_PER_PCU).rename("cap")
        est = est.join(cap, on="edge_id")
        est = est.assign(pcu=(est.load_ratio * est.cap).round(2), flow=est.flow.round(2),
                         load_ratio=est.load_ratio.round(4))
        est["level"] = level_of(est.load_ratio)
        state = state.set_index(["edge_id", "ts"])
        upd = est.set_index(["edge_id", "ts"])[["flow", "pcu", "load_ratio", "level"]]
        state.update(upd)
        state.loc[upd.index, "method"] = "M1"
        state = state.reset_index()
    state = state[STATE_COLUMNS + ["method"]].sort_values(["ts", "edge_id"])
    with sqlite3.connect(db_path) as conn:
        state.to_sql("edge_state", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS edge_state_ts ON edge_state (ts)")
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'edge_truth'").fetchone():
            build_view(conn, state, edges)
    return state, evaluation


def build_view(conn: sqlite3.Connection, state: pd.DataFrame, edges: pd.DataFrame, slot_min: int = 5):
    """edge_view: the camera state in the per-edge layout the KPIs, dashboard and Predict read
    (docs/architecture-next.md 1.1). Nothing in it comes from the simulator:
      veh_h        vehicles on the approach (the slot's snapshot) x the slot's length
      time_loss_s  delay from the measured speed: veh_h x (1 - speed / speed limit), in vehicle-seconds
      wait_s, incident   not measured yet (empty, 0)"""
    from .simulate import TRUTH_COLUMNS

    v = state.merge(edges[["edge_id", "speed_kmh"]], on="edge_id", how="left")
    total = pd.to_numeric(v.total, errors="coerce")
    speed = pd.to_numeric(v.speed, errors="coerce")
    v["veh_h"] = (total * slot_min / 60).round(4)
    v["time_loss_s"] = (v.veh_h * 3600 * (1 - speed / v.speed_kmh).clip(lower=0)).round(1)
    v["wait_s"] = np.nan
    v["incident"] = 0
    v[TRUTH_COLUMNS + ["method"]].to_sql("edge_view", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS edge_view_ts ON edge_view (ts)")


def save_evaluation(db_path: Path, evaluation: pd.DataFrame):
    """Replace this scenario's rows in estimate_eval."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS estimate_eval (scenario TEXT, method TEXT, edge_id TEXT, "
                     "determined INTEGER, flow_mae REAL, flow_wape REAL, flow_wape_1h REAL, load_mae REAL, level_acc REAL, "
                     "congested_slots INTEGER, congested_recall REAL)")
        for s in evaluation.scenario.unique():
            conn.execute("DELETE FROM estimate_eval WHERE scenario = ?", (s,))
        evaluation.to_sql("estimate_eval", conn, if_exists="append", index=False)
