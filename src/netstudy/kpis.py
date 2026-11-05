"""Network KPIs per slot, from the per-edge state (graph.state_table: edge_view from the cameras;
edge_truth only in the old simulation-only study):

  veh_in_town, pcu_in_town  vehicles / PCU on all edges at slot end
  share_congested           share of edge length moving below half its free-flow speed, with >= 5 vehicles
  speed_index               speed / free-flow speed, weighted by vehicle-hours (about 1 = as quiet as at night)
  delay_veh_h               vehicle-hours lost against driving at the allowed speed (stops, queues, crawling)
  stopped_veh_h             vehicle-hours standing still
  worst_edges               the 3 edges with the lowest speed / free-flow speed

Free-flow speed per edge: its median slot speed in the quiet hours 00:00-05:00, so it includes the
normal stops at red lights. It is kept in table `edge_free_flow`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import load_edges, state_table

QUIET_HOURS = range(0, 5)
CONGESTED_BELOW = 0.5        # speed / free-flow speed ...
QUEUE_MIN_VEH = 5            # ... with at least this many vehicles on the edge (one car at a red light is no jam)


def free_flow_speed(truth: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    """km/h per edge: median speed of its quiet-hour slots (speed limit if it never had traffic then)."""
    hour = (pd.to_datetime(truth.ts) - pd.Timedelta(minutes=5)).dt.hour
    quiet = truth[hour.isin(QUIET_HOURS) & truth.speed.notna()]
    free = quiet.groupby("edge_id").speed.median()
    return free.reindex(edges.edge_id).fillna(edges.set_index("edge_id").speed_kmh).rename("free_kmh")


def compute_kpis(truth: pd.DataFrame, edges: pd.DataFrame, free: pd.Series) -> pd.DataFrame:
    d = truth.merge(edges[["edge_id", "length_m"]], on="edge_id").join(free, on="edge_id")
    moving = d.speed.notna() & (d.veh_h > 0)
    d["rel"] = np.where(moving, d.speed / d.free_kmh, np.nan)
    d["cong_m"] = np.where((d.rel < CONGESTED_BELOW) & (d.total >= QUEUE_MIN_VEH), d.length_m, 0.0)
    d["rel_w"] = np.where(moving, d.rel * d.veh_h, 0.0)
    d["w"] = np.where(moving, d.veh_h, 0.0)
    g = d.groupby("ts")
    k = pd.DataFrame({
        "veh_in_town": g.total.sum(),
        "pcu_in_town": g.pcu.sum().round(1),
        "share_congested": (g.cong_m.sum() / g.length_m.sum()).round(4),
        "speed_index": (g.rel_w.sum() / g.w.sum().replace(0, np.nan)).round(3),
        "delay_veh_h": (g.time_loss_s.sum() / 3600).round(2),
        "stopped_veh_h": (g.wait_s.sum() / 3600).round(2),
        "incident_edges": g.incident.sum(),
    })
    top = d[moving].sort_values(["ts", "rel"]).groupby("ts").head(3)
    k["worst_edges"] = top.groupby("ts").apply(
        lambda t: ", ".join(f"{e}:{r:.2f}" for e, r in zip(t.edge_id, t.rel)), include_groups=False)
    return k.reset_index()


def build_kpis(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        edges = load_edges(conn)
        table = state_table(conn)
        truth = pd.read_sql("SELECT edge_id, ts, total, pcu, speed, wait_s, time_loss_s, veh_h, incident "
                            f"FROM {table}", conn)
        if truth.empty:
            raise SystemExit(f"{table} is empty; run `netstudy state` first")
        free = free_flow_speed(truth, edges)
        free.round(2).reset_index().rename(columns={"index": "edge_id"}).to_sql(
            "edge_free_flow", conn, if_exists="replace", index=False)
        k = compute_kpis(truth, edges, free)
        k.to_sql("network_kpi", conn, if_exists="replace", index=False)
    return k


def snapshot(db_path: Path, ts: str) -> tuple[pd.Series, pd.DataFrame]:
    """KPIs and every edge's true and camera state for the slot ending at `ts`."""
    with sqlite3.connect(db_path) as conn:
        k = pd.read_sql("SELECT * FROM network_kpi WHERE ts = ?", conn, params=[ts])
        if k.empty:
            raise SystemExit(f"no slot ending at {ts} (slots end on the 5-minute grid, e.g. 2026-10-06T18:00:00)")
        table = state_table(conn)
        e = pd.read_sql(f"""
            SELECT t.edge_id, g.kind, g.camera_id, t.total, t.pcu, t.load_ratio, t.level, t.flow, t.speed,
                   f.free_kmh, ROUND(t.speed / f.free_kmh, 2) AS speed_rel, ROUND(t.time_loss_s / 60, 1) AS loss_min,
                   t.incident, s.load_ratio AS cam_load, s.level AS cam_level
            FROM {table} t JOIN edges g USING (edge_id) JOIN edge_free_flow f USING (edge_id)
            LEFT JOIN edge_state s ON s.edge_id = t.edge_id AND s.ts = t.ts
            WHERE t.ts = ? ORDER BY speed_rel IS NULL, speed_rel""", conn, params=[ts])
    return k.iloc[0], e


def day_by_hour(db_path: Path, day: str) -> pd.DataFrame:
    start = pd.Timestamp(day)
    with sqlite3.connect(db_path) as conn:      # slots end at 00:05 ... 24:00 (= next day 00:00)
        k = pd.read_sql("SELECT * FROM network_kpi WHERE ts > ? AND ts <= ?", conn,
                        params=[start.isoformat(), (start + pd.Timedelta(days=1)).isoformat()])
    if k.empty:
        raise SystemExit(f"no KPIs for {day}")
    hour = (pd.to_datetime(k.ts) - pd.Timedelta(minutes=5)).dt.hour
    worst = k.worst_edges.str.split(":").str[0]
    out = k.groupby(hour).agg(veh_in_town=("veh_in_town", "mean"), pcu_in_town=("pcu_in_town", "mean"),
                              share_congested=("share_congested", "mean"), speed_index=("speed_index", "mean"),
                              delay_veh_h=("delay_veh_h", "sum"), stopped_veh_h=("stopped_veh_h", "sum"),
                              incident_slots=("incident_edges", lambda x: (x > 0).sum()))
    out["slowest_edge"] = worst.groupby(hour).agg(lambda x: x.mode().iat[0])
    out.index.name = "hour"
    return out.round(3)
