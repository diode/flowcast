"""Phase 5: what the network dashboard reads and draws (docs/phase5.md). Plain functions on the stored
results of the earlier phases, so they can be tested without Streamlit; dashboard/app.py only lays
them out.

A network is a folder `data/<name>/` whose `traffic.db` has studied slots (`network_kpi`). Its per-edge state is
`edge_view` (from the cameras) or, in the old simulation-only study, `edge_truth` (graph.state_table);
`incidents/incidents.db` and `whatif/whatif.db` belong to that old study.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .graph import state_table
from .network import ensure_sumo

DATA = Path(__file__).resolve().parents[2] / "data"
LEVELS = ["free", "moderate", "heavy", "jammed"]
# Status colours for the four levels (state, not identity); grey = no data for this edge.
LEVEL_COLOURS = {"free": "#2e9e5b", "moderate": "#e0b100", "heavy": "#e8710a", "jammed": "#c5221f"}
NO_DATA = "#b0b4ba"
SPEED_BINS = [0.8, 0.5, 0.25]          # speed / free flow: >= 0.8 free, >= 0.5 moderate, >= 0.25 heavy


NUMERIC = ("total", "pcu", "load_ratio", "flow", "speed", "time_loss_s", "free_kmh", "cam_load", "cam_flow")


def _numeric(s: pd.DataFrame) -> pd.DataFrame:
    """Measured columns as numbers: a camera without a speed in a slot gives NULL, which can make the whole
    column `object`."""
    for c in NUMERIC:
        if c in s:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    return s


# ------------------------------------------------------------------ networks
def networks(root: Path = DATA) -> list[str]:
    out = []
    for db in sorted(root.glob("*/traffic.db")):
        with sqlite3.connect(db) as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'network_kpi'").fetchone():
                out.append(db.parent.name)
    return out


class Network:
    def __init__(self, name: str, root: Path = DATA):
        self.name, self.dir = name, root / name
        self.db = self.dir / "traffic.db"
        self.incidents_db = self.dir / "incidents" / "incidents.db"
        self.whatif_db = self.dir / "whatif" / "whatif.db"
        with sqlite3.connect(self.db) as conn:           # time lookups scan by slot
            self.state = state_table(conn)
            for table in (self.state, "edge_state"):
                conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_ts ON {table}(ts)")

    def q(self, sql: str, params=(), db: Path | None = None) -> pd.DataFrame:
        with sqlite3.connect(db or self.db) as conn:
            return pd.read_sql(sql, conn, params=params)

    def has(self, table: str, db: Path | None = None) -> bool:
        db = db or self.db
        if not db.exists():
            return False
        with sqlite3.connect(db) as conn:
            return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is not None

    # ------------------------------------------------------------- graph
    def edges(self) -> pd.DataFrame:
        return self.q("SELECT e.*, f.free_kmh FROM edges e LEFT JOIN edge_free_flow f USING (edge_id)")

    def nodes(self) -> pd.DataFrame:
        return self.q("SELECT * FROM nodes")

    def shapes(self) -> dict[str, list[tuple[float, float]]]:
        net_file = self.dir / "network" / "net.net.xml"
        if net_file.exists():
            return _shapes(str(net_file))
        return _map_shapes(str(self.dir / "network" / "shapes.json"))

    # -------------------------------------------------------------- time
    def days(self) -> list[str]:
        ts = self.q("SELECT DISTINCT ts FROM network_kpi ORDER BY ts").ts
        return sorted({(pd.Timestamp(t) - pd.Timedelta(seconds=1)).date().isoformat() for t in ts})

    @staticmethod
    def slot(day: str, hhmm: str) -> str:
        """Slot ending at hhmm on `day` ("24:00" = midnight after it)."""
        h, m = map(int, hhmm.split(":"))
        return (pd.Timestamp(day) + pd.Timedelta(hours=h, minutes=m)).isoformat()

    # ------------------------------------------------------------- state
    def state_at(self, ts: str) -> pd.DataFrame:
        """Every edge at one slot: true state, what the camera system knows, speed vs free flow."""
        truth = self.q("SELECT edge_id, total, pcu, load_ratio, level, flow, speed, time_loss_s, incident "
                       f"FROM {self.state} WHERE ts = ?", (ts,))
        known = self.q("SELECT edge_id, load_ratio AS cam_load, level AS cam_level, flow AS cam_flow, observed, "
                       "method FROM edge_state WHERE ts = ?", (ts,))
        s = _numeric(self.edges().merge(truth, on="edge_id", how="left").merge(known, on="edge_id", how="left"))
        s["speed_rel"] = s.speed / s.free_kmh
        s["speed_level"] = s.speed_rel.map(speed_level)
        s["loss_min"] = s.time_loss_s / 60
        return s

    def state_from_truth(self, truth: pd.DataFrame) -> pd.DataFrame:
        """state_at's layout for edge_truth-like rows of one slot (a prediction run, a usual median)."""
        cols = [c for c in ("edge_id", "total", "pcu", "load_ratio", "level", "flow", "speed", "time_loss_s",
                            "incident") if c in truth]
        s = _numeric(self.edges().merge(truth[cols], on="edge_id", how="left"))
        s["speed_rel"] = s.speed / s.free_kmh
        s["speed_level"] = s.speed_rel.map(speed_level)
        s["loss_min"] = s.time_loss_s / 60
        return s

    def kpi_at(self, ts: str) -> pd.Series | None:
        k = self.q("SELECT * FROM network_kpi WHERE ts = ?", (ts,))
        return k.iloc[0] if len(k) else None

    def kpi_day(self, day: str) -> pd.DataFrame:
        k = self.q("SELECT * FROM network_kpi WHERE ts > ? AND ts <= ?", _day_range(day))
        k["ts"] = pd.to_datetime(k.ts)
        return k

    def edge_day(self, edge_id: str, day: str) -> pd.DataFrame:
        """One edge over a day: truth, camera, and the usual day (median of the other days of the
        same day type at the same time)."""
        cols = "ts, flow, speed, load_ratio"
        t = self.q(f"SELECT {cols} FROM {self.state} WHERE edge_id = ?", (edge_id,))
        t["ts"] = pd.to_datetime(t.ts)
        start = t.ts - pd.Timedelta(minutes=5)
        t["date"], t["tod"], t["weekend"] = start.dt.date, start.dt.strftime("%H:%M"), start.dt.weekday >= 5
        d = pd.Timestamp(day).date()
        today = t[t.date == d]
        usual = t[(t.date != d) & (t.weekend == (d.weekday() >= 5))].groupby("tod")[["flow", "speed", "load_ratio"]].median()
        out = today.set_index("tod")[["ts", "flow", "speed", "load_ratio"]].join(usual.add_prefix("usual_"))
        cam = self.q("SELECT ts, flow AS cam_flow, load_ratio AS cam_load FROM edge_state "
                     "WHERE edge_id = ? AND ts > ? AND ts <= ? AND observed = 1", (edge_id, *_day_range(day)))
        cam["ts"] = pd.to_datetime(cam.ts)
        return out.reset_index().merge(cam, on="ts", how="left")

    def incident_slots(self, day: str) -> pd.DataFrame:
        """(edge, from, to) of the random incidents on a day."""
        d = self.q(f"SELECT edge_id, ts FROM {self.state} WHERE incident = 1 AND ts > ? AND ts <= ? ORDER BY edge_id, ts",
                   _day_range(day))
        if d.empty:
            return pd.DataFrame(columns=["edge_id", "start", "end"])
        d["ts"] = pd.to_datetime(d.ts)
        new = (d.ts.diff() > pd.Timedelta(minutes=10)) | (d.edge_id != d.edge_id.shift())
        return d.groupby(new.cumsum()).agg(edge_id=("edge_id", "first"), start=("ts", "min"), end=("ts", "max")) \
            .assign(start=lambda x: x.start - pd.Timedelta(minutes=5)).reset_index(drop=True)

    # --------------------------------------------------------- incidents
    def closures(self) -> pd.DataFrame:
        if not self.has("incident_list", self.incidents_db):
            return pd.DataFrame()
        return self.q("SELECT * FROM incident_list WHERE kind = 'closure'", db=self.incidents_db)

    def footprint(self, incident_id: str) -> pd.DataFrame:
        return self.q("SELECT * FROM incident_footprint WHERE incident_id = ?", (incident_id,), self.incidents_db)

    def detections(self, incident_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.has("detect_events", self.incidents_db):
            return pd.DataFrame(), pd.DataFrame()
        ev = self.q("SELECT * FROM detect_events WHERE label = ?", (incident_id,), self.incidents_db)
        sp = self.q("SELECT * FROM detect_spread WHERE label = ?", (incident_id,), self.incidents_db)
        return ev, sp

    # ----------------------------------------------------------- what-if
    def scenarios(self) -> list[str]:
        if not self.has("scenario_impact"):
            return []
        return self.q("SELECT DISTINCT scenario FROM scenario_impact").scenario.tolist()

    def scenario_impact(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (self.q("SELECT * FROM scenario_impact WHERE scenario = ?", (name,)),
                self.q("SELECT * FROM scenario_edges WHERE scenario = ?", (name,)))

    def scenario_days(self, name: str) -> list[str]:
        return self.q("SELECT DISTINCT day FROM runs WHERE scenario = ? AND kind = 'scenario' ORDER BY day",
                      (name,), self.whatif_db).day.tolist()

    def scenario_state(self, name: str, day: str, hhmm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(base, scenario) state of every edge at the same time of the same day."""
        ts = self.slot(day, hhmm)
        base = self.state_at(ts)
        run = self.q("SELECT edge_id, total, load_ratio, level, flow, speed, time_loss_s FROM run_truth "
                     "WHERE run_id = ? AND ts = ?", (f"{name}@{day}", ts), self.whatif_db)
        sc = self.edges().merge(run, on="edge_id", how="left")
        sc["speed_rel"] = sc.speed / sc.free_kmh
        sc["speed_level"] = sc.speed_rel.map(speed_level)
        sc["loss_min"] = sc.time_loss_s / 60
        return base, sc

    # ---------------------------------------------------------- forecast
    def forecast_scores(self) -> pd.DataFrame:
        if not self.has("forecast_eval"):
            return pd.DataFrame()
        f = self.q("SELECT * FROM forecast_eval")
        f["abs_err"], f["actual"] = f.mae * f.n, f.mae * f.n / f.wape
        g = f.groupby(["test", "target", "minutes_ahead", "model"])[["abs_err", "actual"]].sum()
        return (g.abs_err / g.actual).rename("wape").reset_index()


def _day_range(day: str) -> tuple[str, str]:
    d = pd.Timestamp(day)
    return d.isoformat(), (d + pd.Timedelta(days=1)).isoformat()


def speed_level(rel: float) -> str | None:
    if rel is None or np.isnan(rel):
        return None
    return next((name for name, lo in zip(LEVELS, SPEED_BINS) if rel >= lo), "jammed")


SIDE_SHIFT = 0.008          # each direction drawn this share of the map's size off the road's centre


@lru_cache(maxsize=8)
def _shapes(net_file: str) -> dict[str, list[tuple[float, float]]]:
    """Each edge along its middle car lane, pushed sideways (to the side its lane is on) so that the
    two directions of a road show as two lines: real lane offsets (~1.6 m) vanish at map scale."""
    ensure_sumo()
    import sumolib

    net = sumolib.net.readNet(net_file)
    (x0, y0), (x1, y1) = net.getBBoxXY()
    shift = SIDE_SHIFT * float(np.hypot(x1 - x0, y1 - y0))
    out = {}
    for e in net.getEdges():
        lanes = [ln for ln in e.getLanes() if ln.allows("passenger")] or e.getLanes()
        pts = np.array(lanes[len(lanes) // 2].getShape(), dtype=float)
        a, b = np.array(e.getFromNode().getCoord()), np.array(e.getToNode().getCoord())
        along = b - a
        if len(pts) >= 2 and np.hypot(*along) > 0:
            offset = pts.mean(axis=0) - (a + b) / 2                          # which side the lane is on
            side = np.sign(along[0] * offset[1] - along[1] * offset[0]) or 1.0
            seg = np.diff(pts, axis=0)
            d = np.vstack([seg[:1], (seg[:-1] + seg[1:]) / 2, seg[-1:]])   # direction at each point
            n = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
            pts = pts + side * shift * n
        out[e.getID()] = [tuple(p) for p in pts]
    return out


@lru_cache(maxsize=8)
def _map_shapes(shapes_json: str) -> dict[str, list[tuple[float, float]]]:
    """A real town's edges (osm.py: the road's centre line, in driving order), each direction pushed to its side as
    _shapes does: right of the line, so the two directions of a road show as two lines."""
    import json

    raw = {k: np.asarray(v, dtype=float) for k, v in json.loads(Path(shapes_json).read_text(encoding="utf-8")).items()}
    if not raw:
        return {}
    allp = np.vstack(list(raw.values()))
    shift = SIDE_SHIFT * float(np.hypot(*(allp.max(axis=0) - allp.min(axis=0))))
    out = {}
    for eid, pts in raw.items():
        if len(pts) >= 2:
            seg = np.diff(pts, axis=0)
            d = np.vstack([seg[:1], (seg[:-1] + seg[1:]) / 2, seg[-1:]])
            n = np.column_stack([d[:, 1], -d[:, 0]]) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
            pts = pts + shift * n
        out[eid] = [tuple(p) for p in pts]
    return out


# ------------------------------------------------------------------ charts
def map_figure(net: Network, values: pd.DataFrame, colour: str, dashed: set[str] = frozenset(),
               title: str = "", colours: dict[str, str] = LEVEL_COLOURS, legend: list[str] = LEVELS,
               hover: list[str] = ("flow", "speed", "speed_rel", "load_ratio", "loss_min")) -> go.Figure:
    """The network with each edge coloured by `values[colour]` (a level name; missing = grey)."""
    shapes, nodes = net.shapes(), net.nodes()
    v = values.set_index("edge_id")
    fig = go.Figure()
    for edge_id, pts in shapes.items():
        if edge_id not in v.index:
            continue
        r = v.loc[edge_id]
        key = r.get(colour)
        col = colours.get(key, NO_DATA) if isinstance(key, str) else NO_DATA
        xs, ys = zip(*pts)
        text = "<br>".join([f"<b>{edge_id}</b> ({r.get('kind', '')}{', ' + r.camera_id if isinstance(r.get('camera_id'), str) else ''})",
                            f"{colour}: {key if isinstance(key, str) else 'no data'}"]
                           + [f"{h}: {r[h]:.2f}" for h in hover if h in r and pd.notna(r[h])])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=5, dash="dash" if edge_id in dashed else "solid"),
                                 hoverinfo="text", text=text, showlegend=False, name=edge_id))
    for name in legend:                                            # legend entries only
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=colours[name], width=5), name=name))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=NO_DATA, width=5), name="no data"))
    j = nodes[nodes.kind == "junction"]
    fig.add_trace(go.Scatter(x=j.x, y=j.y, mode="markers+text", text=j.node_id if len(j) <= 12 else None,
                             textposition="top center", marker=dict(size=9, color="#3c4043"), hoverinfo="text",
                             hovertext=j.node_id, showlegend=False))
    fig.update_layout(title=title, height=560, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.02), hoverlabel=dict(align="left"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


def delay_classes(per_edge: pd.DataFrame) -> pd.DataFrame:
    """What-if map classes from per-edge extra delay: more / less delay, only where affected."""
    out = per_edge[["edge_id"]].copy()
    d, a = per_edge.extra_delay_veh_min, per_edge.affected == 1
    out["change"] = np.select([a & (d >= 60), a & (d > 0), a & (d < 0)], ["much more delay", "more delay", "less delay"],
                              default="no clear change")
    return out.merge(per_edge, on="edge_id")


CHANGE_COLOURS = {"much more delay": "#c5221f", "more delay": "#e8710a", "less delay": "#1a73e8",
                  "no clear change": "#dadce0"}


def timeline_figure(k: pd.DataFrame, at: pd.Timestamp, incidents: pd.DataFrame) -> go.Figure:
    """The day's network KPIs, one panel each (different units, so no shared axis)."""
    from plotly.subplots import make_subplots

    rows = [("speed_index", "Speed index (1 = free flow)"), ("share_congested", "Share of length congested"),
            ("delay_veh_h", "Delay (vehicle-hours per 5 min)")]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=[t for _, t in rows])
    for i, (col, _) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=k.ts, y=k[col], mode="lines", line=dict(width=2, color="#1a73e8"), name=col,
                                 hovertemplate="%{x|%H:%M}  %{y:.3f}<extra></extra>"), row=i, col=1)
        fig.add_vline(x=at, line=dict(color="#3c4043", width=1, dash="dot"), row=i, col=1)
        for r in incidents.itertuples():
            fig.add_vrect(x0=r.start, x1=r.end, fillcolor="#e8710a", opacity=0.15, line_width=0, row=i, col=1)
    fig.update_layout(height=620, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def edge_figure(d: pd.DataFrame, at: pd.Timestamp) -> go.Figure:
    """One edge's day: flow, speed and load, today against its usual day, one panel each."""
    from plotly.subplots import make_subplots

    rows = [("flow", "Vehicles per 5 min", "cam_flow"), ("speed", "Speed (km/h)", None),
            ("load_ratio", "Load (PCU / capacity)", "cam_load")]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=[t for _, t, _ in rows])
    for i, (col, _, cam) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=d.ts, y=d[f"usual_{col}"], mode="lines", line=dict(width=2, color="#9aa0a6"),
                                 name="usual day", showlegend=i == 1), row=i, col=1)
        fig.add_trace(go.Scatter(x=d.ts, y=d[col], mode="lines", line=dict(width=2, color="#1a73e8"),
                                 name="this day (true)", showlegend=i == 1), row=i, col=1)
        if cam and cam in d and d[cam].notna().any():
            fig.add_trace(go.Scatter(x=d.ts, y=d[cam], mode="lines", line=dict(width=2, color="#e8710a", dash="dot"),
                                     name="camera", showlegend=i == 1), row=i, col=1)
        fig.add_vline(x=at, line=dict(color="#3c4043", width=1, dash="dot"), row=i, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08))
    return fig
