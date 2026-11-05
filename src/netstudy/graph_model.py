"""The graph model (docs/architecture-next.md section 5): one model of a town, fitted only on its own camera-derived
`edge_state`, plus the town config's signal plans (simulation; from video in a real town). No SUMO here.

  fit           per day kind: the usual state (10/50/90%) per road and 15 minutes, demand at the entry roads (in_*),
                turn shares per junction and hour (least squares, traffic in = out); per road: saturation flow and
                green share (signals), implicit capacity (the highest flows seen below free-flow speed), the speed
                against flow curve. Only clean slots: none within an incident_log entry (and an hour after it).
  expected_day  the usual state of the date's day kind (holiday / festival without recorded days: a weekend x
                days.HOLIDAY_FACTORS, marked assumed)
  road_change   a road set to busy / slow / jammed / closed for a while (section 5.3, option b), per 15 minutes:
                capacity cut, drivers on the road's routes move to the next-best routes by travel time (successive
                averages), what fits nowhere queues on the road leading in and is carried into the next slot until
                it clears. Predicted = usual + (model with the change - model without), per road.

Travel time per road: running time (BPR on its length and speed limit) + delay at its end. With a signal plan
(`signals=True`): Webster's uniform delay + the overflow term (Akçelik / HCM), capacity = saturation flow x green
share. Without: the implicit capacity and the overflow term only. Flows are veh/h inside, veh per 5 minutes out
(the unit of `observations`).
"""
from __future__ import annotations

import math
import pickle
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .days import HOLIDAY_FACTORS, JAMMED_KMH, LEVELS, day_kind

SLOT_H = 0.25                       # 15 minutes
SAT_PCU_LANE = 1800.0               # saturation flow per lane when the cameras never saw a full approach
SAT_RANGE = (1200.0, 2400.0)        # pcu/h/lane: a fitted value outside this is not believed
LOST_S = 2.0                        # lost time per green; effective green = green + amber - lost
YIELD_WEIGHT = 0.5                  # a movement that gives way uses half its green
BPR_A, BPR_B = 0.15, 4.0
JAM_GAP_M, JAM_TAU_S = 7.5, 1.0     # jammed: vehicle length + gap, plus 1 s headway -> the flow at 5 km/h
AFTER_INCIDENT = pd.Timedelta(minutes=60)
MAX_AFTER_H = 2.0                   # a change is followed for up to 2 hours after it ends
MSA_ITER = 40
MAX_PATH_EDGES = 7
FREE_FLOW_VPH = 100.0               # below this a road runs at its free-flow speed
FULL_VIEW = 0.8                    # a camera view at least 80% full: queued, discharging at saturation flow
LEVEL_CAPACITY = {"busy": 0.6, "slow": 0.35}


def webster_s(cycle_s: float, green_ratio: float, x: float) -> float:
    """Uniform delay per vehicle at a fixed-time signal (seconds)."""
    return 0.5 * cycle_s * (1 - green_ratio) ** 2 / max(1e-6, 1 - min(1.0, x) * green_ratio)


def overflow_s(x: float, capacity_vph: float, t_h: float = SLOT_H) -> float:
    """Random and overflow delay per vehicle (seconds), HCM's incremental term with k = 0.5: small below capacity,
    growing with the queue above it."""
    if capacity_vph <= 0:
        return math.inf
    return 900 * t_h * ((x - 1) + math.sqrt((x - 1) ** 2 + 4 * x / (capacity_vph * t_h)))


@dataclass
class GraphModel:
    town: str
    edges: pd.DataFrame                     # edge_id, from_node, to_node, kind, length_m, lanes, speed_kmh + fitted
    usual: pd.DataFrame                     # day_kind, slot, edge_id, {flow,speed}_{p10,p50,p90}  (flow veh/5 min)
    turns: pd.DataFrame                     # day_kind, hour, junction, from_edge, to_edge, share
    signals: dict                           # junction -> {cycle_s, phases} (town config), {} without
    days: list[str]                         # the days fitted on
    label_source: str
    speed_curves: dict = field(default_factory=dict)    # edge -> (flow vph bins, speed km/h), non-increasing
    paths: dict = field(default_factory=dict)           # entry edge -> list of paths (tuples of edge ids)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, sim: dict, out_dir: Path, until: date | None = None, town: str | None = None) -> "GraphModel":
        """Fit on the town's days before `until` (all days without)."""
        town = town or out_dir.name
        label_source = "sumo" if (sim.get("source") or {}).get("skip_render_read") else "people"
        with sqlite3.connect(out_dir / "traffic.db") as c:
            edges = pd.read_sql("SELECT edge_id, from_node, to_node, kind, length_m, lanes, speed_kmh, camera_id "
                                "FROM edges", c)
            st = pd.read_sql("SELECT edge_id, ts, total, pcu, flow, speed FROM edge_state", c)
        st["ts"] = pd.to_datetime(st.ts)
        st["day"] = (st.ts - pd.Timedelta(seconds=1)).dt.date
        if until is not None:
            st = st[st.day < until]
        st = _drop_incidents(st, out_dir / "labels.db", label_source)
        st["slot_start"] = (st.ts - pd.Timedelta(seconds=1)).dt.floor("15min")
        st["day_kind"] = st.day.map(lambda d: day_kind(d))
        q = (st.groupby(["edge_id", "slot_start", "day_kind"])
               .agg(flow=("flow", "mean"), speed=("speed", "mean"), n=("flow", "size")).reset_index())
        q = q[q.n >= 2]
        q["slot"] = q.slot_start.dt.strftime("%H:%M")
        q["hour"] = q.slot_start.dt.hour
        usual = _usual(q)
        signals = ((sim.get("network") or {}).get("town") or {}).get("signals") or {}
        m = cls(town, edges, usual, pd.DataFrame(), signals, sorted({str(d) for d in st.day}), label_source)
        m.turns = _turn_shares(q, edges)
        m.paths = _paths(edges)
        m._fit_roads(st, q)
        return m

    def _fit_roads(self, st: pd.DataFrame, q: pd.DataFrame):
        e = self.edges.set_index("edge_id")
        tot = st.groupby("edge_id")[["pcu", "total"]].sum()
        e["pcu_per_veh"] = (tot.pcu / tot.total.where(tot.total > 0)).reindex(e.index).fillna(1.0)
        # Free-flow speed: the 85th percentile at low flow (kpis' edge_free_flow includes stops at red lights).
        low = st[st.flow * 12 < FREE_FLOW_VPH]
        e["free_kmh"] = low.groupby("edge_id").speed.quantile(0.85).reindex(e.index).fillna(e.speed_kmh)
        e["queued_kmh"] = st.groupby("edge_id").speed.quantile(0.1).reindex(e.index).fillna(5.0)
        q["vph"] = q.flow * 12
        # Implicit capacity: the highest 15-minute flows seen below 80% of free-flow speed (or the highest at all).
        slow = q.merge(e[["free_kmh"]], left_on="edge_id", right_index=True)
        slow = slow[slow.speed < 0.8 * slow.free_kmh]
        hi_slow = slow.groupby("edge_id").vph.quantile(0.99)
        n_slow = slow.groupby("edge_id").size()
        hi_all = q.groupby("edge_id").vph.max().reindex(e.index)      # never saturated: at least what was seen
        e["capacity_implicit_vph"] = np.where(n_slow.reindex(e.index).fillna(0) >= 10,
                                              np.fmax(hi_slow.reindex(e.index), hi_all), hi_all)
        # Signals: green share of each approach, weighted by its turn shares; saturation flow from full views.
        shares = self.turns.groupby(["from_edge", "to_edge"]).share.mean()
        e["signal"], e["cycle_s"], e["green_ratio"] = False, np.nan, 1.0
        for j, plan in self.signals.items():
            cycle = sum(float(p["green_s"]) + float(p.get("amber_s", 0)) + float(p.get("all_red_s", 0))
                        for p in plan["phases"])
            for a in e.index[e.to_node == j]:
                outs = shares.loc[a] if a in shares.index.get_level_values(0) else pd.Series(dtype=float)
                tot = outs.sum() or 1.0
                lam = 0.0
                for p in plan["phases"]:
                    g = (float(p["green_s"]) + float(p.get("amber_s", 0)) - LOST_S) / cycle
                    for key, w in (("go", 1.0), ("yield", YIELD_WEIGHT)):
                        for mv in p.get(key, []):
                            fr, to = _movement_edges(j, mv)
                            if fr == a:
                                lam += g * w * float(outs.get(to, 0.0)) / tot
                e.loc[a, ["signal", "cycle_s", "green_ratio"]] = True, cycle, max(lam, 0.05)
        sat_default = SAT_PCU_LANE / e.pcu_per_veh
        view = st.merge(self.edges[["edge_id", "lanes"]], on="edge_id")
        full = view[view.total >= FULL_VIEW * view.lanes * 60 / 6.0]          # camera view: lanes x 60 m / 6 m
        n_full = full.groupby("edge_id").size().reindex(e.index).fillna(0)
        q_full = (full.flow * 12).groupby(full.edge_id).quantile(0.9).reindex(e.index)
        sat_fit = q_full / (e.lanes * e.green_ratio)
        ok = (n_full >= 30) & sat_fit.between(SAT_RANGE[0] / e.pcu_per_veh, SAT_RANGE[1] / e.pcu_per_veh)
        e["sat_vph_lane"] = np.where(ok, sat_fit, sat_default)
        e["sat_source"] = np.where(ok, "fitted", "assumed")
        e["capacity_signal_vph"] = e.sat_vph_lane * e.lanes * e.green_ratio
        self.edges = e.reset_index()
        self.__dict__.pop("_rows", None)
        for edge, g in q.groupby("edge_id"):
            g = g.dropna(subset=["speed"])
            if len(g) < 20:
                continue
            bins = np.unique(np.quantile(g.vph, np.linspace(0, 1, 11)))
            mids, speeds = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                s = g.speed[(g.vph >= lo) & (g.vph <= hi)]
                if len(s):
                    mids.append((lo + hi) / 2)
                    speeds.append(float(s.median()))
            self.speed_curves[edge] = (np.array(mids), np.minimum.accumulate(np.array(speeds)))

    # ------------------------------------------------------------------ expected day
    def kind_for(self, d: date) -> tuple[str, bool]:
        """The day kind to read and whether it is assumed (no recorded day of that kind)."""
        kind = day_kind(d)
        if kind in set(self.usual.day_kind):
            return kind, False
        return ("weekend" if "weekend" in set(self.usual.day_kind) else "working"), True

    def expected_day(self, d: date) -> pd.DataFrame:
        kind, assumed = self.kind_for(d)
        u = self.usual[self.usual.day_kind == kind].copy()
        if assumed and day_kind(d) in ("holiday", "festival"):
            f = u.slot.str[:2].astype(int).map(lambda h: HOLIDAY_FACTORS[h])
            for c in ("flow_p10", "flow_p50", "flow_p90"):
                u[c] = u[c] * f
        u["date"], u["assumed"] = d.isoformat(), assumed
        return u.drop(columns="day_kind").reset_index(drop=True)

    def expected_state(self, edge: str, d: date, hhmm: str) -> dict:
        u = self.expected_day(d)
        slot = _slot(hhmm)
        row = u[(u.edge_id == edge) & (u.slot == slot)]
        return row.iloc[0].to_dict() if len(row) else {}

    # ------------------------------------------------------------------ road change
    def road_change(self, edge: str, level: str, d: date, start: str, minutes: int, signals: bool = True,
                    also: dict[str, str] | None = None) -> dict:
        """A road set to a level for `minutes` from `start`; `also`: more roads changed at the same time ({edge:
        level}), drivers rerouting round all of them together."""
        changes = {edge: level, **{k: v for k, v in (also or {}).items() if k != edge}}
        for k, lv in changes.items():
            if lv not in LEVELS:
                raise ValueError(f"level {lv!r}: one of {', '.join(LEVELS)}")
            if k not in set(self.edges.edge_id):
                raise ValueError(f"no road {k!r} in {self.town}")
        kind, assumed = self.kind_for(d)
        t0 = datetime.combine(d, datetime.min.time()) + _td(_slot(start))
        t_end = datetime.combine(d, datetime.min.time()) + _td(start) + timedelta(minutes=minutes)
        e = self.edges.set_index("edge_id")
        usual = self.expected_day(d).set_index(["edge_id", "slot"])
        queue = {k: 0.0 for k in changes}                  # vehicles waiting at each changed road
        rows, cleared = [], None
        t = t0
        while t < t_end + timedelta(hours=MAX_AFTER_H):
            on = t < t_end
            slot, hour = t.strftime("%H:%M"), t.hour
            base_paths = self._path_flows(kind, slot, hour)
            base_v = _edge_flows(base_paths)
            base_cost = {k: self._cost(k, base_v.get(k, 0.0), None, signals) for k in e.index}
            cuts = changes if on else {}
            paths = _reassign(self, base_paths, cuts, queue, signals) if on else dict(base_paths)
            v = _edge_flows(paths)
            feeding: dict[str, set] = {}                   # road leading in -> the changed roads it feeds
            for c in changes:
                via = {p: f for p, f in paths.items() if c in p}
                want = sum(via.values())
                cap = self._capacity(c, cuts.get(c), signals)
                passed = min(want + queue[c] / SLOT_H, cap)
                queue[c] = max(0.0, (want + queue[c] / SLOT_H - passed) * SLOT_H)
                scale = passed / want if want > 0 else 0.0
                for p, f in via.items():                   # held at the road leading in: it, the road and beyond
                    i = p.index(c)
                    for k in p[max(0, i - 1):]:
                        v[k] = v.get(k, 0.0) + f * (scale - 1)
                    if i > 0:
                        feeding.setdefault(p[i - 1], set()).add(c)
            n_feed = {c: sum(c in f for f in feeding.values()) for c in changes}
            for k in e.index:
                u = usual.loc[(k, slot)] if (k, slot) in usual.index else None
                if u is None:
                    continue
                lv = cuts.get(k)
                b, n = base_v.get(k, 0.0), max(0.0, v.get(k, 0.0))
                flow = 0.0 if lv == "closed" else max(0.0, u.flow_p50 + (n - b) / 12)
                speed = self._speed(k, u.speed_p50, b, n)
                waiting = sum(queue[c] for c in feeding.get(k, ()))
                if lv:
                    speed = _level_speed(lv, e.loc[k], speed)
                elif waiting > 0:
                    speed = min(speed, float(e.loc[k, "queued_kmh"]))
                cost_new = self._cost(k, n, lv, signals)
                extra_h = (n * cost_new - b * base_cost[k]) / 3600 * SLOT_H if math.isfinite(cost_new) else np.nan
                if k in feeding:
                    extra_h = (extra_h if np.isfinite(extra_h) else 0.0) + sum(
                        queue[c] * SLOT_H / n_feed[c] for c in feeding[k])
                rows.append({"edge_id": k, "slot": slot, "ts": t.isoformat(timespec="minutes"), "change_on": on,
                             "flow_usual": u.flow_p50, "flow": flow, "flow_change": flow - u.flow_p50,
                             "flow_p10": u.flow_p10, "flow_p90": u.flow_p90,
                             "speed_usual": u.speed_p50, "speed": speed,
                             "speed_change": speed - u.speed_p50 if pd.notna(speed) else np.nan,
                             "speed_p10": u.speed_p10, "speed_p90": u.speed_p90,
                             "queue_veh": waiting + (queue[k] if changes.get(k) == "closed" else 0.0),
                             "extra_delay_veh_h": extra_h})
            if not on and sum(queue.values()) < 1 and cleared is None:
                cleared = t
            if cleared is not None and t >= cleared + timedelta(minutes=30):
                break
            t += timedelta(minutes=15)
        out = pd.DataFrame(rows)
        out["affected"] = ((out.flow < out.flow_p10) | (out.flow > out.flow_p90) |
                           (out.speed < out.speed_p10) | (out.speed > out.speed_p90)) & \
                          ((out.flow_change.abs() >= 1) | (out.speed_change.abs() >= 3))
        j = out.merge(self.edges[["edge_id", "to_node", "kind"]], on="edge_id")
        j = j[j.kind != "out"]
        junctions = (j.groupby(["to_node", "slot", "ts"]).agg(extra_delay_veh_h=("extra_delay_veh_h", "sum"),
                                                              affected=("affected", "any"))
                     .reset_index().rename(columns={"to_node": "junction"}))
        return {"edges": out, "junctions": junctions,
                "summary": {"edge": edge, "level": level, "also": {k: v for k, v in changes.items() if k != edge},
                            "date": d.isoformat(), "start": start, "minutes": minutes,
                            "day_kind": kind, "assumed": assumed, "signals": signals and bool(self.signals),
                            "max_queue_veh": float(out.queue_veh.max()),
                            "cleared": cleared.isoformat(timespec="minutes") if cleared else None,
                            "affected_edges": sorted(set(out.edge_id[out.affected]))}}

    # ------------------------------------------------------------------ pieces
    def _path_flows(self, kind: str, slot: str, hour: int) -> dict[tuple, float]:
        """Route flows (veh/h) in one slot: each entry's usual flow split over its routes by the turn shares."""
        dem = self.usual[(self.usual.day_kind == kind) & (self.usual.slot == slot)].set_index("edge_id").flow_p50 * 12
        tr = self.turns[(self.turns.day_kind == kind) & (self.turns.hour == hour)]
        if tr.empty:
            tr = self.turns[self.turns.day_kind == kind].groupby(["from_edge", "to_edge"]).share.mean().reset_index()
        share = {(a, b): s for a, b, s in tr[["from_edge", "to_edge", "share"]].itertuples(index=False)}
        flows = {}
        for entry, paths in self.paths.items():
            probs = np.array([math.prod(share.get((p[i], p[i + 1]), 0.0) for i in range(len(p) - 1)) for p in paths])
            if probs.sum() <= 0:
                continue
            for p, pr in zip(paths, probs / probs.sum()):
                flows[p] = float(dem.get(entry, 0.0)) * pr
        return flows

    def _row(self, edge: str):
        rows = self.__dict__.get("_rows")
        if rows is None:
            rows = self.__dict__["_rows"] = {r.edge_id: r for r in self.edges.itertuples(index=False)}
        return rows[edge]

    def _capacity(self, edge: str, level: str | None, signals: bool) -> float:
        r = self._row(edge)
        if signals and r.signal:
            cap = float(r.capacity_signal_vph)
        elif r.kind == "out":                     # leaves town, no junction at its end
            cap = float(r.sat_vph_lane * r.lanes)
        else:
            cap = float(r.capacity_implicit_vph)
        if level == "closed":
            return 0.0
        if level == "jammed":
            v = JAMMED_KMH / 3.6
            return min(cap, float(r.lanes) * 3600 * v / (JAM_GAP_M + JAM_TAU_S * v))
        return cap * LEVEL_CAPACITY.get(level, 1.0)

    def _cost(self, edge: str, vph: float, level: str | None, signals: bool) -> float:
        """Seconds to drive the road and get through its end, at `vph`."""
        r = self._row(edge)
        cap = self._capacity(edge, level, signals)
        if cap <= 0:
            return math.inf
        limit = float(r.speed_kmh) * (LEVELS[level] if level in ("busy", "slow") else 1.0)
        if level == "jammed":
            limit = JAMMED_KMH
        x = vph / cap
        run = float(r.length_m) / (limit / 3.6) * (1 + BPR_A * x ** BPR_B)
        delay = overflow_s(x, cap)
        if signals and r.signal:
            delay += webster_s(float(r.cycle_s), float(r.green_ratio), x)
        return run + delay

    def _speed(self, edge: str, usual: float, base_vph: float, new_vph: float) -> float:
        if pd.isna(usual) or edge not in self.speed_curves:
            return usual
        xs, ys = self.speed_curves[edge]
        f = lambda v: float(np.interp(v, xs, ys))  # noqa: E731
        b = f(base_vph)
        return usual * f(new_vph) / b if b > 0 else usual

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "GraphModel":
        return pickle.loads(path.read_bytes())


# ---------------------------------------------------------------------- helpers
def _movement_edges(junction: str, movement: str) -> tuple[str, str]:
    """As network.movement_edges (kept here so the graph model does not import the network builder)."""
    a, b = (s.strip() for s in movement.split(">"))
    return (f"in_{junction}" if a == "hood" else f"{a}_{junction}",
            f"out_{junction}" if b == "hood" else f"{junction}_{b}")


def _slot(hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    return f"{h:02d}:{m // 15 * 15:02d}"


def _td(hhmm: str) -> timedelta:
    h, m = map(int, hhmm.split(":"))
    return timedelta(hours=h, minutes=m)


def _drop_incidents(st: pd.DataFrame, labels_db: Path, label_source: str) -> pd.DataFrame:
    if not labels_db.exists():
        return st
    with sqlite3.connect(labels_db) as c:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name = 'incident_log'").fetchone():
            return st
        inc = pd.read_sql("SELECT start, end FROM incident_log WHERE label_source = ?", c, params=(label_source,))
    bad = np.zeros(len(st), dtype=bool)
    ts = st.ts.to_numpy()
    for s, e in zip(pd.to_datetime(inc.start), pd.to_datetime(inc.end)):
        bad |= (ts > np.datetime64(s.floor("15min"))) & (ts <= np.datetime64(e + AFTER_INCIDENT))
    return st[~bad]


def _usual(q: pd.DataFrame) -> pd.DataFrame:
    g = q.groupby(["day_kind", "slot", "edge_id"])
    out = pd.concat({f"{v}_p{int(p * 100)}": g[v].quantile(p) for v in ("flow", "speed") for p in (0.1, 0.5, 0.9)},
                    axis=1).reset_index()
    out["days"] = g.size().values
    return out


def _turn_shares(q: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Per day kind, hour and junction: shares p[in -> out] >= 0 with sum over outs = 1, least squares on the
    15-minute flows out = sum p x in. No U-turns."""
    junctions = edges.node_id if "node_id" in edges else sorted(set(edges.to_node) & set(edges.from_node))
    flows = q.pivot_table(index=["day_kind", "hour", "slot_start"], columns="edge_id", values="flow")
    rows = []
    for j in junctions:
        ins = edges[(edges.to_node == j)]
        outs = edges[(edges.from_node == j)]
        moves = [(a.edge_id, b.edge_id) for a in ins.itertuples() for b in outs.itertuples() if b.to_node != a.from_node]
        if not moves or len(ins) < 2:
            continue
        cols = list(ins.edge_id) + list(outs.edge_id)
        for (kind, hour), f in flows.groupby(level=[0, 1]):
            f = f.reindex(columns=cols).dropna()
            if len(f) < 4:
                continue
            a_rows, b = [], []
            for o in outs.edge_id:
                a_rows.append(np.column_stack([f[i].to_numpy() if o == mo else np.zeros(len(f)) for i, mo in moves]))
                b.append(f[o].to_numpy())
            w = 3.0 * float(np.sqrt(len(f))) * float(f[list(ins.edge_id)].to_numpy().mean() or 1.0)
            for i in ins.edge_id:                                   # sum over outs = 1, weighted strongly
                a_rows.append(np.array([[w if mi == i else 0.0 for mi, _ in moves]]))
                b.append(np.array([w]))
            res = lsq_linear(np.vstack(a_rows), np.concatenate(b), bounds=(0, 1))
            p = res.x
            for i in ins.edge_id:                                   # renormalise the small remainder
                ix = [k for k, (mi, _) in enumerate(moves) if mi == i]
                s = p[ix].sum()
                if s > 0:
                    p[ix] /= s
            rows += [{"day_kind": kind, "hour": hour, "junction": j, "from_edge": a, "to_edge": b_, "share": float(s)}
                     for (a, b_), s in zip(moves, p)]
    return pd.DataFrame(rows)


def _paths(edges: pd.DataFrame) -> dict[str, list[tuple]]:
    """Every simple route from each entry road (in_*) to an exit road (out_*), no U-turns."""
    succ = {r.edge_id: [s.edge_id for s in edges.itertuples() if s.from_node == r.to_node and s.to_node != r.from_node]
            for r in edges.itertuples()}
    kind = dict(zip(edges.edge_id, edges.kind))
    to_node = dict(zip(edges.edge_id, edges.to_node))
    from_node = dict(zip(edges.edge_id, edges.from_node))
    out = {}
    for entry in edges.edge_id[edges.kind == "in"]:
        found, stack = [], [((entry,), {from_node[entry], to_node[entry]})]
        while stack:
            p, seen = stack.pop()
            for n in succ[p[-1]]:
                if kind[n] == "out":
                    found.append(p + (n,))
                elif to_node[n] not in seen and len(p) < MAX_PATH_EDGES:
                    stack.append((p + (n,), seen | {to_node[n]}))
        out[entry] = found
    return out


def _edge_flows(paths: dict[tuple, float]) -> dict[str, float]:
    v: dict[str, float] = {}
    for p, f in paths.items():
        for k in p:
            v[k] = v.get(k, 0.0) + f
    return v


def _reassign(m: GraphModel, base: dict[tuple, float], cuts: dict[str, str], queue: dict[str, float],
              signals: bool) -> dict:
    """Drivers on routes through the changed roads (`cuts`: edge -> level) move to cheaper routes between the same
    entry and exit, by successive averages; the queue waiting at a road counts in its delay."""
    paths = dict(base)
    od: dict[tuple, list[tuple]] = {}
    for entry, ps in m.paths.items():
        for p in ps:
            od.setdefault((p[0], p[-1]), []).append(p)
    hit = [k for k, ps in od.items() if any(base.get(p, 0) > 0 and any(c in p for c in cuts) for p in ps)]
    for it in range(MSA_ITER):
        v = _edge_flows(paths)
        cost = {k: m._cost(k, v[k] + queue.get(k, 0.0) / SLOT_H * (k in cuts), cuts.get(k), signals) for k in v}
        step = 1.0 / (it + 2)
        for key in hit:
            ps = od[key]
            c = [sum(cost.get(k, m._cost(k, 0.0, cuts.get(k), signals)) for k in p) for p in ps]
            best = int(np.argmin(c))
            if not math.isfinite(c[best]):
                continue                                   # no way round: they wait
            for p, cp in zip(ps, c):
                if p is ps[best] or paths.get(p, 0) <= 0 or cp <= c[best] + 1.0:
                    continue
                moved = paths[p] * (step if math.isfinite(cp) else 1.0)     # a closed route: all of it
                paths[p] -= moved
                paths[ps[best]] = paths.get(ps[best], 0.0) + moved
    return paths


def _level_speed(level: str, r: pd.Series, speed: float) -> float:
    if level == "closed":
        return np.nan                                      # nothing passes the camera
    if level == "jammed":
        return JAMMED_KMH
    cap = float(r.speed_kmh) * LEVELS[level]
    return min(speed, cap) if pd.notna(speed) else cap


def model_path(out_dir: Path) -> Path:
    return out_dir / "predict" / "graph_model.pkl"
