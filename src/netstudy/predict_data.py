"""What the Predict views show (dashboard/predict_view.py), from the graph model (graph_model.py) through the same
tools Glimmer calls (assist.Tools): the expected state at a future date and time, the day's curve against the usual
range, and a road change's impact with its actions. No SUMO here (architecture-next 1.1).

Predict answers only inside the town's filmed windows (recording.windows, implementation-next step 4): outside them
the graph model has no camera flow of its own, so the time is refused with the covered periods. A town config
without windows films the whole day."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import assist
from .assist import Tools
from .cli import load_sim_config
from .dashboard_data import Network
from .days import day_kind
from .overview import all_kpis

KPI = ["veh_in_town", "share_congested", "speed_index", "delay_veh_h"]
TOWNS = Path(__file__).resolve().parents[2] / "towns"
ENGINES = ("graph", "sumo")
ASSISTS = ("glimmer", "muse_spark", "off")


def town_config(name: str) -> dict | None:
    f = TOWNS / f"{name}.yaml"
    return load_sim_config(f)[0] if f.exists() else None


def predict_settings(sim: dict | None) -> dict:
    """The town's `predict:` switches (architecture-next 1.6): engine graph (default) | sumo, assist glimmer
    (default) | muse_spark | off. engine: sumo is refused, for now always (SUMO only makes sample data)."""
    p = (sim or {}).get("predict") or {}
    engine, assist = str(p.get("engine", "graph")), str(p.get("assist", "glimmer"))
    if engine not in ENGINES:
        raise SystemExit(f"predict.engine: {engine!r}, expected one of {', '.join(ENGINES)}")
    if engine == "sumo":
        raise SystemExit("predict.engine: sumo is refused (disabled): SUMO only makes sample data, Predict answers "
                         "from the graph model. Set predict.engine: graph.")
    if assist not in ASSISTS:
        raise SystemExit(f"predict.assist: {assist!r}, expected one of {', '.join(ASSISTS)}")
    if assist == "muse_spark":
        raise SystemExit("predict.assist: muse_spark is not built yet (later); use glimmer or off")
    return {"engine": engine, "assist": assist}


def load_tools(net: Network) -> Tools | None:
    """The town's tools on its graph model (fitted once if not stored), or None without camera-derived state."""
    sim = town_config(net.name)
    if sim is None or not net.has("edge_state") or not len(net.q("SELECT 1 FROM edge_state LIMIT 1")):
        return None
    predict_settings(sim)
    return Tools.load(net.name, sim, net.dir)


# ------------------------------------------------------------------ filmed windows
def in_windows(sim: dict | None, hhmm: str) -> bool:
    return assist.in_windows(assist.windows(sim), hhmm)


def covered(sim: dict | None) -> str:
    return assist.covered(assist.windows(sim))


# ------------------------------------------------------------------ expected day
def usual_band(net: Network, d: date) -> pd.DataFrame:
    """KPIs per time of day over the studied days of the same type: median, 10th and 90th percentile."""
    k = all_kpis(net)
    weekend = d.weekday() >= 5 or day_kind(d) == "holiday"
    k = k[k.weekend == weekend]
    q = k.groupby("tod")[KPI].quantile([0.1, 0.5, 0.9]).unstack()
    q.columns = [f"{c}_{int(p * 100)}" for c, p in q.columns]
    return q


def _slot(hhmm: str) -> str:
    return f"{hhmm[:2]}:{int(hhmm[3:]) // 15 * 15:02d}"


def _state(net: Network, edge_id, flow, speed) -> pd.DataFrame:
    """state_at's layout from the graph model's flow (veh per 5 minutes) and speed; it has no time loss."""
    return net.state_from_truth(pd.DataFrame({"edge_id": list(edge_id), "total": list(flow), "flow": list(flow),
                                              "speed": list(speed), "time_loss_s": np.nan}))


def expected(net: Network, tools: Tools, d: date, hhmm: str) -> dict:
    """The graph model's expected state at the time without a road change. kpi: the usual KPIs then; band: the
    usual range of the day type; assumed: no recorded day of the date's kind (holiday / festival)."""
    band = usual_band(net, d)
    u = tools.model.expected_day(d)
    u = u[u.slot == _slot(hhmm)]
    state = _state(net, u.edge_id, u.flow_p50, u.speed_p50)
    kind, assumed = tools.model.kind_for(d)
    return {"state": state, "kpi": {c: band.at[hhmm, f"{c}_50"] for c in KPI} if hhmm in band.index else {},
            "band": band, "curve": band[[f"{c}_50" for c in KPI]].rename(columns=lambda c: c[:-3]),
            "kind": kind, "assumed": assumed, "days": len(tools.model.days)}


def day_actions(tools: Tools, exp: dict, hhmm: str, event: str | None = None) -> list[dict]:
    """Actions for the chosen time without a road change (rules, actions.for_day)."""
    from . import actions

    return actions.for_day(tools.places, tools.edges, tools.nodes, exp["state"], f"around {hhmm}", event=event)


# ------------------------------------------------------------------ road change
def case_result(net: Network, tools: Tools, d: date, hhmm: str, change: dict) -> dict:
    """A road change on the graph model: per junction and road impact, the slot of the largest extra delay, the state
    then (case and usual), and the actions. Same model call and rules as Glimmer's set_road / impact / actions."""
    also = dict(change.get("also") or ())                  # more roads changed at the same time
    r = tools.model.road_change(change["edge"], change["level"], d, hhmm, int(change["minutes"]), also=also)
    case = {**change, "also": also, "date": d.isoformat(), "time": hhmm, "result": r}
    e, j = r["edges"], r["junctions"]
    on = e[e.change_on]
    worst = on.groupby("slot").extra_delay_veh_h.sum().idxmax() if len(on) else _slot(hhmm)
    at = e[e.slot == worst]
    case_state = _state(net, at.edge_id, at.flow, at.speed)
    base_state = _state(net, at.edge_id, at.flow_usual, at.speed_usual)
    hit = j[j.affected]
    junction = (j.groupby("junction").agg(extra_delay_veh_h=("extra_delay_veh_h", "sum"), affected=("affected", "any"))
                .reset_index())
    span = hit.groupby("junction").ts.agg(["min", "max"])
    junction["from_ts"] = junction.junction.map(span["min"]).str[-5:]
    junction["to_ts"] = junction.junction.map(lambda x: _plus15(span["max"].get(x)))
    junction["minutes"] = junction.junction.map(hit.groupby("junction").size() * 15).fillna(0).astype(int)
    junction["affected"] = junction.affected.astype(int)
    per_edge = (e.assign(extra_delay_veh_min=e.extra_delay_veh_h * 60, extra_vehicles=e.flow_change * 3)
                .groupby("edge_id").agg(affected=("affected", "any"), extra_delay_veh_min=("extra_delay_veh_min", "sum"),
                                        extra_vehicles=("extra_vehicles", "sum")).reset_index())
    slots = j.assign(delay=j.extra_delay_veh_h * 60)[["junction", "slot", "delay"]]
    end = (pd.Timestamp(f"{d} {hhmm}") + pd.Timedelta(minutes=int(change["minutes"]))).strftime("%H:%M")
    s = r["summary"]
    return {"junction": junction, "edges": per_edge, "per_slot": e, "slots": slots, "worst": worst,
            "case_state": case_state,
            "base_state": base_state, "actions": tools.case_actions(case), "end": end, "summary": s,
            "over_at": (s["cleared"] or "")[-5:], "max_queue_veh": s["max_queue_veh"],
            "total_extra_h": float(junction.extra_delay_veh_h[junction.affected == 1].clip(lower=0).sum())}


# ------------------------------------------------------------------ badges
SOURCES = {"observed": "a camera watches the road", "estimated": "no camera: from the graph model's neighbours",
           "assumed": "no recorded day of this kind: a usual day scaled"}


def road_sources(net: Network, assumed: bool) -> pd.Series:
    """Per road where its numbers come from: observed (camera), estimated (no camera), assumed (a holiday or festival
    without recorded days, every road)."""
    e = net.edges().set_index("edge_id")
    if assumed:
        return pd.Series("assumed", index=e.index)
    return e.camera_id.notna().map({True: "observed", False: "estimated"})


def accuracy(net: Network, sim: dict | None, change: dict | None = None, hhmm: str | None = None) -> dict | None:
    """Method accuracy in simulation (scripts/check_predict.py against the paired SUMO runs, label_source sumo):
    the change's own test case when it is one, else the median over the road-change cases; without a change the
    expected-day score. None in a real town or before scoring."""
    if ((sim or {}).get("source") or {}).get("kind") != "simulation" or not net.has("predict_eval_summary"):
        return None
    s = net.q("SELECT * FROM predict_eval_summary")
    if change is None:
        d = s[s.kind == "expected_day"]
        return None if d.empty else {"what": "expected day", "wape": float(d.wape.iloc[0]),
                                     "in_range": float(d.in_range.iloc[0]), "case": None}
    rc = s[(s.kind == "road_change") & (s.variant == "signals")]
    if rc.empty:
        return None
    cid = f"{change['edge']}_{change['level']}_{hhmm.replace(':', '')}_{int(change['minutes'])}" if hhmm else None
    one = rc[rc.case == cid]
    if len(one):
        r = one.iloc[0]
        return {"what": "this test case", "case": cid, "wape": float(r.wape), "sign_ok": float(r.sign_ok),
                "hit": int(r.hit), "missed": int(r.missed), "false": int(r.false)}
    return {"what": f"median of {len(rc)} test cases", "case": None, "wape": float(rc.wape.median()),
            "sign_ok": float(rc.sign_ok.median())}


# ------------------------------------------------------------------ paired runs (technical view, simulation only)
def paired_cases(net: Network) -> pd.DataFrame:
    """The answer key's cases (labels.db whatif_cases, netstudy whatif-labels): one row per case, its test days."""
    db = net.dir / "labels.db"
    if not net.has("whatif_cases", db):
        return pd.DataFrame(columns=["case_id", "edge", "level", "start", "minutes", "days"])
    c = net.q("SELECT * FROM whatif_cases", db=db)
    return (c.groupby(["case_id", "edge", "level", "start", "minutes"], sort=False).day.apply(sorted)
            .rename("days").reset_index())


def paired_truth(out_dir: Path, case_id: str) -> pd.DataFrame:
    """SUMO's true change for a case: change run - base run per road and 15 minutes, mean over its test days
    (label_source sumo; scoring only, never fed to the model)."""
    import sqlite3

    with sqlite3.connect(out_dir / "labels.db") as c:
        wl = pd.read_sql("SELECT * FROM whatif_labels WHERE case_id = ? AND label_source = 'sumo'", c, params=(case_id,))
    wl["slot"] = (pd.to_datetime(wl.ts) - pd.Timedelta(seconds=1)).dt.floor("15min").dt.strftime("%H:%M")
    wl["d_flow"] = wl.flow_change - wl.flow_base
    wl["d_speed"] = wl.speed_change - wl.speed_base
    per_day = wl.groupby(["day", "edge_id", "slot"])[["d_flow", "d_speed"]].mean().reset_index()
    return per_day.groupby(["edge_id", "slot"]).agg(d_flow=("d_flow", "mean"), d_speed=("d_speed", "mean"),
                                                    d_flow_sd=("d_flow", "std"), days=("day", "nunique")).reset_index()


def paired_compare(net: Network, model, case: pd.Series) -> dict:
    """The graph model's change (fitted on the days before the test days, as check_predict.py) next to SUMO's
    paired runs: per road and 15 minutes from the start to 2 hours after the end; the stored scores of both variants."""
    preds = [model.road_change(case.edge, case.level, date.fromisoformat(d), case.start, int(case.minutes))["edges"]
             for d in case.days]
    p = pd.concat(preds).groupby(["edge_id", "slot"]).agg(p_flow=("flow_change", "mean"),
                                                          p_speed=("speed_change", "mean")).reset_index()
    start = pd.Timestamp(f"2000-01-01 {case.start}")
    first = start.floor("15min").strftime("%H:%M")
    last = (start + pd.Timedelta(minutes=int(case.minutes) + 120)).strftime("%H:%M")
    t = paired_truth(net.dir, case.case_id)
    x = t[(t.slot >= first) & (t.slot < last)].merge(p, on=["edge_id", "slot"], how="left").fillna({"p_flow": 0.0})
    scores = net.q("SELECT * FROM predict_eval_summary WHERE kind = 'road_change' AND \"case\" = ?",
                   (case.case_id,)) if net.has("predict_eval_summary") else pd.DataFrame()
    roads = net.q("SELECT edge_id, metric, predicted, observed, sign_ok FROM predict_eval WHERE kind = 'road_change' "
                  "AND \"case\" = ? AND variant = 'signals'", (case.case_id,)) if net.has("predict_eval") else pd.DataFrame()
    return {"slots": x, "scores": scores, "roads": roads,
            "end": (start + pd.Timedelta(minutes=int(case.minutes))).strftime("%H:%M")}


def _plus15(ts: str | None) -> str | None:
    return (pd.Timestamp(ts) + pd.Timedelta(minutes=15)).strftime("%H:%M") if isinstance(ts, str) else None


def change_curve(slots: pd.DataFrame, junctions: list[str]) -> pd.DataFrame:
    """Extra delay (veh-min per 15 min) per affected junction and slot, wide."""
    s = slots[slots.junction.isin(junctions)]
    return s.pivot_table(index="slot", columns="junction", values="delay", aggfunc="sum", sort=False).fillna(0)


def kpi_tiles(exp: dict, hhmm: str) -> dict:
    """now and usual range (p10, p90) per KPI at the time, in tile units."""
    b = exp["band"].loc[hhmm] if hhmm in exp["band"].index else None
    def conv(c, x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        if c == "speed_index":
            return max(0.0, 100 * (1 / x - 1)) if x > 0 else np.nan
        return 100 * x if c == "share_congested" else x
    out = {}
    for c in KPI:
        now = conv(c, exp["kpi"].get(c))
        lo, hi = (conv(c, b[f"{c}_10"]), conv(c, b[f"{c}_90"])) if b is not None else (np.nan, np.nan)
        out[c] = {"now": now, "usual": conv(c, b[f"{c}_50"]) if b is not None else np.nan,
                  "lo": min(lo, hi), "hi": max(lo, hi)}
    return out
