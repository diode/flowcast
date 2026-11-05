"""The customer dashboard's widgets (dashboard/overview.py): headline numbers against the usual day,
and Apache ECharts options for the map, gauge, day profile, weekly heatmap and road ranking. Plain
functions on a dashboard_data.Network, so they can be tested without Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dashboard_data import LEVEL_COLOURS, LEVELS, NO_DATA, Network
from .places import Places, scenario_title

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
MAP_BG, MAP_INK = "#0b1626", "#cbd5e1"
ACCENT = "#0f766e"                                # sequential hue (teal), light -> dark
HEAT = ["#f0fdfa", "#99f6e4", "#2dd4bf", "#0d9488", "#115e59"]
LEVEL_LABEL = {"free": "Flowing", "moderate": "Busy", "heavy": "Slow", "jammed": "Jammed"}
EFFECT_SPEED = {"free": 70, "moderate": 40, "heavy": 18, "jammed": 6}    # px/s of the moving dots
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def places(net: Network) -> Places:
    return Places(net.name, net.nodes(), net.edges())


# ------------------------------------------------------------------ numbers
def all_kpis(net: Network) -> pd.DataFrame:
    k = net.q("SELECT ts, veh_in_town, share_congested, speed_index, delay_veh_h FROM network_kpi")
    k["ts"] = pd.to_datetime(k.ts)
    start = k.ts - pd.Timedelta(minutes=5)                           # a slot is named by its end
    k["date"], k["tod"] = start.dt.date.astype(str), k.ts.dt.strftime("%H:%M").replace("00:00", "24:00")
    k["weekday"], k["hour"] = start.dt.weekday, start.dt.hour
    k["weekend"] = k.weekday >= 5
    return k


def usual_day(k: pd.DataFrame, day: str) -> pd.DataFrame:
    """Median of the other days of the same type (weekday / weekend), per time of day."""
    weekend = pd.Timestamp(day).weekday() >= 5
    other = k[(k.date != day) & (k.weekend == weekend)]
    return other.groupby("tod")[["veh_in_town", "share_congested", "speed_index", "delay_veh_h"]].median()


def avg_speed_kmh(state: pd.DataFrame) -> float:
    """Vehicle-km over vehicle-hours on the network in the slot."""
    s = state.dropna(subset=["flow", "speed"])
    s = s[(s.flow > 0) & (s.speed > 0)]
    vkm = s.flow * s.length_m / 1000
    return float(vkm.sum() / (vkm / s.speed).sum()) if len(s) else float("nan")


def headline(net: Network, k: pd.DataFrame, state: pd.DataFrame, day: str, hhmm: str) -> dict:
    """The four tiles: now, the usual value at this time, and the last two hours for a sparkline.
    Extra travel time: how much longer trips take than on empty roads (1 / speed index - 1)."""
    today = k[k.date == day].sort_values("ts")
    upto = today[today.tod <= hhmm] if hhmm != "24:00" else today
    usual = usual_day(k, day)
    now = upto.iloc[-1] if len(upto) else None
    u = usual.loc[hhmm] if hhmm in usual.index else None
    spark = upto.tail(24)
    extra = lambda si: max(0.0, 100 * (1 / si - 1)) if pd.notna(si) and si > 0 else np.nan  # noqa: E731
    out = {
        "vehicles": dict(now=now.veh_in_town if now is not None else np.nan,
                         usual=u.veh_in_town if u is not None else np.nan, spark=spark.veh_in_town.tolist()),
        "travel": dict(now=extra(now.speed_index) if now is not None else np.nan,
                       usual=extra(u.speed_index) if u is not None else np.nan,
                       spark=[extra(x) for x in spark.speed_index]),
        "congested": dict(now=now.share_congested * 100 if now is not None else np.nan,
                          usual=u.share_congested * 100 if u is not None else np.nan,
                          spark=(spark.share_congested * 100).tolist()),
        "delay": dict(now=upto.delay_veh_h.sum(), usual=usual.loc[usual.index <= hhmm, "delay_veh_h"].sum(),
                      spark=upto.delay_veh_h.cumsum().tail(24).tolist()),
    }
    return out


def flow_score(speed_index: float) -> int:
    """0-100: the network's speed against free flow (speed index), capped at 100."""
    return int(round(100 * min(max(speed_index, 0.0), 1.0))) if pd.notna(speed_index) else 0


def score_word(score: int) -> tuple[str, str]:
    for lo, word, col in [(80, "Flowing well", LEVEL_COLOURS["free"]), (60, "Busy", LEVEL_COLOURS["moderate"]),
                          (40, "Slow going", LEVEL_COLOURS["heavy"])]:
        if score >= lo:
            return word, col
    return "Jammed", LEVEL_COLOURS["jammed"]


def road_levels(state: pd.DataFrame) -> dict[str, int]:
    """Roads in each level now (by speed against free flow), plus those without data."""
    lv = state.speed_level.value_counts()
    out = {name: int(lv.get(name, 0)) for name in LEVELS}
    out["no data"] = int(state.speed_level.isna().sum())
    return out


def slowest_roads(state: pd.DataFrame, pl: Places, n: int = 6) -> pd.DataFrame:
    s = state.dropna(subset=["speed_rel"]).nsmallest(n, "speed_rel").copy()
    s["road"] = s.edge_id.map(pl.short_road)
    s["pct"] = (100 * s.speed_rel.clip(upper=1.2)).round(0)
    return s[["edge_id", "road", "pct", "speed", "free_kmh", "speed_level", "flow"]]


def weekly_profile(k: pd.DataFrame) -> pd.DataFrame:
    """Mean delay (vehicle-hours per hour) by weekday and hour, over all days."""
    per_hour = k.groupby(["date", "weekday", "hour"]).delay_veh_h.sum().reset_index()
    return per_hour.groupby(["weekday", "hour"]).delay_veh_h.mean().reset_index()


def peak_hour(k: pd.DataFrame, day: str) -> tuple[int, float]:
    h = k[k.date == day].groupby("hour").delay_veh_h.sum()
    return (int(h.idxmax()), float(h.max())) if len(h) else (0, 0.0)


# ------------------------------------------------------------- highlights
def whatif_cards(net: Network) -> list[dict]:
    out = []
    for sc in net.scenarios():
        j, _ = net.scenario_impact(sc)
        hit = j[j.affected == 1]
        pl = places(net)
        title, sub = scenario_title(sc)
        out.append(dict(id=sc, title=title, sub=sub, extra_h=float(hit.extra_delay_veh_h.sum()),
                        districts=[f"{pl.node(r.junction)} {r.from_ts}–{r.to_ts}" for r in hit.itertuples()]))
    return out


def detection_summary(net: Network, min_missing: int = 20) -> dict | None:
    if not net.has("detect_eval", net.incidents_db):
        return None
    d = net.q("SELECT * FROM detect_eval WHERE min_missing = ?", (min_missing,), net.incidents_db)
    if d.empty:
        return None
    r = d.iloc[0]
    return dict(closures=int(r.closures), found=int(r.found), minutes=float(r.known_after_min),
                false_per_day=float(r.false_per_day_clean))


def forecast_accuracy(net: Network) -> dict | None:
    f = net.forecast_scores()
    if f.empty:
        return None
    f = f[(f.test == "holdout") & (f.target == "flow") & (f.model == "network")]
    return {int(r.minutes_ahead): 100 * (1 - r.wape) for r in f.itertuples()}


# ------------------------------------------------------------------ charts
def _box(shapes: dict, aspect: float) -> tuple[float, float, float, float]:
    pts = np.array([p for v in shapes.values() for p in v])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    pad = 0.06 * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - 2.2 * pad, y1 + pad         # room for the legend below
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:                                       # widen to the panel's shape (equal scale)
        cx, w = (x0 + x1) / 2, h * aspect
        x0, x1 = cx - w / 2, cx + w / 2
    else:
        cy, h = (y0 + y1) / 2, w / aspect
        y0, y1 = cy - h / 2, cy + h / 2
    return x0, x1, y0, y1


def map_option(net: Network, state: pd.DataFrame, pl: Places, aspect: float = 1.6, highlight: str | None = None,
               marks: list[dict] | None = None) -> dict:
    """Roads coloured by how fast traffic moves against free flow, with dots moving along each road at
    a speed matching its level; districts labelled."""
    s = state.set_index("edge_id")
    rows = {}
    for e in s.index:
        r = s.loc[e]
        lv = r.speed_level if isinstance(r.speed_level, str) else "no data"
        speed = f"{r.speed:.0f} km/h ({100 * r.speed_rel:.0f}% of free flow)" if pd.notna(r.speed_rel) else "no data"
        rows[e] = (lv, f"<b>{pl.road(e)}</b><br/>{LEVEL_LABEL.get(lv, 'No vehicles')} · {speed}"
                       f"<br/>{0 if pd.isna(r.flow) else r.flow:.0f} vehicles in 5 min", r.lanes)
    groups = {lv: (LEVEL_LABEL[lv], LEVEL_COLOURS[lv], EFFECT_SPEED[lv]) for lv in LEVELS}
    groups["no data"] = ("No vehicles", NO_DATA, 0)      # truth has no speed only on an empty road
    return lines_map(net, rows, groups, pl, aspect, highlight, marks)


CHANGE_GROUPS = {"changed": ("The road you set", "#a855f7", 0), "much more": ("Much more delay", "#c5221f", 10),
                 "more": ("More delay", "#e8710a", 25), "less": ("Less delay", "#1a73e8", 60),
                 "none": ("No clear change", "#334155", 0)}


def impact_map_option(net: Network, per_edge: pd.DataFrame, pl: Places, changed: str | set, aspect: float = 1.6,
                      highlight: str | None = None, marks: list[dict] | None = None) -> dict:
    """Each road by its extra delay from the change (predict.impact's per-edge table)."""
    e = per_edge.set_index("edge_id")
    changed = {changed} if isinstance(changed, str) else set(changed)       # one road, or two or three together
    lanes = net.edges().set_index("edge_id").lanes
    rows = {}
    for edge_id in lanes.index:
        if edge_id in changed:
            g = "changed"
        elif edge_id in e.index and e.at[edge_id, "affected"]:
            d = e.at[edge_id, "extra_delay_veh_min"]
            g = "much more" if d >= 60 else "more" if d > 0 else "less"
        else:
            g = "none"
        d = e.at[edge_id, "extra_delay_veh_min"] if edge_id in e.index else 0.0
        v = e.at[edge_id, "extra_vehicles"] if edge_id in e.index else 0.0
        rows[edge_id] = (g, f"<b>{pl.road(edge_id)}</b><br/>{CHANGE_GROUPS[g][0]}<br/>{d:+.0f} min lost by drivers, "
                            f"{v:+.0f} vehicles", lanes[edge_id])
    return lines_map(net, rows, CHANGE_GROUPS, pl, aspect, highlight, marks)


CASE_MARK = "#a855f7"


def case_marks(net: Network, marks: list[dict]) -> dict:
    """Numbered pins at the middle of the roads in `marks` [{edge, label, tip, case, on, here}]: `on` the case
    shown (filled), `here` at the time shown (else faded). Each pin's data carries its `case` (for clicks)."""
    shapes = net.shapes()
    data = []
    for m in marks:
        if m["edge"] not in shapes:
            continue
        pts = shapes[m["edge"]]
        a, b = pts[len(pts) // 2 - 1] if len(pts) > 1 else pts[0], pts[len(pts) // 2]
        data.append({"value": [round((a[0] + b[0]) / 2, 1), round((a[1] + b[1]) / 2, 1)], "name": m["label"],
                     "tip": m["tip"], "case": m["case"],
                     "symbolSize": 24 if m["on"] else 20,
                     "itemStyle": {"color": CASE_MARK if m["on"] else "#1f2937", "borderColor": CASE_MARK,
                                   "borderWidth": 2, "opacity": 1 if m["here"] else 0.45},
                     "label": {"color": "#ffffff" if m["on"] else "#e9d5ff"}})
    return {"type": "scatter", "name": "Your cases", "zlevel": 3, "z": 10, "symbol": "circle",
            "label": {"show": True, "formatter": "{b}", "position": "inside", "fontSize": 11, "fontWeight": 700},
            "itemStyle": {"color": CASE_MARK}, "data": data}


def lines_map(net: Network, rows: dict, groups: dict, pl: Places, aspect: float = 1.6,
              highlight: str | None = None, marks: list[dict] | None = None) -> dict:
    """The network on a dark panel: rows {edge: (group, tooltip, lanes)}, groups {group: (legend name,
    colour, speed of the moving dots in px/s, 0 = none)}. Each road's data carries its `edge` id (for
    clicks); `highlight` draws a glow under one road; `marks` pins numbered cases on their roads (case_marks)."""
    shapes = net.shapes()
    x0, x1, y0, y1 = _box({e: shapes[e] for e in rows if e in shapes}, aspect)
    data: dict[str, list] = {g: [] for g in groups}
    for e, (g, tip, lanes) in rows.items():
        if e in shapes:
            data[g].append({"coords": [[round(x, 1), round(y, 1)] for x, y in shapes[e]], "tip": tip, "edge": e,
                            "lineStyle": {"width": 2 + 1.5 * min(int(lanes), 3)}})
    series = []
    if highlight in shapes:
        series.append({"type": "lines", "name": "Selected road", "coordinateSystem": "cartesian2d", "polyline": True,
                       "zlevel": 1, "z": 1, "silent": True,
                       "data": [{"coords": [[round(x, 1), round(y, 1)] for x, y in shapes[highlight]]}],
                       "lineStyle": {"color": "#f0abfc", "width": 18, "opacity": 0.85, "cap": "round"}})
    for g, items in data.items():
        if not items:
            continue
        name, col, speed = groups[g]
        series.append({"type": "lines", "name": name, "coordinateSystem": "cartesian2d",
                       "polyline": True, "data": items, "zlevel": 1,
                       "lineStyle": {"color": col, "opacity": 0.9, "cap": "round"},
                       "effect": {"show": speed > 0, "constantSpeed": speed,
                                  "symbol": "circle", "symbolSize": 4, "trailLength": 0.25, "color": "#ffffff"},
                       "emphasis": {"lineStyle": {"width": 9}}})
    hit = [item for items in data.values() for item in items]       # wide, invisible: easy to hover and click
    series.append({"type": "lines", "name": "_roads", "coordinateSystem": "cartesian2d", "polyline": True,
                   "zlevel": 1, "z": 5, "data": [{**it, "lineStyle": {}} for it in hit],
                   "lineStyle": {"color": "rgba(255,255,255,0.01)", "width": 10, "cap": "round"},
                   "emphasis": {"lineStyle": {"color": "rgba(255,255,255,0.35)", "width": 10}}})
    nodes = net.nodes()
    j = nodes[nodes.kind == "junction"]
    labels = len(j) <= 12
    series.append({"type": "scatter", "name": "Junctions", "zlevel": 2, "symbolSize": 11 if labels else 7,
                   "itemStyle": {"color": "#f8fafc", "borderColor": MAP_BG, "borderWidth": 2},
                   "label": {"show": labels, "formatter": "{b}", "position": "top", "color": "#f8fafc",
                             "fontSize": 12, "fontWeight": 600, "textBorderColor": MAP_BG, "textBorderWidth": 3},
                   "data": [{"name": pl.label(r.node_id), "value": [r.x, r.y],
                             "tip": f"<b>{pl.node(r.node_id)}</b>"} for r in j.itertuples()]})
    if marks:
        series.append(case_marks(net, marks))
    axis = {"show": False, "type": "value"}
    return {
        "backgroundColor": MAP_BG,
        "tooltip": {"trigger": "item", "confine": True, "formatter": JS_TIP, "backgroundColor": "#111827",
                    "borderColor": "#374151", "textStyle": {"color": "#f9fafb"}},
        "legend": {"bottom": 8, "left": "center", "textStyle": {"color": MAP_INK}, "icon": "roundRect",
                   "itemWidth": 18, "itemHeight": 6,
                   "data": [g["name"] for g in series if g["type"] == "lines" and not g["name"].startswith("_")]},
        "grid": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "xAxis": {**axis, "min": x0, "max": x1}, "yAxis": {**axis, "min": y0, "max": y1},
        "series": series,
    }


JS_TIP = "function (p) { return (p.data && p.data.tip) || p.name; }"


def gauge_option(score: int) -> dict:
    word, col = score_word(score)
    return {"series": [{
        "type": "gauge", "startAngle": 210, "endAngle": -30, "min": 0, "max": 100, "radius": "92%",
        "center": ["50%", "58%"],
        "progress": {"show": True, "width": 14, "roundCap": True, "itemStyle": {"color": col}},
        "axisLine": {"roundCap": True, "lineStyle": {"width": 14, "color": [[1, GRID]]}},
        "axisTick": {"show": False}, "splitLine": {"show": False},
        "axisLabel": {"show": False}, "pointer": {"show": False}, "anchor": {"show": False},
        "title": {"offsetCenter": [0, "38%"], "fontSize": 14, "color": MUTED},
        "detail": {"valueAnimation": True, "offsetCenter": [0, "0%"], "fontSize": 44, "fontWeight": 700,
                   "color": INK, "formatter": "{value}"},
        "data": [{"value": score, "name": word}],
    }]}


def donut_option(levels: dict[str, int]) -> dict:
    data = [{"name": LEVEL_LABEL[k], "value": v, "itemStyle": {"color": LEVEL_COLOURS[k]}}
            for k, v in levels.items() if k in LEVEL_LABEL and v]
    if levels.get("no data"):
        data.append({"name": "No data", "value": levels["no data"], "itemStyle": {"color": NO_DATA}})
    return {
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} roads ({d}%)"},
        "legend": {"orient": "vertical", "right": 4, "top": "middle", "icon": "circle", "textStyle": {"color": INK}},
        "series": [{"type": "pie", "radius": ["55%", "80%"], "center": ["36%", "50%"], "avoidLabelOverlap": True,
                    "itemStyle": {"borderColor": "#ffffff", "borderWidth": 2, "borderRadius": 4},
                    "label": {"show": True, "position": "center", "formatter": f"{sum(levels.values())}\nroads",
                              "fontSize": 16, "fontWeight": 600, "color": INK, "lineHeight": 20},
                    "labelLine": {"show": False}, "data": data}],
    }


def rhythm_option(k: pd.DataFrame, day: str, hhmm: str) -> dict:
    """Vehicles on the network through the day, against the usual day of this type."""
    today = k[k.date == day].sort_values("tod")
    usual = usual_day(k, day)
    x = today.tod.tolist()
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"}},
        "legend": {"top": 0, "left": 0, "icon": "roundRect", "itemHeight": 4, "textStyle": {"color": MUTED}},
        "grid": {"left": 44, "right": 16, "top": 40, "bottom": 28},
        "xAxis": {"type": "category", "data": x, "boundaryGap": False, "axisLine": {"lineStyle": {"color": GRID}},
                  "axisTick": {"show": False},
                  "axisLabel": {"color": MUTED, "interval": JS_EVERY_3H}},
        "yAxis": {"type": "value", "splitLine": {"lineStyle": {"color": GRID}}, "axisLabel": {"color": MUTED}},
        "series": [
            {"name": "Usual day", "type": "line", "data": [round(float(usual.veh_in_town.get(t, np.nan)), 1) for t in x],
             "showSymbol": False, "lineStyle": {"color": "#9ca3af", "width": 2, "type": "dashed"}, "itemStyle": {"color": "#9ca3af"}},
            {"name": "This day", "type": "line", "data": today.veh_in_town.round(1).tolist(), "showSymbol": False,
             "smooth": 0.2, "lineStyle": {"color": ACCENT, "width": 2}, "itemStyle": {"color": ACCENT},
             "areaStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                                     "colorStops": [{"offset": 0, "color": "rgba(15,118,110,0.28)"},
                                                    {"offset": 1, "color": "rgba(15,118,110,0.02)"}]}},
             "markLine": {"symbol": "none", "label": {"formatter": hhmm, "color": INK},
                          "lineStyle": {"color": INK, "type": "dotted"}, "data": [{"xAxis": hhmm}]}},
        ],
    }


JS_EVERY_3H = "function (i, v) { return v.slice(3) === '00' && parseInt(v, 10) % 3 === 0; }"


def heatmap_option(w: pd.DataFrame) -> dict:
    data = [[int(r.hour), int(r.weekday), round(float(r.delay_veh_h), 2)] for r in w.itertuples()]
    top = float(w.delay_veh_h.max()) if len(w) else 1.0
    return {
        "tooltip": {"position": "top", "formatter": JS_HEAT},
        "grid": {"left": 44, "right": 12, "top": 8, "bottom": 64},
        "xAxis": {"type": "category", "data": [f"{h:02d}" for h in range(24)], "splitArea": {"show": False},
                  "axisTick": {"show": False}, "axisLine": {"show": False}, "axisLabel": {"color": MUTED}},
        "yAxis": {"type": "category", "data": DAYS, "inverse": True, "axisTick": {"show": False},
                  "axisLine": {"show": False}, "axisLabel": {"color": MUTED}},
        "visualMap": {"min": 0, "max": round(top, 1), "calculable": False, "orient": "horizontal", "left": "center",
                      "bottom": 4, "itemHeight": 160, "itemWidth": 10, "inRange": {"color": HEAT},
                      "text": ["more time lost", "less"], "textStyle": {"color": MUTED}},
        "series": [{"type": "heatmap", "data": data, "itemStyle": {"borderColor": "#ffffff", "borderWidth": 2, "borderRadius": 3},
                    "emphasis": {"itemStyle": {"borderColor": INK, "borderWidth": 1}}}],
    }


JS_HEAT = ("function (p) { var d = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];"
           " return d[p.value[1]] + ' ' + p.value[0] + ':00<br/><b>' + p.value[2].toFixed(1) + ' h</b> lost by drivers'; }")


def ranking_option(slow: pd.DataFrame) -> dict:
    s = slow.iloc[::-1]                                       # slowest at the top
    return {
        "tooltip": {"trigger": "item", "formatter": "{b}<br/><b>{c}%</b> of free-flow speed"},
        "grid": {"left": 230, "right": 44, "top": 4, "bottom": 4},          # room for the street names
        "xAxis": {"type": "value", "max": 100, "show": False},
        "yAxis": {"type": "category", "data": s.road.tolist(), "axisTick": {"show": False},
                  "axisLine": {"show": False},
                  "axisLabel": {"color": INK, "width": 220, "overflow": "truncate", "ellipsis": "…"}},
        "series": [{"type": "bar", "barWidth": 12, "showBackground": True,
                    "backgroundStyle": {"color": "#f3f4f6", "borderRadius": 6},
                    "label": {"show": True, "position": "right", "formatter": "{c}%", "color": INK},
                    "data": [{"value": float(r.pct), "itemStyle": {"color": LEVEL_COLOURS.get(r.speed_level, NO_DATA),
                                                                     "borderRadius": 6}}
                             for r in s.itertuples()]}],
    }


# ------------------------------------------------------------------ the study towns' findings
def camera_accuracy(net: Network) -> dict | None:
    """Vision against a perfect camera on the rendered day (direct_check_summary, all cameras together)."""
    if not net.has("direct_check_summary"):
        return None
    d = net.q("SELECT * FROM direct_check_summary WHERE camera_id = 'all' ORDER BY checked_at DESC LIMIT 1")
    if d.empty:
        return None
    r = d.iloc[0]
    return {"flow_wape": float(r.flow_wape), "speed_mae": float(r.speed_mae_kmh), "day": r.day, "slots": int(r.slots)}


def predict_accuracy(net: Network) -> dict | None:
    """The graph model against SUMO's answer key (predict_eval_summary): the usual traffic of the test days inside
    the predicted range, and the roads the paired runs' road changes really affected that the model found."""
    if not net.has("predict_eval_summary"):
        return None
    e = net.q("SELECT * FROM predict_eval_summary")
    day = e[e.kind == "expected_day"]
    rc = e[(e.kind == "road_change") & (e.variant == "signals")]
    out = {}
    if len(day) and pd.notna(day.in_range.iloc[0]):
        out["in_range"] = float(day.in_range.iloc[0])
    if len(rc):
        out.update(cases=len(rc), found=int(rc.hit.sum()), affected=int(rc.hit.sum() + rc.missed.sum()),
                   false=int(rc["false"].sum()))
    return out or None


def closure_study(net: Network) -> pd.DataFrame | None:
    """scripts/study_closures.py: every road closed for an hour at each study period of a working day."""
    if not net.has("study_closures"):
        return None
    s = net.q("SELECT * FROM study_closures")
    return s if len(s) else None


def busiest_hour(k: pd.DataFrame) -> tuple[str, int, float] | None:
    """The weekday and hour with the most time lost on average (weekly_profile)."""
    w = weekly_profile(k)
    if w.empty:
        return None
    r = w.loc[w.delay_veh_h.idxmax()]
    return DAYS[int(r.weekday)], int(r.hour), float(r.delay_veh_h)
