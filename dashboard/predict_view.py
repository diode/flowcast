"""Predict (docs/architecture-next.md 5.3-5.4): a future date and time on the mock calendar, the traffic expected
then and what to do about it, and the impact of setting one road's traffic level. All from the town's graph model
through the tools Glimmer calls (predict_data, assist.Tools): answers in under a second, no SUMO. Times outside the
town's filmed windows are refused."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from netstudy import assist, days
from netstudy import overview as ov
from netstudy import predict_data as pdd
from netstudy.places import city, impact_word, time_lost
from shared import KEY_HELP, choose_city, choose_period, day_periods, network
from widgets import chart

KINDS = {"working": ("Working day", "#e5e7eb", "#374151"), "weekend": ("Weekend", "#bfdbfe", "#1e3a8a"),
         "holiday": ("Holiday", "#c4b5fd", "#4c1d95"), "festival": ("Festival", "#fdba74", "#7c2d12")}
LEVEL_WORDS = {"busy": "Busy", "slow": "Slow", "jammed": "Jammed", "closed": "Closed"}
TIMES = [f"{m // 60:02d}:{m % 60:02d}" for m in range(15, 24 * 60, 15)]
DEFAULT_DATE = date(2026, 11, 27)                  # a festival day: Harvest Fair (mock)
DEFAULT_TIME = "09:00"                             # inside the morning window (08:30-11:00)
PREDICT_FORM = False                               # the road form, off for now: Glimmer drives the predictions
QUESTIONS = Path(__file__).resolve().parents[1] / "tests"


@st.cache_resource
def tools_for(name: str):
    """The town's graph model and tools (assist.Tools), or None before it has camera-derived state."""
    return pdd.load_tools(network(name))


# ------------------------------------------------------------------ choices
def _from_link() -> None:
    """A shared link: ?mode=Predict&date=2026-11-27&time=18:00&edge=J2_J3&level=closed&minutes=60."""
    if st.session_state.get("p_link_read"):
        return
    st.session_state["p_link_read"] = True
    q = st.query_params
    try:
        if q.get("date"):
            st.session_state["p_date"] = date.fromisoformat(q["date"])
        if q.get("time") in TIMES:
            st.session_state["p_time"] = q["time"]
        if q.get("edge"):                                # becomes a case once the town and time are known
            st.session_state["p_link_change"] = {"edge": q["edge"], "level": q.get("level", "jammed"),
                                                 "minutes": int(q.get("minutes", 60))}
    except ValueError:
        pass


def time_periods(sim: dict | None) -> dict[str, list[str]]:
    """The times Predict answers for: {"Morning 08:30–11:00": [the window's start and 15-minute slots]} per filmed
    window; a town filmed all day: the parts of the day (shared.day_periods)."""
    ws = assist.windows(sim)
    if not ws:
        return day_periods(TIMES, [], step=15)
    return {f"{w['period'].capitalize()} {w['start']}–{w['end']}":
            sorted({w["start"], *(t for t in TIMES if w["start"] <= t < w["end"])}) for w in ws}


def choose_when(day_where, time_where, sim: dict | None = None) -> tuple[date, str]:
    """Date and time: one button per filmed window (picked: its start; a time inside it from a case or a link is
    kept, one outside moves to the nearest window)."""
    _from_link()
    cur = st.session_state.get("p_date", DEFAULT_DATE)
    d = day_where.date_input("Date", value=cur, min_value=days.CAL_FROM, max_value=days.CAL_TO,
                             format="DD/MM/YYYY")
    if d != cur:
        st.session_state["p_date"] = d
    t0 = st.session_state.get("p_time", DEFAULT_TIME)
    t = choose_period(time_where, time_periods(sim), t0, "Time", "p_period", "p_time")
    time_where.caption(f"Forecast from {t}.")
    st.session_state["p_time"] = t
    return st.session_state.get("p_date", d), t


def calendar_option(selected: date) -> dict:
    cal = days.calendar()
    idx = {k: i for i, k in enumerate(KINDS)}
    cals, series = [], []
    for i, (month, left) in enumerate((("2026-11", "2%"), ("2026-12", "52%"))):
        cals.append({"range": month, "orient": "vertical", "left": left, "top": 52, "width": "46%",
                     "cellSize": ["auto", 30], "yearLabel": {"show": False},
                     "dayLabel": {"firstDay": 1, "nameMap": ["S", "M", "T", "W", "T", "F", "S"], "color": ov.MUTED},
                     "monthLabel": {"show": False},
                     "itemStyle": {"borderColor": "#ffffff", "borderWidth": 3, "color": "#f8fafc"},
                     "splitLine": {"show": False}})
        m = cal[[x.strftime("%Y-%m") == month for x in cal.date]]
        data = [[x.isoformat(), idx[k], ev] for x, k, ev in zip(m.date, m.kind, m.event)]
        series.append({"type": "heatmap", "coordinateSystem": "calendar", "calendarIndex": i, "data": data,
                       "itemStyle": {"borderRadius": 6, "borderColor": "#ffffff", "borderWidth": 2}})
        series.append({"type": "scatter", "coordinateSystem": "calendar", "calendarIndex": i, "symbolSize": 1,
                       "data": [[x.isoformat(), idx[k], ev] for x, k, ev in zip(m.date, m.kind, m.event)],
                       "label": {"show": True, "formatter": JS_DAY, "color": ov.INK, "fontSize": 11},
                       "silent": False, "z": 3})
        if selected.strftime("%Y-%m") == month:
            series.append({"type": "effectScatter", "coordinateSystem": "calendar", "calendarIndex": i,
                           "data": [[selected.isoformat(), 0, ""]], "symbolSize": 26, "z": 2,
                           "itemStyle": {"color": "rgba(15,118,110,0.0)", "borderColor": ov.ACCENT, "borderWidth": 2},
                           "rippleEffect": {"scale": 1.6, "brushType": "stroke"}})
    return {
        "tooltip": {"formatter": JS_CAL_TIP},
        "title": [{"text": t, "left": x, "top": 0, "textStyle": {"fontSize": 13, "color": ov.INK, "fontWeight": 700}}
                  for t, x in (("November 2026", "2%"), ("December 2026", "52%"))],
        "visualMap": {"type": "piecewise", "dimension": 1, "orient": "horizontal", "left": "center", "bottom": 0,
                      "itemWidth": 14, "itemHeight": 14, "textStyle": {"color": ov.MUTED}, "selectedMode": False,
                      "pieces": [{"value": i, "label": lab, "color": col} for i, (lab, col, _) in enumerate(KINDS.values())],
                      "seriesIndex": [k for k, s in enumerate(series) if s["type"] == "heatmap"]},
        "calendar": cals, "series": series,
    }


JS_DAY = "function (p) { var d = +p.value[0].slice(8); return p.value[2] ? d + '★' : '' + d; }"
JS_CAL_TIP = ("function (p) { var k = ['Working day','Weekend','Holiday','Festival'][p.value[1]];"
              " return '<b>' + p.value[0] + '</b><br/>' + k + (p.value[2] ? '<br/>' + p.value[2] + ' (mock)' : ''); }")
JS_CLICK = "function (p) { return p.value[0]; }"
JS_MAP = ("function (p) { var d = p.data || {}; var v = d.case ? 'case:' + d.case : d.edge;"
          " return v ? v + '|' + Date.now() : null; }")              # new on every click, even of the same road


# ------------------------------------------------------------------ pieces
def range_tile(label: str, value: float, unit: str, lo: float, hi: float, note: str, digits: int = 0) -> str:
    fmt = lambda x: "–" if pd.isna(x) else f"{x:,.{digits}f}"  # noqa: E731
    if pd.notna(value) and pd.notna(hi) and value > hi * 1.05 + (0.5 if digits else 1):
        badge = '<span class="delta bad">▲ above usual</span>'
    elif pd.notna(value) and pd.notna(lo) and value < lo * 0.95 - (0.5 if digits else 1):
        badge = '<span class="delta good">▼ below usual</span>'
    else:
        badge = '<span class="delta flat">≈ usual</span>'
    return (f'<div class="tile"><div class="tile-label">{label}</div><div class="tile-row"><div>'
            f'<span class="tile-value">{fmt(value)}</span><span class="tile-unit">{unit}</span>'
            f'<div class="tile-delta">{badge} range {fmt(lo)}–{fmt(hi)}</div></div></div>'
            f'<div class="tile-note">{note}</div></div>')


def action_cards(acts: list[dict], pl) -> str:
    if not acts:
        return ('<div class="insight muted"><div class="insight-icon">✅</div><div>No action needed: '
                'traffic is expected within its usual range here.</div></div>')
    by = {}
    for a in acts:                                     # one card per junction, most urgent first
        by.setdefault(a["junction"], []).append(a)
    out = []
    for j, items in by.items():
        lines = "".join(f'<div class="act">{act_symbol(a)}<div>{a["action"]} {priority_chip(a)}'
                        f'<div class="ans-sub">{a["when"]} · {a["why"]}</div></div></div>' for a in items)
        out.append(f'<div class="insight stack"><b>{pl.node(j)}</b>{lines}</div>')
    return '<div class="insights">' + "".join(out) + "</div>"


def act_symbol(a: dict) -> str:
    """The action's symbol (actions.symbol: a detour, signals, an officer, …), tinted by its priority."""
    from netstudy.actions import symbol

    return f'<span class="sym {a["priority"]}">{symbol(a["action"])}</span>'


def priority_chip(a: dict) -> str:
    return f'<span class="pri-chip {a["priority"]}">{a["priority"].capitalize()}</span>'


def curve_option(exp: dict, hhmm: str, label: str) -> dict:
    """Vehicles on the roads through the day: the usual range of the day type, and the prediction."""
    b = exp["band"]
    x = b.index.tolist()
    lo, hi = b.veh_in_town_10.round(1).tolist(), b.veh_in_town_90.round(1).tolist()
    pred = exp["curve"].veh_in_town.reindex(x).round(1).tolist()
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 0, "left": 0, "icon": "roundRect", "itemHeight": 4, "textStyle": {"color": ov.MUTED},
                   "data": ["Usual range", label]},
        "grid": {"left": 44, "right": 16, "top": 40, "bottom": 28},
        "xAxis": {"type": "category", "data": x, "boundaryGap": False, "axisTick": {"show": False},
                  "axisLine": {"lineStyle": {"color": ov.GRID}}, "axisLabel": {"color": ov.MUTED, "interval": ov.JS_EVERY_3H}},
        "yAxis": {"type": "value", "splitLine": {"lineStyle": {"color": ov.GRID}}, "axisLabel": {"color": ov.MUTED}},
        "series": [
            {"name": "low", "type": "line", "data": lo, "stack": "band", "showSymbol": False, "lineStyle": {"opacity": 0},
             "tooltip": {"show": False}},
            {"name": "Usual range", "type": "line", "data": [round(h - l, 1) for h, l in zip(hi, lo)], "stack": "band",
             "showSymbol": False, "lineStyle": {"opacity": 0}, "areaStyle": {"color": "rgba(148,163,184,0.25)"},
             "itemStyle": {"color": "#cbd5e1"}},
            {"name": label, "type": "line", "data": pred, "showSymbol": False, "smooth": 0.2,
             "lineStyle": {"color": ov.ACCENT, "width": 2}, "itemStyle": {"color": ov.ACCENT},
             "markLine": {"symbol": "none", "label": {"formatter": hhmm, "color": ov.INK},
                          "lineStyle": {"color": ov.INK, "type": "dotted"}, "data": [{"xAxis": hhmm}]}},
        ],
    }


def impact_curve_option(curve: pd.DataFrame, pl, start: str, end: str) -> dict:
    colours = ["#c5221f", "#e8710a", "#1a73e8", "#0f766e", "#7c3aed", "#db2777"]
    return {
        "tooltip": {"trigger": "axis", "valueFormatter": JS_VMIN},
        "legend": {"top": 0, "left": 0, "icon": "roundRect", "itemHeight": 4, "textStyle": {"color": ov.MUTED}},
        "grid": {"left": 48, "right": 16, "top": 40, "bottom": 28},
        "xAxis": {"type": "category", "data": curve.index.tolist(), "axisTick": {"show": False},
                  "axisLine": {"lineStyle": {"color": ov.GRID}}, "axisLabel": {"color": ov.MUTED}},
        "yAxis": {"type": "value", "splitLine": {"lineStyle": {"color": ov.GRID}}, "axisLabel": {"color": ov.MUTED}},
        "series": [{"name": pl.node(j), "type": "line", "data": curve[j].round(1).tolist(), "showSymbol": False,
                    "lineStyle": {"width": 2, "color": colours[i % len(colours)]},
                    "itemStyle": {"color": colours[i % len(colours)]},
                    **({"markArea": {"itemStyle": {"color": "rgba(168,85,247,0.08)"},
                                     "data": [[{"xAxis": start}, {"xAxis": end}]]}} if i == 0 else {})}
                   for i, j in enumerate(curve.columns)],
    }


JS_VMIN = "function (v) { return (v > 0 ? '+' : '') + v.toFixed(0) + ' min lost'; }"


def day_banner(info: dict, d: date, hhmm: str, exp: dict | None) -> str:
    kind = days.day_kind(d)
    lab, bg, ink = KINDS[kind]
    e = days.event_on(d)
    ev = ""
    if e:
        times = f" {e['start']}–{e['end']}" if e.get("start") else ""
        ev = f' · <b>{e["name"]}</b>{times} <span class="mock">mock event</span>'
    ctx = days.time_context(kind, hhmm, e)
    src = ""
    if exp is not None:
        src = f" · from {exp['days']} days of camera data" + (
            f", based on a usual {exp['kind']} day (no {kind} day recorded yet)" if exp["assumed"]
            else f" ({exp['kind']} days)")
    return (f'<div class="pbanner"><div><div class="hero-kind">{info["name"]} · forecast</div>'
            f'<div class="pdate">{d:%A %d %B %Y} · {hhmm}</div>'
            f'<div class="ptag"><span class="chip" style="background:{bg};color:{ink}">{lab}</span>{ev}</div>'
            f'<div class="ptag">{ctx[0].upper() + ctx[1:]}{src}</div></div></div>')


def source_badges(net, assumed: bool) -> str:
    """How many roads are observed (camera) / estimated (no camera) / assumed (holiday without recorded days)."""
    n = pdd.road_sources(net, assumed).value_counts().astype(int).to_dict()
    return "".join(f'<span class="badge {k}" title="{pdd.SOURCES[k]}">{k} · {n[k]} road{"s" * (n[k] != 1)}</span>'
                   for k in pdd.SOURCES if n.get(k))


def accuracy_badge(acc: dict | None) -> str:
    if not acc:
        return ""
    if "in_range" in acc:
        txt = f"flow WAPE {acc['wape']:.2f}, {100 * acc['in_range']:.0f}% inside the usual 10–90%"
    else:
        txt = f"WAPE {acc['wape']:.2f}, sign right on {100 * acc['sign_ok']:.0f}% of roads" + (
            f", {acc['hit']} hit / {acc['missed']} missed / {acc['false']} false" if acc["case"] else "")
    return (f'<span class="badge acc" title="How close the traffic model came to simulated test runs of the same road changes">method accuracy in simulation ({acc["what"]}): {txt}</span>')


def badges(*parts: str) -> str:
    inner = "".join(p for p in parts if p)
    return f'<div class="badges">{inner}</div>' if inner else ""


PSTYLE = """<style>
.pbanner { padding: 20px 26px; border-radius: 18px; margin: 4px 0 16px; color:#f8fafc;
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #0f766e 140%); }
.pdate { font-size: 30px; font-weight: 800; letter-spacing: -.02em; margin: 2px 0 8px; }
@media (max-width: 640px) { .pdate { font-size: 22px; } .pbanner { padding: 16px 18px; } }
.ptag { color:#e2e8f0; font-size: 14.5px; margin-top: 4px; }
.chip { display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 700; font-size: 12.5px; margin-right: 6px; }
.mock { font-size: 11px; color:#c7d2fe; background: rgba(255,255,255,.12); border-radius: 6px; padding: 0 5px; margin-left: 4px; }
.sumrow { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin: 6px 0 12px; }
@media (max-width: 900px) { .sumrow { grid-template-columns: repeat(2, minmax(0,1fr)); } }
.sum { background: var(--soft); border-radius: 12px; padding: 10px 14px; }
.sum b { display:block; font-size: 22px; color: var(--ink); }
.sum span { font-size: 12.5px; color: var(--muted); }
.badges { display:flex; flex-wrap:wrap; gap:6px; margin: 14px 0 2px; }
.ptiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.ptiles .tile { display:flex; flex-direction:column; }
.ptiles .tile-note { margin-top: auto; padding-top: 10px; }
.badge { display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: var(--soft); color: var(--muted); }
.badge.observed { background:#e3f8ee; color:#065f46; }
.badge.estimated { background:#e6effe; color:#1e40af; }
.badge.assumed { background:#fdf0e3; color:#9a3412; }
.badge.acc { background:#efebfe; color:#5b21b6; }
.acts { margin: 4px 0 0 0; padding-left: 18px; }
.acts li { margin: 3px 0; }
.verdict { border-radius: 12px; padding: 10px 14px; margin: 8px 0 6px; font-size: 15px; box-shadow: var(--shadow-sm); }
.verdict.recommended { background:#e9f9f0; color:#05603a; }
.verdict.caution { background:#fff6e0; color:#93370d; }
.verdict.not { background:#fdeeec; color:#912018; }
.verdict .best { font-size: 11.5px; font-weight: 700; background:#fff; border-radius: 999px; padding: 1px 8px;
  margin-left: 6px; box-shadow: var(--shadow-sm); }
.ans-head { font-weight: 700; color: var(--ink); font-size: 14px; margin: 10px 0 4px; }
.ans { margin: 0 0 4px 0; padding-left: 20px; color: var(--ink); font-size: 14px; line-height: 1.45; }
.ans li { margin: 3px 0; }
.ans.items { list-style: none; padding-left: 0; }
.ans.items li { display:flex; gap: 10px; align-items: flex-start; margin: 6px 0; }
.ans-sub { font-size: 12.5px; color: var(--muted); font-weight: 400; }
.ask-sel { display:flex; gap: 10px; align-items: flex-start; background: var(--soft); border-radius: 12px;
  padding: 10px 12px; margin: 2px 0 10px; font-size: 14px; line-height: 1.45; color: var(--ink); }
.sym.q { background:#e6f4f1; color: var(--accent); }
</style>"""
ACTION_ITEMS = 4
VERDICT_SHOW = {"recommended": ("✅", "Recommended"), "caution": ("⚠️", "With caution"),
                "not": ("⛔", "Not recommended")}


# ------------------------------------------------------------------ overview
def overview(bar=None) -> None:
    """The map in the middle: the expected traffic, or the impact of the prediction case picked beside it. Cases are
    the road changes of Glimmer's answers (or the form's, PREDICT_FORM); each is predicted on its own, and removing
    them brings the map back to the expected traffic."""
    st.markdown(PSTYLE, unsafe_allow_html=True)
    c = (bar or st).columns([1.1, 1.1, 2.8], vertical_alignment="bottom")
    name = choose_city(c[0], page="predict")
    _reset_on_town(name)
    net, info = network(name), city(name)
    tools, sim = tools_for(name), pdd.town_config(name)
    d, hhmm = choose_when(c[1], c[2], sim)
    ok = tools is not None and pdd.in_windows(sim, hhmm)
    st.markdown(day_banner(info, d, hhmm, expected_at(name, d, hhmm) if ok else None), unsafe_allow_html=True)
    if _glimmer_on(name):
        with st.container(border=True):
            _server_row()
    workspace(name, d, hhmm)
    with st.container(border=True):
        st.markdown('<div class="card-title">Calendar <span class="card-sub">click a day · ★ festival or holiday '
                    '(mock)</span></div>', unsafe_allow_html=True)
        clicked = st_click(calendar_option(d), 300, f"cal-{d}")
        if clicked and clicked != st.session_state.get("p_cal_last"):
            st.session_state["p_cal_last"] = clicked
            st.session_state["p_date"] = date.fromisoformat(clicked)
            st.rerun()


RESET_KEYS = ("p_cases", "p_next_case", "p_shown", "p_change", "p_answer", "p_q", "p_q_road", "p_q_mine", "p_road",
              "p_click_seen", "p_click_case", "p_road_pending", "p_link_change")


def _reset_on_town(name: str) -> None:
    """Another town: a fresh prediction view (no cases, no question or answer, the map's click forgotten)."""
    ss = st.session_state
    old = ss.get("p_town")
    if old is not None and old != name:
        for k in RESET_KEYS + (map_key(old),):
            ss.pop(k, None)
    ss["p_town"] = name


@st.cache_data(max_entries=256, show_spinner=False)
def expected_at(name: str, d: date, hhmm: str) -> dict:
    return pdd.expected(network(name), tools_for(name), d, hhmm)


@st.cache_data(max_entries=256, show_spinner=False)
def case_at(name: str, d: date, hhmm: str, edge: str, level: str, minutes: int, also: tuple = ()) -> dict:
    """`also`: ((edge, level), …) changed at the same time."""
    return pdd.case_result(network(name), tools_for(name), d, hhmm,
                           {"edge": edge, "level": level, "minutes": minutes, "also": also})


def map_key(name: str) -> str:
    """One map per town, kept across reruns: a new option redraws it in place (no remount)."""
    return f"pmap-{name}"


@st.fragment
def workspace(name: str, d: date, hhmm: str) -> None:
    """The map, Glimmer's question box (or the road form), the cases and the shown case's details. A click, a question
    or a case picked reruns only this part; a case at another date or time reruns the page (its controls move there)."""
    ss = st.session_state
    v = ss.get(map_key(name))                     # the map's last click: "edge|time" or "case:N|time", new each click
    if v and v != ss.get("p_click_seen"):
        ss["p_click_seen"] = v
        what = v.split("|")[0]
        if what.startswith("case:"):                       # a numbered pin: show that case
            _select(next((c for c in _cases(name) if str(c["id"]) == what[5:]), None))
        elif _form_on(name):
            ss["p_click_case"] = ss["p_road_pending"] = what
        else:                                              # a road: a question about it in Glimmer's box
            ss["p_road"] = ss["p_q_road"] = what
    if (ss.get("p_date", d), ss.get("p_time", hhmm)) != (d, hhmm):
        st.rerun()
    net, info = network(name), city(name)
    tools, sim = tools_for(name), pdd.town_config(name)
    pl = ov.places(net)
    ok = tools is not None and pdd.in_windows(sim, hhmm)
    if ok and "p_link_change" in ss:
        _add_case(name, d, hhmm, ss.pop("p_link_change"))
    clicked = ss.pop("p_click_case", None)
    if ok and clicked:                                     # a road clicked on the map: a case at the form's level
        _add_case(name, d, hhmm, {"edge": clicked, "level": ss.get("p_level_w") or "jammed",
                                  "minutes": int(ss.get("p_min_w", 60))})
    exp = expected_at(name, d, hhmm) if ok else None
    case = _shown_case(name, d, hhmm) if ok else None
    r = case_at(name, d, hhmm, case["edge"], case["level"], int(case["minutes"]),
                tuple(tuple(x) for x in case.get("also", []))) if case else None

    a, b = st.columns([2.2, 1], gap="medium")
    with a.container(border=True):
        _map_card(name, net, info, pl, d, hhmm, sim, tools, exp, case, r)
    with b:
        if _form_on(name):
            _road_form(name, net, pl, d, hhmm, enabled=ok)
        else:
            _ask_card(name, net, tools, sim, pl, d, hhmm)
        _case_list(name, pl, case)
    if r is not None:
        _impact(net, pl, {"town": name, "date": d.isoformat(), "time": hhmm, "change": _change(case)}, r)
    elif exp is not None:
        _expected(name, net, pl, d, hhmm, sim, tools, exp)


def _glimmer_on(name: str) -> bool:
    return tools_for(name) is not None and pdd.predict_settings(pdd.town_config(name))["assist"] != "off"


def _form_on(name: str) -> bool:
    """The road form: switched on (PREDICT_FORM), or the only way to set a road (no Glimmer for the town)."""
    return PREDICT_FORM or not _glimmer_on(name)


def _map_card(name, net, info, pl, d, hhmm, sim, tools, exp, case, r) -> None:
    if tools is None or exp is None:
        st.markdown('<div class="card-title">Road map</div>', unsafe_allow_html=True)
        if tools is None:
            st.info(f"{info['name']} has no camera data yet, so there is nothing to predict from.")
        else:
            st.warning(f"{hhmm} is outside the study periods, so there is no camera data to predict from. "
                       f"Study periods: {pdd.covered(sim)}.")
        return
    sel = _selected_road(net, pl)
    marks = [{"edge": e, "label": str(c["id"]), "case": c["id"], "on": case is not None and c["id"] == case["id"],
              "here": c["date"] == d.isoformat() and c["time"] == hhmm,
              "tip": f"<b>Case {_case_label(c, pl)}</b><br/>click to show it"}
             for c in _cases(name) for e in [c["edge"]] + [x[0] for x in c.get("also", [])]]   # a pin on each road
    if case is None:
        st.markdown(f'<div class="card-title">Expected road map <span class="card-sub">{d:%a %d %b} {hhmm} · no road '
                    'change</span></div>', unsafe_allow_html=True)
        opt = ov.map_option(net, exp["state"], pl, aspect=1.6, highlight=sel, marks=marks)
        cap = ("Click a road to add it as a case, at the level and length set on the right; numbered pins are your "
               "cases." if _form_on(name) else
               "Click a road to ask the Traffic Assistant about it; numbered pins are your forecast cases.")
    else:
        st.markdown(f'<div class="card-title">Impact of case {case["id"]}: {_what(case, pl)} '
                    f'{case["minutes"]} min <span class="card-sub">extra delay per road from {hhmm} '
                    'until the queue has cleared</span></div>', unsafe_allow_html=True)
        opt = ov.impact_map_option(net, r["edges"], pl, {case["edge"], *(x[0] for x in case.get("also", []))},
                                   aspect=1.6, highlight=sel, marks=marks)
        cap = (f"Purple: the road of the case shown. Click another road to "
               f"{'add it as a new case' if _form_on(name) else 'ask the Traffic Assistant about it'}, or a numbered pin to show "
               "that case; remove the cases (right) to go back to the expected traffic.")
    st_click(opt, 520, map_key(name), JS_MAP)               # its click is read at the top of workspace()
    st.caption(cap)


def _expected(name, net, pl, d, hhmm, sim, tools, exp) -> None:
    a, b = st.columns([1.25, 1], gap="medium")
    with a.container(border=True):
        st.markdown(f'<div class="card-title">Expected at {hhmm}</div>', unsafe_allow_html=True)
        t = pdd.kpi_tiles(exp, hhmm)
        src = "usual for this kind of day" + (" (assumed)" if exp["assumed"] else "")
        st.markdown('<div class="ptiles">' + "".join([
            range_tile("Vehicles on the roads", t["veh_in_town"]["now"], "", t["veh_in_town"]["lo"],
                       t["veh_in_town"]["hi"], f"Forecast: {src}"),
            range_tile("Extra travel time", t["speed_index"]["now"], "%", t["speed_index"]["lo"],
                       t["speed_index"]["hi"], "How much longer trips take than on empty roads"),
            range_tile("Time lost per 5 minutes", t["delay_veh_h"]["now"] * 60, "min", t["delay_veh_h"]["lo"] * 60,
                       t["delay_veh_h"]["hi"] * 60, "All drivers on the network together"),
        ]) + "</div>" + badges(source_badges(net, exp["assumed"]), accuracy_badge(pdd.accuracy(net, sim))),
            unsafe_allow_html=True)
    with b.container(border=True):
        st.markdown('<div class="card-title">Recommended actions <span class="card-sub">around this time</span></div>',
                    unsafe_allow_html=True)
        ev = days.event_on(d)
        st.markdown(action_cards(pdd.day_actions(tools, exp, hhmm, ev and ev["name"]), pl), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f'<div class="card-title">Through the day <span class="card-sub">vehicles on the roads, against '
                    f'the usual range of {"weekends and holidays" if d.weekday() >= 5 or days.day_kind(d) == "holiday" else "working days"}'
                    f' · study periods: {pdd.covered(sim)}</span></div>', unsafe_allow_html=True)
        chart(curve_option(exp, hhmm, "Expected (usual median)"), 280, f"pcurve-{name}-{d}-{hhmm}")


# ------------------------------------------------------------------ prediction cases
def _cases(name: str) -> list[dict]:
    return [c for c in st.session_state.get("p_cases", []) if c["town"] == name]


def _change(c: dict | None) -> dict | None:
    return c and {**{k: c[k] for k in ("edge", "level", "minutes")}, "also": [list(x) for x in c.get("also") or []]}


def _what(c: dict, pl) -> str:
    """The road(s) of a case and their level: 'Elm Park → Ashford Centre closed', two or three roads together."""
    road, lv = assist.change_label(pl, c["edge"], c["level"], dict(tuple(x) for x in c.get("also") or []))
    return f"{road} {lv}" if lv != "changed" else road


def _add_case(name: str, d: date, hhmm: str, change: dict, by: str = "you") -> None:
    """A road change at a date and time as a case (the same one again picks the existing case) and show it."""
    cases = st.session_state.setdefault("p_cases", [])
    spec = {"town": name, "date": d.isoformat(), "time": hhmm, **change,
            "also": [list(x) for x in change.get("also") or []]}
    c = next((c for c in cases if all(c[k] == v for k, v in spec.items())), None)
    if c is None:
        c = {**spec, "id": st.session_state.get("p_next_case", 1), "by": by}
        st.session_state["p_next_case"] = c["id"] + 1
        cases.append(c)
    _select(c)


def _select(c: dict | None) -> None:
    """Show a case on the map (the page moves to its date and time), or none: the expected traffic."""
    st.session_state["p_shown"] = c and c["id"]
    st.session_state["p_change"] = _change(c)                # the technical view's road change
    if c:
        st.session_state["p_date"], st.session_state["p_time"] = date.fromisoformat(c["date"]), c["time"]


def _remove(cid: int) -> None:
    st.session_state["p_cases"] = [c for c in st.session_state.get("p_cases", []) if c["id"] != cid]
    if st.session_state.get("p_shown") == cid:
        _select(None)


def _clear(name: str) -> None:
    st.session_state["p_cases"] = [c for c in st.session_state.get("p_cases", []) if c["town"] != name]
    _select(None)


def _shown_case(name: str, d: date, hhmm: str) -> dict | None:
    """The case on the map: the one picked, while the page is at its date and time."""
    return next((c for c in _cases(name) if c["id"] == st.session_state.get("p_shown")
                 and c["date"] == d.isoformat() and c["time"] == hhmm), None)


def _case_label(c: dict, pl) -> str:
    return (f"{c['id']}. {_what(c, pl)} · {c['minutes']} min · "
            f"{date.fromisoformat(c['date']):%a %d %b} {c['time']}")


def _road_form(name, net, pl, d, hhmm, enabled: bool) -> None:
    ss = st.session_state
    with st.container(border=True):
        st.markdown(f'<div class="card-title">Set a road\'s traffic level <span class="card-sub">from {hhmm} · or '
                    'click the road on the map</span></div>', unsafe_allow_html=True)
        ids = sorted(net.edges().edge_id, key=pl.road)
        if "p_road_pending" in ss:                          # a map click; set before the picker is drawn
            ss["p_road_w"] = ss.pop("p_road_pending")
        if ss.get("p_road_w") not in ids:                   # first time, or another town
            ss["p_road_w"] = _selected_road(net, pl)
        ss["p_road"] = st.selectbox("Road", ids, key="p_road_w", format_func=pl.road)
        ss.setdefault("p_level_w", "jammed")
        ss.setdefault("p_min_w", 60)
        st.segmented_control("Traffic level", list(LEVEL_WORDS), format_func=LEVEL_WORDS.get, key="p_level_w")
        st.selectbox("For", [30, 60, 90, 120], key="p_min_w", format_func=lambda m: f"{m} min")
        st.button("Add case", type="primary", disabled=not enabled, icon=":material/add:", width="stretch",
                  on_click=_add_from_form, args=(name, d, hhmm))
        if not enabled:
            st.caption("Pick a time inside the filmed windows (above).")


def _add_from_form(name: str, d: date, hhmm: str) -> None:
    ss = st.session_state
    _add_case(name, d, hhmm, {"edge": ss["p_road_w"], "level": ss.get("p_level_w") or "jammed",
                              "minutes": int(ss["p_min_w"])})


def _case_list(name, pl, shown: dict | None) -> None:
    cases = _cases(name)
    with st.container(border=True):
        st.markdown('<div class="card-title">Forecast cases <span class="card-sub">each forecast on its own · '
                    'click one to show it on the map</span></div>', unsafe_allow_html=True)
        if not cases:
            st.caption("None: the map shows the expected traffic. "
                       + ("Add a road change above." if _form_on(name) else
                          "Ask the Traffic Assistant about a road change: it appears here and on the map."))
            return
        for c in cases:
            col = st.columns([6, 1], vertical_alignment="center", gap="small")
            on = shown is not None and shown["id"] == c["id"]
            col[0].button(_case_label(c, pl), key=f"p_case_{c['id']}", type="primary" if on else "secondary",
                          width="stretch", on_click=_select, args=(c,), help=f"added by {c['by']}")
            col[1].button("✕", key=f"p_del_{c['id']}", help="Remove this case", on_click=_remove, args=(c["id"],))
        col = st.columns([1, 3])
        col[0].button("", key="p_none", disabled=shown is None, icon=":material/restore:", width="stretch",
                      on_click=_select, args=(None,), help="Expected traffic: the map without any case (keeps them)")
        col[1].button("Remove all", key="p_clear", icon=":material/delete_sweep:", width="stretch",
                      on_click=_clear, args=(name,), help="Back to the expected traffic, no cases")


# ------------------------------------------------------------------ Glimmer
SERVER_WORDS = {"off": "⚪ off", "loading": "🟡 starting (about a minute)", "ready": "🟢 ready",
                "running elsewhere": "🟢 ready", "no key": "🔑 needs an API key"}


@st.cache_data(show_spinner=False)
def sample_questions(name: str) -> list[str]:
    """The town's Glimmer test questions that it answers (tests/assist_questions_<town>.yaml; Ashford's is
    tests/assist_questions.yaml), not the ones it should refuse."""
    import yaml

    f = QUESTIONS / ("assist_questions.yaml" if name == "ashford" else f"assist_questions_{name}.yaml")
    if not f.exists():
        return []
    return [q["q"] for q in yaml.safe_load(f.read_text(encoding="utf-8"))["questions"] if not q.get("no_change")]


def _ask_card(name, net, tools, sim, pl, d, hhmm) -> None:
    """Ask Glimmer (architecture-next 4.3), beside the map: an editable dropdown of the town's sample questions
    (a road clicked on the map adds one about it, on top). Glimmer only reads the question and words the insight
    (Assistant.predict); the page's date and time fill in a question without them; its road changes become the
    cases."""
    from netstudy import assist_backend as ab

    ss = st.session_state
    with st.container(border=True):
        st.markdown('<div class="card-title">Ask the Traffic Assistant <span class="card-sub">pick an example or '
                    'type your own question</span></div>', unsafe_allow_html=True)
        mine = ss.setdefault("p_q_mine", {}).setdefault(name, [])     # questions from map clicks, newest first
        if ss.get("p_q_road"):                                       # set before the box is drawn
            q = (f"What if {pl.road(ss.pop('p_q_road'))} is jammed on {d:%A} {d.isoformat()} at {hhmm} "
                 "for 60 minutes?")
            mine[:] = [q] + [x for x in mine if x != q][:4]
            ss["p_q"] = q
        options = mine + [q for q in sample_questions(name) if q not in mine]
        if ss.get("p_q") and ss["p_q"] not in options:              # typed earlier: keep it in the list
            options.insert(0, ss["p_q"])
        if ab.status() != "ready":
            st.caption(("Add the Meta Model API key (see above)" if ab.current() == "muse_spark" else
                        "Start the Traffic Assistant (above)") + " to ask questions. Until then the map shows the "
                       "expected traffic and your forecast cases.")
            return
        q = st.selectbox("Question", options, key="p_q", index=None, accept_new_options=True,
                         label_visibility="collapsed", placeholder="Pick an example question or type your own")
        if q:                                                  # the dropdown cuts long questions: in full here
            import html

            st.markdown(f'<div class="ask-sel"><span class="sym q">chat</span><div>{html.escape(q)}</div></div>',
                        unsafe_allow_html=True)
        go = st.button("Ask", type="primary", icon=":material/send:", width="stretch", key="p_ask_go",
                       disabled=not (q or "").strip())
        if go and (q or "").strip():
            ss["p_answer"] = a = _ask(name, net, tools, sim, q.strip(), (d, hhmm))
            for c in a.get("changes", []):                 # the answer's road changes join the cases, the last shown
                _add_case(name, date.fromisoformat(c["date"]), c["time"], _change(c), by="the Traffic Assistant")
            if a.get("changes"):
                _rerun_here()
        elif ss.get("p_answer", {}).get("town") == name:
            _show_answer(ss["p_answer"])


def _rerun_here() -> None:
    """Rerun the workspace only (its map and cases); the whole page when this run was not the fragment's own."""
    from streamlit.errors import StreamlitAPIException

    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def _server_row() -> None:
    """The Traffic Assistant's state for the model chosen in the app menu: Glimmer with Start / Stop for its local
    server, Muse Spark with where to put its API key when none is set."""
    from netstudy import assist_backend as ab
    from netstudy import glimmer_server as gs

    if ab.current() == "muse_spark":
        s = ab.status("muse_spark")
        st.markdown(f"**Traffic Assistant** · {SERVER_WORDS[s]} · Muse Spark through the Meta Model API · answers your "
                    "questions beside the map")
        if s == "no key":
            st.info(KEY_HELP)
        return
    s = gs.status()
    c = st.columns([3, 1, 1], vertical_alignment="center")
    c[0].markdown(f"**Traffic Assistant** · {SERVER_WORDS[s]} · answers your questions beside the map")
    if c[1].button("Start", disabled=s != "off", icon=":material/play_arrow:", width="stretch", key="p_srv_start",
                   help="Start the Traffic Assistant (about a minute)"):
        try:
            gs.start()
        except SystemExit as e:
            st.session_state["p_server_msg"] = str(e)
        st.rerun()
    if c[2].button("Stop", disabled=s not in ("loading", "ready"), icon=":material/stop:", width="stretch",
                   key="p_srv_stop", help="Stop the Traffic Assistant and free its graphics card"):
        gs.stop()
        st.session_state.pop("p_answer", None)
        st.rerun()
    if st.session_state.get("p_server_msg"):
        st.warning(st.session_state.pop("p_server_msg"))
    if s == "loading":
        _wait_ready(gs)


@st.fragment(run_every=5)
def _wait_ready(gs) -> None:
    """Poll while the model loads (about a minute), then redraw the page."""
    if gs.status() != "loading":
        st.rerun(scope="app")
    st.caption("Starting the Traffic Assistant …")


def _ask(name, net, tools, sim, q: str, when: tuple[date, str] | None = None) -> dict:
    """Glimmer's plan, the graph model's prediction, Glimmer's insight (about 10 s); the answer kept for the page."""
    from urllib.error import URLError

    from netstudy.assist import Assistant

    out = {"town": name, "question": q}
    with st.spinner("Forecasting …"):
        try:
            ans = Assistant(name, sim, net.dir, tools=tools).predict(q, when)
        except (URLError, OSError, ValueError) as e:              # ValueError: a plan that is not JSON
            st.error(f"The Traffic Assistant could not answer ({e}). The map still shows the expected traffic and the cases you "
                     "have.")
            return out
    out.update(text=ans.text, numbers_ok=ans.numbers_ok, facts=ans.facts, seconds=ans.seconds,
               insights=ans.insights, actions=ans.actions, verdicts=ans.verdicts,
               changes=[{**{k: c[k] for k in ("edge", "level", "minutes", "date", "time")},
                         "also": [[k, v] for k, v in (c.get("also") or {}).items()]} for c in tools.cases.values()])
    _show_answer(out)
    return out


def best_tag(v: dict, n: int) -> str:
    """The tag of the least harmful option when changes are compared: 'better' only when it is acceptable."""
    return "Better option" if v["verdict"] != "not" else f"Less harmful of the {n}"


def _show_answer(a: dict) -> None:
    """Quick insights (Glimmer's, one line each) and action items (the rules', by priority); the facts they were
    written from folded away (shown as the insights when the wording used a number that is not in them)."""
    if "text" not in a:
        return
    import html

    for v in a.get("verdicts", []):
        icon, word = VERDICT_SHOW[v["verdict"]]
        st.markdown(f'<div class="verdict {v["verdict"]}"><b>{icon} {word}</b> · {impact_word(v["delay"])} impact'
                    + (f' <span class="best">{best_tag(v, len(a["verdicts"]))}</span>' if v["best"] else "")
                    + f'<div class="ans-sub">{html.escape(v["road"])} {v["level"]} {v["minutes"]} min, '
                      f'{html.escape(v["when"])} · {html.escape(v["why"])}</div></div>', unsafe_allow_html=True)
    ins = a.get("insights", []) if a["numbers_ok"] else [f for f in a["facts"]
                                                         if not f.startswith(("Actions:", "Verdict"))]
    st.markdown('<div class="ans-head">💡 Quick insights</div><ul class="ans">'
                + "".join(f"<li>{html.escape(x)}</li>" for x in ins) + "</ul>", unsafe_allow_html=True)
    acts = a.get("actions", [])
    if acts:
        many = len({x["road"] for x in acts}) > 1                # two cases compared: say which road each is for
        st.markdown('<div class="ans-head">📋 Action items</div><ul class="ans items">' + "".join(
            f'<li>{act_symbol(x)}<div><b>{html.escape(x["action"])}</b> {priority_chip(x)}'
            f'<div class="ans-sub">{html.escape(x["where"])} · {html.escape(str(x["when"]))}'
            + (f' · {html.escape(x["road"])}' if many else "") + "</div></div></li>" for x in acts[:ACTION_ITEMS])
            + "</ul>", unsafe_allow_html=True)
    n = len(a.get("changes", []))
    st.caption(f"{a['question']} · {a['seconds']:.0f} s" + (f" · {n} case{'s' * (n > 1)} on the map" if n else "")
               + ("" if a["numbers_ok"] else " · showing the forecast's own figures"))
    if a["numbers_ok"] and a["facts"]:
        with st.expander("Facts behind it"):
            st.markdown("\n".join(f"- {f}" for f in a["facts"]))


def st_click(option: dict, height: int, key: str, on_click: str = JS_CLICK):
    from streamlit_echarts import st_echarts

    from widgets import js
    return st_echarts(js(option), height=f"{height}px", key=key, events={"click": on_click})


def _selected_road(net, pl) -> str:
    """The road picked for a change: clicked or chosen last, else the last change's, else the first."""
    ids = net.edges().edge_id.tolist()
    for cand in (st.session_state.get("p_road"), (st.session_state.get("p_change") or {}).get("edge")):
        if cand in ids:
            return cand
    return sorted(ids, key=pl.road)[0]




def _impact(net, pl, spec, r) -> None:
    """The shown case below the map: summary, actions at the affected junctions, their extra delay over time."""
    j = r["junction"]
    hit = j[j.affected == 1].sort_values("extra_delay_veh_h", ascending=False)
    ch = spec["change"]
    st.markdown(
        '<div class="sumrow">'
        f'<div class="sum"><b>{len(hit)} of {len(j)}</b><span>junctions affected</span></div>'
        f'<div class="sum"><b>+{time_lost(r["total_extra_h"])}</b><span>lost by drivers at those junctions · '
        f'{impact_word(r["total_extra_h"]).lower()} impact</span></div>'
        f'<div class="sum"><b>{r["worst"]}</b><span>worst moment · longest queue {r["max_queue_veh"]:.0f} vehicles'
        '</span></div>'
        f'<div class="sum"><b>{spec["time"]}–{r["end"]}</b><span>{_what(ch, pl)}'
        f'{" · over at " + r["over_at"] if r["over_at"] else ""}</span></div>'
        '</div>' + badges(source_badges(net, r["summary"]["assumed"]),
                          accuracy_badge(None if ch.get("also") else      # combined: no test case to score
                                         pdd.accuracy(net, pdd.town_config(spec["town"]), ch, spec["time"]))),
        unsafe_allow_html=True)
    a, b = st.columns([1, 1.6], gap="medium")
    with a.container(border=True):
        st.markdown('<div class="card-title">Actions at affected junctions <span class="card-sub">affected: outside '
                    'the usual 10–90% range</span></div>', unsafe_allow_html=True)
        st.markdown(action_cards(r["actions"], pl), unsafe_allow_html=True)
    with b.container(border=True):
        st.markdown('<div class="card-title">Time lost at each affected junction <span class="card-sub">minutes '
                    'lost by drivers per 15 minutes, all together; shaded: while the road is set</span></div>',
                    unsafe_allow_html=True)
        if len(hit):
            curve = pdd.change_curve(r["slots"], hit.junction.tolist())
            chart(impact_curve_option(curve, pl, spec["time"], r["end"]), 280, f"icurve-{spec['town']}-{ch}")
            st.dataframe(hit.assign(junction=hit.junction.map(pl.node), lost=hit.extra_delay_veh_h.map(time_lost))[
                ["junction", "from_ts", "to_ts", "minutes", "lost"]].rename(columns={
                    "junction": "Junction", "from_ts": "From", "to_ts": "To", "minutes": "Minutes",
                    "lost": "Time lost by drivers"}),
                hide_index=True, width="stretch")
        else:
            st.write("No junction leaves its usual range.")


# ------------------------------------------------------------------ technical
@st.cache_resource
def scoring_model(name: str, first_test_day: str):
    """The graph model fitted on the days before the answer key's test days (as scripts/check_predict.py)."""
    from netstudy.graph_model import GraphModel

    return GraphModel.fit(pdd.town_config(name), network(name).dir, until=date.fromisoformat(first_test_day), town=name)


def _paired_runs(name, net, sim) -> None:
    """The graph model's road change next to SUMO's paired runs (whatif_labels): method accuracy in simulation."""
    cases = pdd.paired_cases(net)
    if ((sim or {}).get("source") or {}).get("kind") != "simulation" or cases.empty:
        st.info("No answer key: paired SUMO runs exist only in simulation, after `netstudy whatif-labels`.")
        return
    pl = ov.places(net)
    label = {r.case_id: f"{pl.road(r.edge)} {r.level} {r.start} for {r.minutes} min" for r in cases.itertuples()}
    cid = st.selectbox("Test case", cases.case_id, format_func=label.get, key="pt_case")
    case = cases.set_index("case_id").loc[cid].rename(cid)
    case["case_id"] = cid
    first = min(d for ds in cases.days for d in ds)
    model = scoring_model(name, first)
    r = pdd.paired_compare(net, model, case)
    st.caption(f"Graph model fitted on {len(model.days)} days before {first}; SUMO: change run − base run, mean of "
               f"{len(case.days)} test days ({', '.join(case.days)}); label_source sumo, scoring only. Shaded: while "
               "the road is set.")
    if len(r["scores"]):
        st.dataframe(r["scores"][["variant", "wape", "sign_ok", "hit", "missed", "false", "missed_edges", "false_edges",
                                  "over_true", "over_pred"]].round(2), hide_index=True, width="stretch")
    x = r["slots"]
    changed = (x.groupby("edge_id").d_flow.apply(lambda s: s.abs().max()) >= 2).loc[lambda s: s].index.tolist()
    roads = sorted(set(x.edge_id), key=lambda e: (e not in changed, pl.road(e)))
    edge = st.selectbox("Road", roads, format_func=lambda e: pl.road(e) + (" · changed in SUMO" if e in changed else ""),
                        key=f"pt_road_{cid}")
    g = x[x.edge_id == edge].sort_values("slot")
    chart({"tooltip": {"trigger": "axis"}, "legend": {"top": 0, "left": 0, "textStyle": {"color": ov.MUTED}},
           "grid": {"left": 48, "right": 16, "top": 40, "bottom": 28},
           "xAxis": {"type": "category", "data": g.slot.tolist(), "axisLabel": {"color": ov.MUTED}},
           "yAxis": {"type": "value", "name": "Δ veh / 5 min", "axisLabel": {"color": ov.MUTED},
                     "splitLine": {"lineStyle": {"color": ov.GRID}}},
           "series": [{"name": "SUMO paired runs", "type": "line", "data": g.d_flow.round(1).tolist(), "showSymbol": False,
                       "lineStyle": {"color": "#334155", "width": 2}, "itemStyle": {"color": "#334155"},
                       "markArea": {"itemStyle": {"color": "rgba(168,85,247,0.08)"},
                                    "data": [[{"xAxis": g.slot.min()}, {"xAxis": max(s for s in g.slot if s < r["end"])}]]}},
                      {"name": "Graph model", "type": "line", "data": g.p_flow.round(1).tolist(), "showSymbol": False,
                       "lineStyle": {"color": ov.ACCENT, "width": 2}, "itemStyle": {"color": ov.ACCENT}}]},
          280, f"pt-{cid}-{edge}")
    if len(r["roads"]):
        w = r["roads"].pivot_table(index="edge_id", columns="metric", values=["predicted", "observed"])
        w.columns = [f"{b}_{a}" for a, b in w.columns]
        st.dataframe(w.reset_index().assign(road=lambda t: t.edge_id.map(pl.road)).round(1), hide_index=True,
                     width="stretch")
        st.caption("Mean change while the road is set (veh per 5 minutes, km/h): predicted = graph model, observed = "
                   "SUMO's paired runs; signals variant.")


def technical() -> None:
    from netstudy import dashboard_data as dd

    name = choose_city(st.sidebar, page="ptech")
    net = network(name)
    tools, sim = tools_for(name), pdd.town_config(name)
    d, hhmm = choose_when(st.sidebar, st.sidebar, sim)
    change = st.session_state.get("p_change")
    st.title(f"Forecast · {city(name)['name']} · {d:%a %d %b %Y} {hhmm}")
    if tools is None:
        st.info("No graph model: the town has no camera-derived state yet.")
        return
    m = tools.model
    st.caption(f"Network `{name}` · graph model data/{name}/predict/graph_model.pkl · {len(m.days)} days "
               f"({m.days[0]} .. {m.days[-1]}) · label_source {m.label_source} · signals "
               f"{'from the town config' if m.signals else 'none (implicit capacity)'} · filmed: {pdd.covered(sim)}")
    tabs = st.tabs(["Paired runs", "Case", "Impact", "Expected state"])
    with tabs[0]:
        _paired_runs(name, net, sim)
    if not pdd.in_windows(sim, hhmm):
        for tab in tabs[1:]:
            tab.warning(f"{hhmm} is outside the filmed windows ({pdd.covered(sim)}): refused.")
        return
    kind, assumed = m.kind_for(d)
    tabs = tabs[1:]
    with tabs[0]:
        st.json({"date": d.isoformat(), "time": hhmm, "day_kind": days.day_kind(d), "model_uses": kind,
                 "assumed": assumed, "change": change}, expanded=2)
        st.caption("The road change is the case shown on the overview's map (none: day only).")
    with tabs[1]:
        if not change:
            st.info("No road change for this date and time (add a case on the overview's Forecast).")
        else:
            r = pdd.case_result(net, tools, d, hhmm, change)
            st.json(r["summary"], expanded=1)
            st.subheader("Junctions")
            st.dataframe(r["junction"], hide_index=True)
            st.caption("Affected: a road in leaves its usual 10–90% range of flow or speed (and by at least 1 vehicle "
                       "per 5 minutes or 3 km/h).")
            mm = net.edges()[["edge_id", "kind", "camera_id"]].merge(dd.delay_classes(r["edges"]), on="edge_id",
                                                                     how="left")
            st.plotly_chart(dd.map_figure(net, mm, "change", title="Extra delay per edge", colours=dd.CHANGE_COLOURS,
                                          legend=list(dd.CHANGE_COLOURS), hover=("extra_delay_veh_min", "extra_vehicles")),
                            width="stretch")
            st.subheader("Edges per 15 minutes")
            st.dataframe(r["per_slot"].round(1), hide_index=True)
    with tabs[2]:
        exp = pdd.expected(net, tools, d, hhmm)
        st.caption(f"Source: graph model, usual {exp['kind']} (median of its camera days)"
                   + (" · assumed: no recorded day of this kind" if exp["assumed"] else ""))
        st.plotly_chart(dd.map_figure(net, exp["state"], "speed_level"), width="stretch")
        st.dataframe(exp["state"][["edge_id", "kind", "flow", "speed", "free_kmh", "speed_rel", "speed_level"]]
                     .round(2), hide_index=True)
