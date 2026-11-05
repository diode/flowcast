"""What both pages share: the cached network, the city / day / time choices (kept when switching
page), and the overview's style."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from netstudy import dashboard_data as dd
from netstudy.places import city_name

DEFAULT_CITY, DEFAULT_DAY, DEFAULT_TIME = "maple_falls", "2026-10-06", "18:00"
TIMES = [f"{m // 60:02d}:{m % 60:02d}" for m in range(5, 24 * 60 + 1, 5)]


@st.cache_resource
def network(name: str) -> dd.Network:
    return dd.Network(name)


def _keep(key: str, widget_key: str) -> None:
    st.session_state[key] = st.session_state[widget_key]


def choose_city(where=st.sidebar, page: str = "overview") -> str:
    names = dd.networks()
    if not names:
        st.error("No network results found in data/ (run netstudy network / simulate / state / kpis first).")
        st.stop()
    cur = st.session_state.get("city")
    if cur is None and st.query_params.get("city") in names:     # a link such as ?city=westbrook
        cur = st.session_state["city"] = st.query_params["city"]
    if cur not in names:
        cur = DEFAULT_CITY if DEFAULT_CITY in names else names[0]
        st.session_state["city"] = cur
    wk = f"city_{page}"
    return where.selectbox("City", names, index=names.index(cur), format_func=city_name, key=wk,
                           on_change=_keep, args=("city", wk))


MODES = ["Observed", "Forecast"]
OLD_MODES = {"Retrospective": "Observed", "Predict": "Forecast"}     # links made before the rename
KEY_HELP = ("**Muse Spark needs a Meta Model API key.** Create one at [ai.developer.meta.com](https://ai.developer.meta.com), "
            "then either\n\n"
            "- set it for your Windows account: run `setx MODEL_API_KEY \"your-key\"` in a terminal, or\n"
            "- put `MODEL_API_KEY = \"your-key\"` in `.streamlit/secrets.toml` (next to `config.toml`)\n\n"
            "and restart the dashboard. Questions and the model's results are then sent to Meta.")


def choose_mode(where=st, page: str = "overview") -> str:
    """Observed (the recorded days) or Forecast (a future date), kept across pages; ?mode= also takes the old
    names (Retrospective, Predict)."""
    wk = f"mode_{page}"
    q = OLD_MODES.get(st.query_params.get("mode"), st.query_params.get("mode"))
    if q in MODES and "mode" not in st.session_state:
        st.session_state["mode"] = q
    cur = OLD_MODES.get(st.session_state.get("mode"), st.session_state.get("mode")) or MODES[0]   # a click on the
    if cur not in MODES:                                                                        # selected one clears it
        cur = MODES[0]
    mode = where.segmented_control("View", MODES, default=cur, key=wk, label_visibility="collapsed",
                                   on_change=_keep, args=("mode", wk))
    return mode or cur


@st.cache_data(show_spinner=False)
def day_times(name: str, day: str) -> list[str]:
    """The slots of `day` with network results, as slot-end HH:MM ("24:00" = midnight after it): a filmed day of a
    study town has only its windows."""
    net = network(name)
    ts = net.q("SELECT DISTINCT ts FROM network_kpi WHERE ts > ? AND ts <= ? ORDER BY ts", dd._day_range(day)).ts
    start = pd.Timestamp(day)
    return [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in ((pd.to_datetime(ts) - start).dt.total_seconds() // 60)]


def _minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def period_name(hhmm: str) -> str:
    h = _minutes(hhmm) / 60
    return ("Night" if h < 5 else "Morning" if h < 11 else "Noon" if h < 14 else "Afternoon" if h < 17
            else "Evening" if h < 21 else "Night")


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


DAY_PERIODS = [("Late night", "00:00", "05:00", "03:00"), ("Morning", "05:00", "11:00", "08:00"),   # name, from, to,
               ("Noon", "11:00", "14:00", "12:30"), ("Afternoon", "14:00", "17:00", "15:30"),    # the time shown
               ("Evening", "17:00", "21:00", "18:00"), ("Night", "21:00", "24:00", "22:00")]     # when picked


def day_periods(times: list[str], windows: list[dict], step: int = 5) -> dict[str, list[str]]:
    """The periods of a day with results, {"Morning 08:00–08:30": [slot-end times], …}, the time shown first when
    the period is picked: the town's filmed windows when it has them (every day, the study's times), else the
    recorded stretches of a partly recorded day, else the parts of the day (DAY_PERIODS)."""
    out = {}
    if windows:
        for w in windows:
            ts = [t for t in times if w["start"] < t <= w["end"]]
            if ts:
                out[f"{w['period'].capitalize()} {w['start']}–{w['end']}"] = ts
        if out:
            return out
    if len(times) * step < 20 * 60:                       # partly recorded: its stretches, named by the time of day
        runs: list[list[str]] = []
        for t in times:
            if runs and _minutes(t) - _minutes(runs[-1][-1]) == step:
                runs[-1].append(t)
            else:
                runs.append([t])
        for r in runs:
            a = _hhmm(_minutes(r[0]) - step)
            out[f"{period_name(a)} {a}–{r[-1]}"] = r
        return out
    for name, a, b, show in DAY_PERIODS:
        ts = [t for t in times if a < t <= b]
        if ts:
            out[f"{name} {a}–{b}"] = sorted(ts, key=lambda t: t != show)     # `show` first (when it has results)
    return out


def _pick_period(key: str, state_key: str, periods: dict[str, list[str]]) -> None:
    if st.session_state.get(key):                       # a click on the picked one clears it: keep the time
        st.session_state[state_key] = periods[st.session_state[key]][0]


def choose_period(where, periods: dict[str, list[str]], cur: str, label: str, key: str, state_key: str) -> str:
    """One button per period: picking one moves the time (st.session_state[state_key]) to its first time; the
    time is kept while it is inside a period (a case at 08:15), else it moves to the start of the nearest one."""
    sel = next((p for p, ts in periods.items() if min(ts) <= cur <= max(ts)), None)
    if sel is None:
        sel = min(periods, key=lambda p: abs(_minutes(periods[p][0]) - _minutes(cur)))
        name, span = sel.rsplit(" ", 1)
        where.caption(f"{cur} is not recorded: showing {name} ({span}).")
        cur = periods[sel][0]
    if st.session_state.get(key) != sel:
        st.session_state[key] = sel
    where.segmented_control(label, list(periods), key=key, on_change=_pick_period, args=(key, state_key, periods),
                            format_func=lambda p: period_label(p, periods))
    return cur


def period_label(p: str, periods: dict) -> str:
    """A period's button: its name only ("Morning"); with its times when two periods share the name."""
    name = p.rsplit(" ", 1)[0]
    return p if sum(q.rsplit(" ", 1)[0] == name for q in periods) > 1 else name


def choose_time(net: dd.Network, day_where=st.sidebar, time_where=st.sidebar, page: str = "overview") -> tuple[str, str]:
    """Day and time: one button per period with results that day (Morning, Evening, …; day_periods)."""
    days = net.days()
    cur = st.session_state.get("day")
    if cur not in days:
        cur = DEFAULT_DAY if DEFAULT_DAY in days else days[0]
        st.session_state["day"] = cur
    dk = f"day_{page}"
    day = day_where.selectbox("Day", days, index=days.index(cur), key=dk, on_change=_keep, args=("day", dk),
                              format_func=lambda d: f"{pd.Timestamp(d):%a %d %b %Y}")
    times = day_times(net.name, day) or TIMES
    want = st.session_state.get("hhmm", DEFAULT_TIME)
    periods = day_periods(times, town_windows(net.name))
    hhmm = st.session_state["hhmm"] = choose_period(time_where, periods, want, "Time", f"period_{page}", "hhmm")
    time_where.caption(f"Showing the 5 minutes ending {hhmm}.")
    return day, hhmm


@st.cache_data(show_spinner=False)
def town_windows(name: str) -> list[dict]:
    """The town's filmed windows (recording.windows) for naming periods; none for a city without a town config."""
    from netstudy import assist, predict_data

    return assist.windows(predict_data.town_config(name))


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stMarkdown, .stMarkdown p, label, input { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 3.8rem; max-width: 1400px; }
div[data-testid="stButtonGroup"] button { font-weight: 600; padding: 4px 18px; }
:root { --ink:#1f2937; --muted:#6b7280; --line:#eef1f5; --card:#ffffff; --soft:#f6f8fb; --accent:#0f766e;
  --shadow-sm: 0 1px 2px rgba(16,24,40,.05), 0 1px 4px rgba(16,24,40,.04);
  --shadow: 0 1px 2px rgba(16,24,40,.04), 0 2px 8px rgba(16,24,40,.05); }
/* borderless: cards, tiles and controls stand out by a light shadow or a tinted surface, not a line */
div[data-testid="stExpander"] details { border: none !important; background: var(--soft); border-radius: 12px; }
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
.hero { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap;
  padding: 26px 30px; border-radius: 20px; margin: 4px 0 18px;
  background: radial-gradient(1200px 300px at 85% -40%, rgba(45,212,191,.35), transparent 60%),
              linear-gradient(135deg, #0b1626 0%, #12304a 55%, #0f766e 130%); color: #f8fafc; }
.hero-kind { text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 600; color: #99f6e4; }
.hero-name { font-size: 40px; font-weight: 800; line-height: 1.1; margin: 4px 0 6px; letter-spacing: -.02em; }
.hero-tag { color: #cbd5e1; font-size: 15px; }
.hero-live { display:flex; align-items:center; gap:8px; font-size: 13px; color:#e2e8f0;
  background: rgba(255,255,255,.12); padding: 7px 12px; border-radius: 999px; }
.pulse { width:8px; height:8px; border-radius:50%; background:#34d399; box-shadow:0 0 0 0 rgba(52,211,153,.7);
  animation: pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(52,211,153,.6);} 70% { box-shadow:0 0 0 10px rgba(52,211,153,0);}
  100% { box-shadow:0 0 0 0 rgba(52,211,153,0);} }
.tiles { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin-bottom: 16px; }
@media (max-width: 1100px) { .tiles { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 640px) { .tiles { grid-template-columns: 1fr; } .hero-name { font-size: 30px; } }
.tile { background: var(--card); border-radius: 16px; padding: 16px 18px; box-shadow: var(--shadow); }
.tile-label { color: var(--muted); font-size: 13px; font-weight: 600; }
.tile-row { display:flex; justify-content:space-between; align-items:flex-end; gap: 8px; margin-top: 6px; }
.tile-value { font-size: 32px; font-weight: 800; color: var(--ink); letter-spacing: -.02em; }
.tile-unit { font-size: 15px; color: var(--muted); margin-left: 4px; font-weight: 600; }
.tile-delta { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.delta { font-weight: 700; padding: 1px 6px; border-radius: 6px; }
.delta.bad { color:#b42318; background:#fef3f2; } .delta.good { color:#067647; background:#ecfdf3; }
.delta.flat { color:#475467; background:#f2f4f7; }
.tile-note { font-size: 12px; color: var(--muted); margin-top: 10px; }
.card-title { font-weight: 700; color: var(--ink); font-size: 16px; margin: 2px 0 6px; }
.card-sub { display: block; font-weight: 500; color: var(--muted); font-size: 12.5px; margin: 2px 0 0; }
.insights { display:flex; flex-direction:column; gap: 10px; }
.insight { display:flex; gap: 12px; padding: 12px 14px; border-radius: 12px; background: var(--soft);
  color: var(--ink); font-size: 14px; line-height: 1.45; }
.insight.stack { flex-direction: column; gap: 8px; }
.act { display:flex; gap: 10px; align-items: flex-start; }
.sym { font-family: 'Material Symbols Rounded'; font-size: 19px; line-height: 1; width: 32px; height: 32px; flex-shrink: 0;
  display:inline-flex; align-items:center; justify-content:center; border-radius: 10px; }
.sym.high { background:#fde8e6; color:#b42318; } .sym.medium { background:#fdf0d5; color:#b54708; }
.sym.low { background:#dcf5e7; color:#067647; }
.sym.info { background:#e6f4f1; color:#0f766e; }
.pri-chip { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 999px; margin-left: 4px; vertical-align: 1px; }
.pri-chip.high { background:#fde8e6; color:#b42318; } .pri-chip.medium { background:#fdf0d5; color:#b54708; }
.pri-chip.low { background:#dcf5e7; color:#067647; }
.insight.muted { color: var(--muted); }
.insight-icon { font-size: 20px; line-height: 1.2; }
.hl { font-weight: 700; color: #b42318; }
.spacer { height: 6px; }
</style>
"""
