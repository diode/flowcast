"""City overview: the customer-facing page (docs/dashboard.md); Predict is in predict_view.py.

Observed: headline tiles against the usual day, a live map with moving traffic, the city's flow
score, the day's rhythm, the slowest roads, the weekly pattern and what the studies found. Charts are
Apache ECharts (streamlit-echarts, helpers in widgets.py); numbers come from netstudy.overview.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from netstudy import overview as ov  # noqa: E402
from netstudy.places import city, impact_word, time_lost  # noqa: E402
from shared import STYLE, choose_city, choose_mode, choose_time, network  # noqa: E402
from widgets import chart, tile  # noqa: E402

st.markdown(STYLE, unsafe_allow_html=True)
bar = st.container(border=True)                    # the view and its choices, on one card
if choose_mode(bar) == "Forecast":                 # the observed view below is unchanged
    import predict_view

    predict_view.overview(bar)
    st.stop()


# ------------------------------------------------------------------ choices
c = bar.columns([1.1, 1.1, 2.8], vertical_alignment="bottom")
name = choose_city(c[0])
net = network(name)
info = city(name)
day, hhmm = choose_time(net, c[1], c[2])
pl = ov.places(net)
k = ov.all_kpis(net)

st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-kind">{info['kind']}</div>
    <div class="hero-name">{info['name']}</div>
    <div class="hero-tag">{' · '.join(x for x in (info['tagline'], f"{int((net.nodes().kind == 'junction').sum())} junctions", f"{len(net.edges())} roads") if x)}</div>
  </div>
  <div class="hero-live"><span class="pulse"></span>{'Measured from the cameras' if info.get('real') else 'Simulated live view'}</div>
</div>""", unsafe_allow_html=True)

ts = net.slot(day, hhmm)
state = net.state_at(ts)
kpi = net.kpi_at(ts)
h = ov.headline(net, k, state, day, hhmm)
weekday = f"{pd.Timestamp(day):%A}"

# ------------------------------------------------------------------ tiles
fmt = lambda x, d=0: "–" if pd.isna(x) else f"{x:,.{d}f}"  # noqa: E731
tiles = [
    tile("Vehicles on the roads", fmt(h["vehicles"]["now"]), "", h["vehicles"]["now"], h["vehicles"]["usual"],
         h["vehicles"]["spark"], True, f"Usual {weekday} at {hhmm}: {fmt(h['vehicles']['usual'])}"),
    tile("Extra travel time", fmt(h["travel"]["now"]), "%", h["travel"]["now"], h["travel"]["usual"],
         h["travel"]["spark"], True, "How much longer trips take than on empty roads"),
    tile("Road length congested", fmt(h["congested"]["now"], 1), "%", h["congested"]["now"], h["congested"]["usual"],
         h["congested"]["spark"], True, "Share of the network's length that is slow or jammed"),
    tile("Time lost in traffic today", fmt(h["delay"]["now"], 1), "h", h["delay"]["now"], h["delay"]["usual"],
         h["delay"]["spark"], True, f"Since midnight, all drivers together · avg speed {fmt(ov.avg_speed_kmh(state))} km/h"),
]
st.markdown('<div class="tiles">' + "".join(tiles) + "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ map + score
left, right = st.columns([2.1, 1], gap="medium")
with left.container(border=True):
    st.markdown(f'<div class="card-title">Road map <span class="card-sub">{weekday} {hhmm} · '
                'dots move at the speed traffic flows</span></div>', unsafe_allow_html=True)
    chart(ov.map_option(net, state, pl, aspect=1.75), 470, f"map-{name}-{ts}")
score = ov.flow_score(kpi.speed_index if kpi is not None else float("nan"))
with right.container(border=True):
    st.markdown('<div class="card-title">City flow score <span class="card-sub">100 = free flow everywhere</span></div>',
                unsafe_allow_html=True)
    chart(ov.gauge_option(score), 200, f"gauge-{name}-{ts}")
with right.container(border=True):
    st.markdown('<div class="card-title">Roads by condition</div>', unsafe_allow_html=True)
    chart(ov.donut_option(ov.road_levels(state)), 180, f"donut-{name}-{ts}")

# ------------------------------------------------------------------ rhythm + slowest
a, b = st.columns(2, gap="medium")
ph, pv = ov.peak_hour(k, day)
with a.container(border=True):
    st.markdown(f'<div class="card-title">Daily rhythm <span class="card-sub">vehicles on the roads · busiest hour '
                f'{ph:02d}:00–{ph + 1:02d}:00</span></div>', unsafe_allow_html=True)
    chart(ov.rhythm_option(k, day, hhmm), 300, f"rhythm-{name}-{day}-{hhmm}")
with b.container(border=True):
    st.markdown('<div class="card-title">Slowest roads right now <span class="card-sub">speed as % of free flow'
                '</span></div>', unsafe_allow_html=True)
    chart(ov.ranking_option(ov.slowest_roads(state, pl)), 300, f"rank-{name}-{ts}")

# ------------------------------------------------------------------ weekly + insights
WEEKDAY = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday",
           "Sat": "Saturday", "Sun": "Sunday"}


AT = {"morning": "in the morning", "noon": "at noon", "evening": "in the evening", "night": "at night"}


def finding(symbol: str, tone: str, title: str, body: str) -> str:
    return (f'<div class="insight"><span class="sym {tone}">{symbol}</span><div><b>{title}</b><br/>{body}'
            "</div></div>")


def study_cards(net, info, k) -> list[str]:
    """The town's study results in plain words: the closure study (every road closed for an hour), the busiest
    hour, the prediction and camera checks, and the older incident, forecast and what-if studies where run."""
    cards = []
    cs = ov.closure_study(net)
    if cs is not None:
        worst = cs.sort_values("extra_delay_veh_h", ascending=False).drop_duplicates("edge").head(3)
        cards.append(finding("warning", "high", "Most critical roads", "Closing one for an hour on a working day "
                             "costs drivers the most time on " + "; ".join(
                                 f"<b>{r.road}</b> ({r.period}: {time_lost(r.extra_delay_veh_h)}, "
                                 f"{impact_word(r.extra_delay_veh_h).lower()} impact)" for r in worst.itertuples())
                             + "."))
        per = cs.groupby("period").extra_delay_veh_h.median()
        calm, busy = per.idxmin(), per.idxmax()
        cards.append(finding("construction", "low", "Best time for roadworks",
                             f"<b>{calm.capitalize()}</b>: closing a road then costs drivers "
                             f"{time_lost(per[calm])} on a typical road, against {time_lost(per[busy])} "
                             f"{AT.get(busy, 'in the ' + busy)}."))
    bh = ov.busiest_hour(k)
    if bh and bh[2] > 0:
        cards.append(finding("schedule", "medium", "Slowest hour of the week",
                             f"{WEEKDAY[bh[0]]} {bh[1]:02d}:00–{bh[1] + 1:02d}:00: drivers lose "
                             f"{time_lost(bh[2])} in that hour on average."))
    pa = ov.predict_accuracy(net)
    if pa:
        parts = []
        if "found" in pa:
            parts.append(f"for {pa['cases']} simulated road changes it found {pa['found']} of the {pa['affected']} "
                         f"roads really affected, with {pa['false'] or 'no'} false alarm{'s' * (pa['false'] != 1)}")
        if "in_range" in pa:
            parts.append(f"{100 * pa['in_range']:.0f}% of the real traffic on test days fell inside its usual range")
        cards.append(finding("verified", "info", "Forecasts checked", "Against test runs: " + "; ".join(parts)
                             + "."))
    ca = ov.camera_accuracy(net)
    if ca:
        cards.append(finding("videocam", "info", "Reading the cameras",
                             f"Counting traffic from the video had a typical error of {100 * ca['flow_wape']:.0f}% per "
                             f"5 minutes against a perfect count, and speeds were within {ca['speed_mae']:.0f} km/h "
                             f"({pd.Timestamp(ca['day']):%d %b}, the filmed day)."))
    det = ov.detection_summary(net)
    if det:
        cards.append(finding("sensors", "info", "Closures spotted automatically",
                             f"{det['found']} of {det['closures']} test closures found from the cameras alone, "
                             f"typically {det['minutes']:.0f} minutes after the road shut."))
    acc = ov.forecast_accuracy(net)
    if acc:
        m = min(acc)
        cards.append(finding("trending_up", "info", "Traffic forecast",
                             f"Vehicle counts {m} minutes ahead are {acc[m]:.0f}% accurate"
                             + (f", {acc[max(acc)]:.0f}% at {max(acc)} minutes." if len(acc) > 1 else ".")))
    for c in ov.whatif_cards(net):
        where = ", ".join(c["districts"]) or "no district clearly affected"
        cards.append(finding("science", "medium", f"What if: {c['title']}",
                             f"{c['sub']}. <span class=\"hl\">{impact_word(c['extra_h'])} impact</span>: drivers "
                             f"lose {time_lost(c['extra_h'])} more; affected: {where}."))
    if not cards:
        cards.append(f'<div class="insight muted"><div class="insight-icon">⏳</div><div>No studies have been run '
                     f'for {info["name"]} yet. The technical view lists what is available.</div></div>')
    return cards


# ------------------------------------------------------------------ weekly + insights
a, b = st.columns([1.35, 1], gap="medium")
with a.container(border=True):
    st.markdown('<div class="card-title">When delays happen <span class="card-sub">average time lost per hour, '
                f'{len(k.date.unique())} days</span></div>', unsafe_allow_html=True)
    chart(ov.heatmap_option(ov.weekly_profile(k)), 300, f"heat-{name}")
with b.container(border=True):
    st.markdown('<div class="card-title">What our studies found</div>', unsafe_allow_html=True)
    st.markdown('<div class="insights">' + "".join(study_cards(net, info, k)) + "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ footer
c1, c2 = st.container(border=True).columns([3, 1], vertical_alignment="center")
c1.caption(f"{info['name']} is {info['source']}. Every number comes from "
           f"{'its camera recordings' if info.get('real') else 'simulated traffic'} over {len(net.days())} days; "
           "the map shows the roads with a camera.")
c2.page_link("technical.py", label="Open the technical view", icon=":material/arrow_forward:")
