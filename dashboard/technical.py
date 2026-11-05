"""Technical view (docs/phase5.md): the stored results of phases 1b-4 for a chosen time, per edge,
with the ids the data uses. Opened from the city overview.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from netstudy import dashboard_data as dd  # noqa: E402
from netstudy.places import city_name  # noqa: E402
from shared import choose_city, choose_mode, choose_time, network  # noqa: E402

# ------------------------------------------------------------------ sidebar
st.sidebar.page_link("overview.py", label="Back to the city overview", icon=":material/arrow_back:")
st.sidebar.title("Technical view")
if not dd.networks():                             # no town with results yet: bring one in
    import onboarding_view

    onboarding_view.panel()
    st.stop()
if choose_mode(page="technical") == "Forecast":    # the observed view below is unchanged
    import predict_view

    predict_view.technical()
    st.stop()
name = choose_city(st.sidebar, page="technical")
net = network(name)
day, hhmm = choose_time(net, st.sidebar, st.sidebar, page="technical")
view = st.sidebar.radio("View", ["True state", "Cameras"], help="True state: every edge, from SUMO. "
                        "Cameras: what the system knows (camera edges; estimated edges dashed; the rest grey).")
colour_by = st.sidebar.radio("Colour edges by", ["Level (load)", "Speed vs free flow"])
ts = net.slot(day, hhmm)
at = pd.Timestamp(ts)

st.title(f"{city_name(name)}  ·  {pd.Timestamp(day):%a %d %b %Y}  {hhmm}")
st.caption(f"Network `{name}` · data/{name}/")
tabs = st.tabs(["Network", "Timeline", "Edge", "Incidents", "What-if", "Forecast", "Onboarding & ingest"])


def colour_column(state: pd.DataFrame) -> tuple[pd.DataFrame, str, set[str]]:
    """The column the map colours by, for the chosen view, and the edges to draw dashed."""
    s = state.copy()
    if view == "Cameras":
        s["show"] = s.cam_level.where(s.observed.notna() & (s.method.notna()) & s.cam_level.notna())
        dashed = set(s.edge_id[s.method.eq("M1")])
        if colour_by == "Speed vs free flow":       # cameras give no speed
            s["show"] = None
    else:
        s["show"] = s.level if colour_by == "Level (load)" else s.speed_level
        dashed = set()
    return s, "show", dashed


# ------------------------------------------------------------------ Network
with tabs[0]:
    state = net.state_at(ts)
    kpi = net.kpi_at(ts)
    if kpi is not None:
        c = st.columns(5)
        c[0].metric("Vehicles on the network", f"{kpi.veh_in_town:.0f}")
        c[1].metric("Speed index", "–" if pd.isna(kpi.speed_index) else f"{kpi.speed_index:.2f}",   # none: no
                    help="Speed against free flow, weighted by vehicle-hours")                       # vehicle-hours
        c[2].metric("Share of length congested", f"{kpi.share_congested:.1%}")
        c[3].metric("Delay (veh-h per 5 min)", f"{kpi.delay_veh_h:.1f}")
        c[4].metric("Edges with an incident", f"{kpi.incident_edges:.0f}")
    s, col, dashed = colour_column(state)
    if view == "Cameras" and colour_by == "Speed vs free flow":
        st.info("Cameras measure counts and queues, not speed: switch to Level, or to the true state.")
    st.plotly_chart(dd.map_figure(net, s, col, dashed), width="stretch")
    n_cam = int(state.camera_id.notna().sum())
    st.caption(f"{len(state)} edges, {n_cam} with a camera. Hover an edge for its numbers.")
    worst = state.dropna(subset=["speed_rel"]).nsmallest(5, "speed_rel")
    st.subheader("Slowest edges (true speed against free flow)")
    st.dataframe(worst[["edge_id", "kind", "camera_id", "total", "flow", "speed", "free_kmh", "speed_rel", "level",
                        "loss_min", "incident"]].round(2), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Timeline
with tabs[1]:
    k = net.kpi_day(day)
    inc = net.incident_slots(day)
    st.plotly_chart(dd.timeline_figure(k, at, inc), width="stretch")
    if len(inc):
        st.caption("Shaded: random incidents on " + ", ".join(f"{r.edge_id} {r.start:%H:%M}-{r.end:%H:%M}"
                                                            for r in inc.itertuples()))
    else:
        st.caption("No random incident on this day.")

# ------------------------------------------------------------------ Edge
with tabs[2]:
    edges = net.edges()
    default = state.dropna(subset=["speed_rel"]).nsmallest(1, "speed_rel").edge_id
    edge_ids = edges.edge_id.tolist()
    edge = st.selectbox("Edge", edge_ids, index=edge_ids.index(default.iloc[0]) if len(default) else 0,
                        help="Defaults to the slowest edge at the chosen time")
    e = edges.set_index("edge_id").loc[edge]
    st.caption(f"{e.kind}, {e.from_node} → {e.to_node}, {e.length_m:.0f} m, {e.lanes} lanes, "
               f"free flow {e.free_kmh:.0f} km/h, camera: {e.camera_id if isinstance(e.camera_id, str) else 'none'}")
    st.plotly_chart(dd.edge_figure(net.edge_day(edge, day), at), width="stretch")

# ------------------------------------------------------------------ Incidents
with tabs[3]:
    cl = net.closures()
    if cl.empty:
        st.info(f"No phase 2 closure runs for {name} (run `netstudy incidents` on it).")
    else:
        inc_id = st.selectbox("Closure run", cl.incident_id.tolist(),
                              format_func=lambda i: f"{i}  ({cl.set_index('incident_id').at[i, 'day']})")
        r = cl.set_index("incident_id").loc[inc_id]
        fp = net.footprint(inc_id)
        ev, sp = net.detections(inc_id)
        a, b = st.columns([3, 2])
        with a:
            m = net.edges()[["edge_id", "kind", "camera_id"]].merge(fp[["edge_id", "effect", "extra_delay_veh_h", "flow_change",
                                                                       "delay_min"]], on="edge_id", how="left")
            m["effect"] = m.effect.where(m.edge_id != r.edge_id, "closed")
            colours = {"closed": "#202124", "slower": "#c5221f", "more": "#e8710a", "fewer": "#1a73e8", "faster": "#7baaf7"}
            st.plotly_chart(dd.map_figure(net, m, "effect", title=f"Footprint: {r.edge_id} closed {r.start[11:16]}-{r.end[11:16]}",
                                          colours=colours, legend=list(colours),
                                          hover=("extra_delay_veh_h", "flow_change", "delay_min")), width="stretch")
            st.caption("Footprint: edges changed beyond twice the largest placebo change (paired runs, phase 2). "
                       "Grey: no clear change.")
        with b:
            st.subheader("Detected from the cameras")
            during = ev[(pd.to_datetime(ev.known) > pd.Timestamp(r.start)) & (pd.to_datetime(ev.start) <= pd.Timestamp(r.end))] \
                if len(ev) else ev
            if during.empty:
                st.write("No event during the closure.")
            else:
                for x in during.itertuples():
                    mins = (pd.Timestamp(x.known) - pd.Timestamp(r.start)).total_seconds() / 60
                    st.write(f"**{x.origin}** dead from {x.start[11:16]}, known {x.known[11:16]} "
                             f"({mins:.0f} min after the closure began)" + (" ✓ the closed road" if x.origin == r.edge_id else ""))
                spread = sp[sp.event_id.isin(during.event_id)]
                if len(spread):
                    st.dataframe(spread[["edge_id", "effect", "first", "delay_min"]], hide_index=True)
            st.subheader("Footprint (truth)")
            st.dataframe(fp[["edge_id", "hop", "effect", "first_ts", "delay_min", "extra_delay_veh_h", "flow_change"]]
                         .sort_values("hop"), hide_index=True)

# ------------------------------------------------------------------ What-if
with tabs[4]:
    scen = net.scenarios()
    if not scen:
        st.info(f"No phase 4 scenarios for {name} (run `netstudy whatif --scenario <name>`).")
    else:
        sc = st.selectbox("Scenario", scen)
        junction, per_edge = net.scenario_impact(sc)
        a, b = st.columns([2, 3])
        with a:
            st.subheader("Junctions")
            st.dataframe(junction.drop(columns=["scenario", "limit_veh_min"]), hide_index=True)
            st.caption("Affected: extra delay on its roads in beyond twice the placebo's largest (mean of 5 days).")
        with b:
            m = net.edges()[["edge_id", "kind", "camera_id"]].merge(dd.delay_classes(per_edge), on="edge_id", how="left")
            st.plotly_chart(dd.map_figure(net, m, "change", title="Extra delay per edge (mean of 5 days)",
                                          colours=dd.CHANGE_COLOURS, legend=list(dd.CHANGE_COLOURS),
                                          hover=("extra_delay_veh_min", "extra_vehicles")), width="stretch")
        sdays = net.scenario_days(sc)
        if sdays:
            st.subheader(f"Base and scenario at {hhmm}")
            sday = st.selectbox("Day", sdays, format_func=lambda d: f"{pd.Timestamp(d):%a %d %b}", key="whatif_day")
            base, run = net.scenario_state(sc, sday, hhmm)
            key = "level" if colour_by == "Level (load)" else "speed_level"
            c1, c2 = st.columns(2)
            c1.plotly_chart(dd.map_figure(net, base, key, title="Base (as it was)"), width="stretch")
            c2.plotly_chart(dd.map_figure(net, run, key, title=f"With {sc}"), width="stretch")

# ------------------------------------------------------------------ Forecast
with tabs[5]:
    f = net.forecast_scores()
    if f.empty:
        st.info(f"No phase 3 forecast results for {name} (run scripts/check_forecast.py).")
    else:
        test = st.radio("Test", sorted(f.test.unique()), horizontal=True,
                        format_func=lambda t: {"holdout": "Last 7 days", "incident": "Around closures"}.get(t, t))
        g = f[f.test == test]
        models = ["v1", "network", "persistence", "same_time_yesterday"]
        colours = {"v1": "#1a73e8", "network": "#e8710a", "persistence": "#9aa0a6", "same_time_yesterday": "#5f6368"}
        c = st.columns(2)
        for i, target in enumerate(sorted(g.target.unique())):
            t = g[g.target == target]
            fig = go.Figure()
            for m_ in models:
                x = t[t.model == m_].sort_values("minutes_ahead")
                fig.add_trace(go.Bar(x=x.minutes_ahead.astype(str) + " min", y=x.wape, name=m_, marker_color=colours[m_],
                                     hovertemplate=f"{m_}: %{{y:.3f}}<extra></extra>"))
            fig.update_layout(title=f"{target}: error (WAPE, lower is better)", barmode="group", height=380,
                              margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15))
            c[i % 2].plotly_chart(fig, width="stretch")
        st.dataframe(g.pivot_table(index=["target", "minutes_ahead"], columns="model", values="wape")[models].round(3))

# ------------------------------------------------------------------ Onboarding & ingest
with tabs[6]:
    import onboarding_view

    onboarding_view.panel(name)
