"""Actions for affected junctions (docs/predict.md): plain rules on what a prediction shows.

  road closed or jammed     divert at its start junction along the shortest detour; signs upstream
  junction affected         signals: more green for its slowest road in; no signals: an officer
  large extra delay         officer on site (>= BIG_VEH_H), tow / clearance crew for a closure
  festival venue            drop-off zone and no parking on its streets; message signs on the way in
  usual peak hotspot        keep its slowest road clear (no stopping / loading) and watch the queue
"""
from __future__ import annotations

import heapq

import pandas as pd

from .places import Places, time_lost

BIG_VEH_H = 5.0
# a symbol for each kind of action (Material Symbols names, drawn by the dashboard), matched on its words in order
SYMBOLS = (("no action", "check_circle"), ("divert", "alt_route"), ("tow crew", "construction"),
           ("signal", "traffic"), ("longer green", "traffic"), ("extend green", "traffic"), ("message signs", "campaign"),
           ("drop-off", "local_parking"), ("officer", "local_police"), ("stewards", "local_police"),
           ("watch the queues", "visibility"))


def symbol(action: str) -> str:
    a = action.lower()
    return next((s for k, s in SYMBOLS if k in a), "task_alt")
PRIORITY = {"high": 0, "medium": 1, "low": 2}


def detour(edges: pd.DataFrame, kinds: dict, edge_id: str, avoid=()) -> list[str] | None:
    """Shortest path (by length) from the road's start to its end without it, through junctions only; `avoid`:
    other roads closed or jammed at the same time."""
    e = edges.set_index("edge_id")
    a, b = e.at[edge_id, "from_node"], e.at[edge_id, "to_node"]
    if kinds.get(a) == "boundary" or kinds.get(b) == "boundary":
        return None
    usable = edges[(edges.edge_id != edge_id) & ~edges.edge_id.isin(list(avoid)) & edges.from_node.map(kinds).eq("junction")
                   & edges.to_node.map(kinds).eq("junction")]
    out = {}
    for r in usable.itertuples():
        out.setdefault(r.from_node, []).append((r.length_m, r.to_node, r.edge_id))
    best, heap = {a: 0.0}, [(0.0, a, [])]
    while heap:
        d, n, path = heapq.heappop(heap)
        if n == b:
            return path
        for length, m, eid in out.get(n, []):
            if d + length < best.get(m, float("inf")):
                best[m] = d + length
                heapq.heappush(heap, (d + length, m, path + [eid]))
    return None


def _slowest_in(state: pd.DataFrame, edges: pd.DataFrame, junction: str, skip: str | None = None):
    """The road into the junction moving slowest against its free flow (`edges` with free_kmh, as
    dashboard_data.Network.edges())."""
    s = state[["edge_id", "speed", "total"]].merge(edges[["edge_id", "to_node", "free_kmh"]], on="edge_id")
    s = s[(s.to_node == junction) & (s.edge_id != skip) & s.speed.notna() & (s.free_kmh > 0)]
    if s.empty:
        return None
    return s.assign(rel=s.speed / s.free_kmh).sort_values("rel").iloc[0]


def for_change(pl: Places, edges: pd.DataFrame, nodes: pd.DataFrame, change: dict, junction: pd.DataFrame,
               state: pd.DataFrame) -> list[dict]:
    """Actions for a road set to a traffic level: `junction` from predict.impact, `state` the case's
    state at its worst slot."""
    kinds = dict(zip(nodes.node_id, nodes.kind))
    signal = dict(zip(nodes.node_id, nodes.signal))
    e = edges.set_index("edge_id").loc[change["edge"]]
    road, level = pl.road(change["edge"]), change["level"]
    t0, t1 = change["from"], change["to"]
    acts = []
    if level in ("closed", "jammed"):
        path = detour(edges, kinds, change["edge"], change.get("avoid", ()))
        via = " → ".join(dict.fromkeys(pl.node(edges.set_index("edge_id").at[p, "to_node"]) for p in path[:-1])) if path else ""
        acts.append({"junction": e.from_node, "priority": "high", "when": f"{t0}–{t1}",
                     "action": f"Divert traffic for {pl.node(e.to_node)} " + (f"via {via}" if via else "on other roads")
                               + f"; close the entry to {road}" * (level == "closed"),
                     "why": f"{road} is {level}"})
        if level == "closed":
            acts.append({"junction": e.from_node, "priority": "high", "when": f"from {t0}",
                         "action": "Send a clearance / tow crew and set up cones before the closed road",
                         "why": "a closure stops all traffic on the road"})
    up = edges[(edges.to_node == e.from_node) & edges.from_node.map(kinds).eq("junction")
               & (edges.from_node != e.to_node)].from_node.unique()
    if len(up):
        acts.append({"junction": e.from_node, "priority": "medium", "when": f"{t0}–{t1}",
                     "action": "Warn drivers on message signs at " + ", ".join(pl.node(u) for u in up)
                               + f": '{road} {level}, use other routes'",
                     "why": "drivers pick other routes before they reach the queue"})
    for r in junction[junction.affected == 1].itertuples():
        when = f"{r.from_ts}–{r.to_ts}"
        slow = _slowest_in(state, edges, r.junction, skip=change["edge"] if level == "closed" else None)
        big = abs(r.extra_delay_veh_h or 0) >= BIG_VEH_H
        less = (r.extra_delay_veh_h or 0) < 0
        if less:
            acts.append({"junction": r.junction, "priority": "low", "when": when,
                         "action": "No action: less traffic than usual arrives here while the road is "
                                   f"{level}; spare green time can go to the diverted traffic",
                         "why": f"drivers lose {time_lost(r.extra_delay_veh_h)} less here"})
            continue
        where = f" for {pl.road(slow.edge_id)}" if slow is not None else ""
        if signal.get(r.junction):
            acts.append({"junction": r.junction, "priority": "high" if big else "medium", "when": when,
                         "action": f"Retime the signals: more green{where} (+10–20 s per cycle)",
                         "why": f"drivers lose {time_lost(r.extra_delay_veh_h)} more on the roads into it"})
        else:
            acts.append({"junction": r.junction, "priority": "high" if big else "medium", "when": when,
                         "action": f"Station a traffic officer to give way{where}",
                         "why": f"no signals; drivers lose {time_lost(r.extra_delay_veh_h)} more on the roads "
                                "into it"})
        if big and signal.get(r.junction):
            acts.append({"junction": r.junction, "priority": "medium", "when": when,
                         "action": "Officer on site to keep the junction box clear",
                         "why": "queues this long block crossing traffic"})
    return sorted(acts, key=lambda a: PRIORITY[a["priority"]])


def for_day(pl: Places, edges: pd.DataFrame, nodes: pd.DataFrame, state: pd.DataFrame, when: str,
            venue: str | None = None, event: str | None = None, busier: pd.DataFrame | None = None) -> list[dict]:
    """Actions for a predicted time without a road change: hotspots (a road in at under half its
    free-flow speed), junctions busier than usual (holidays, festivals) and the festival venue."""
    signal = dict(zip(nodes.node_id, nodes.signal))
    acts = []
    hot = set()
    for j in nodes.node_id[nodes.kind == "junction"]:
        slow = _slowest_in(state, edges, j)
        if slow is not None and slow.rel < 0.5 and slow.total >= 5:
            hot.add(j)
            acts.append({"junction": j, "priority": "high" if slow.rel < 0.25 else "medium", "when": when,
                         "action": (f"Peak signal plan: longer green for {pl.road(slow.edge_id)}" if signal.get(j)
                                    else f"Officer to manage {pl.road(slow.edge_id)}")
                                   + "; no stopping or loading on it",
                         "why": f"{pl.road(slow.edge_id)} at {100 * slow.rel:.0f}% of free-flow speed"})
    if busier is not None and len(busier):
        for j, g in busier.groupby("junction"):
            if j in hot:
                continue
            acts.append({"junction": j, "priority": "medium", "when": f"{g.slot.min()}–{g.slot_end.max()}",
                         "action": "Watch the queues; be ready to extend green on its busiest road in",
                         "why": f"busier than on usual days of this type ({len(g) * 15} min above the usual range)"})
    if venue:
        acts.append({"junction": venue, "priority": "high", "when": when,
                     "action": f"Drop-off zone and no parking on {pl.node(venue)}'s streets; stewards at the crossing",
                     "why": f"{event} brings extra trips to {pl.node(venue)}"})
        near = edges[edges.to_node == venue].from_node
        near = [n for n in near.unique() if n in pl.junction]
        if near:
            acts.append({"junction": venue, "priority": "medium", "when": when,
                         "action": "Message signs at " + ", ".join(pl.node(n) for n in near)
                                   + f": '{event}: expect delays near {pl.node(venue)}'",
                         "why": "spreads arrivals over other routes"})
    return sorted(acts, key=lambda a: PRIORITY[a["priority"]])
