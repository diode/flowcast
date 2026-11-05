"""Display names for the customer dashboard: each network as a town, its junctions as districts or
crossings, its roads as streets. Ids in the data stay as they are (J1_J2, A0B0, -43); only labels change.

  maple_falls  (was town5)   our five-junction town
  westbrook    (was grid)    v1's 3 x 2 signalised grid
  harbor_city  (was Town05)  v1's CARLA Town05
"""
from __future__ import annotations

import re
from pathlib import Path

TOWNS = Path(__file__).resolve().parents[2] / "towns"

CITIES = {
    "maple_falls": {"name": "Maple Falls", "tagline": "Five neighbourhoods around a busy downtown",
                    "kind": "Town network", "source": "our own SUMO town", "was": "town5"},
    "westbrook": {"name": "Westbrook", "tagline": "A signalised street grid with ten gateways",
                  "kind": "Grid district", "source": "v1's signalised grid", "was": "grid"},
    "harbor_city": {"name": "Harbor City", "tagline": "A compact waterfront city inside two ring roads",
                    "kind": "City network", "source": "v1's CARLA Town05 map", "was": "Town05"},
}

# Maple Falls: J2 is the city centre, J1 the residential area (sim/maple_falls.yaml).
MAPLE_DISTRICTS = {"J1": "Oakwood", "J2": "Downtown", "J3": "Riverside", "J4": "Hillcrest", "J5": "Millbrook"}
MAPLE_STREETS = {frozenset(p): s for p, s in [
    (("J1", "J2"), "Oakwood Avenue"), (("J2", "J3"), "River Road"), (("J2", "J4"), "Hill Street"),
    (("J2", "J5"), "Mill Lane"), (("J1", "J4"), "West Parkway"), (("J4", "J5"), "North Boulevard"),
    (("J5", "J3"), "East Parkway"), (("J3", "J1"), "South Boulevard")]}

# Westbrook: columns A-C are avenues, rows 0-1 streets; gateways sit on the grid's edge.
WEST_AVENUES = {"A": "Ash Avenue", "B": "Birch Avenue", "C": "Cedar Avenue"}
WEST_STREETS = {"0": "Market Street", "1": "Station Street"}
GATEWAY_SIDE = {"bottom": "south", "top": "north", "left": "west", "right": "east"}

# Harbor City: 23 unnamed SUMO crossings, named from the centre outwards.
HARBOR_NAMES = ["Harbour Square", "Quayside", "Lighthouse", "Anchor Point", "Old Town", "Fishmarket",
                "Dockyard", "Pier Gate", "Saltmarsh", "Beacon Hill", "Ferry Landing", "Wharf End",
                "Customs House", "Seawall", "Rope Walk", "Tidewater", "Gull Rock", "Mariner's Cross",
                "Boatyard", "Shipwright", "North Mole", "Coral Bay", "Breakwater", "Sandbar"]


def _config(net_name: str) -> dict:
    f = TOWNS / f"{net_name}.yaml"
    if not f.exists():
        return {}
    import yaml
    return yaml.safe_load(f.read_text(encoding="utf-8")) or {}


def town_names(net_name: str) -> dict:
    """Node names: a simulated study city's from its config (network.town.names); a real town's from its map import
    (data/<town>/network/names.json: the streets that meet at each junction), with the config's network.names on top."""
    cfg = _config(net_name)
    net = cfg.get("network") or {}
    if net.get("source") == "osm":
        import json
        out_dir = Path(cfg.get("out_dir") or f"../data/{net_name}")
        out_dir = out_dir if out_dir.is_absolute() else TOWNS / out_dir
        f = out_dir / "network" / "names.json"
        names = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        return {**names, **{str(k): str(v) for k, v in (net.get("names") or {}).items()}}
    return {str(k): str(v) for k, v in ((net.get("town") or {}).get("names") or {}).items()}


def city(net_name: str) -> dict:
    """The town's display name and what its numbers come from: its config's `name` / `tagline`, real (cameras) or
    simulated (SUMO sample data)."""
    if net_name in CITIES:
        return {**CITIES[net_name], "real": False}
    cfg = _config(net_name)
    real = (cfg.get("source") or {}).get("kind") == "real"
    return {"name": cfg.get("name") or net_name.replace("_", " ").title(), "tagline": cfg.get("tagline", ""),
            "kind": "Network", "real": real, "was": net_name,
            "source": "a real town, measured by its cameras" if real else "a simulated town (SUMO sample data)"}


def city_name(net_name: str) -> str:
    return city(net_name)["name"]


class Places:
    """Names for one network's junctions and roads, from its `nodes` and `edges` tables."""

    def __init__(self, net_name: str, nodes, edges):
        self.net = net_name
        self.kind = dict(zip(nodes.node_id, nodes.kind))
        self.xy = {r.node_id: (r.x, r.y) for r in nodes.itertuples()}
        self.edges = edges.set_index("edge_id")
        self.junction = {}
        if net_name == "maple_falls":
            self.junction = dict(MAPLE_DISTRICTS)
        elif net_name == "westbrook":
            self.junction = {n: f"{WEST_AVENUES.get(n[0], n[0])} & {WEST_STREETS.get(n[1:], n[1:])}"
                             for n in nodes.node_id if self.kind[n] == "junction"}
        elif town_names(net_name):
            self.junction = town_names(net_name)
        else:
            j = nodes[nodes.kind == "junction"].copy()
            cx, cy = j.x.mean(), j.y.mean()
            j["d"] = (j.x - cx) ** 2 + (j.y - cy) ** 2
            names = HARBOR_NAMES if net_name == "harbor_city" else []
            for i, n in enumerate(j.sort_values(["d", "node_id"]).node_id):
                self.junction[n] = names[i] if i < len(names) else f"Junction {n}"

    def node(self, node_id: str) -> str:
        if node_id in self.junction:
            return self.junction[node_id]
        if node_id.startswith("F_") and node_id[2:] in self.junction:     # a study city's neighbourhood road end
            return f"{self.junction[node_id[2:]]} neighbourhood"
        if self.net == "westbrook":
            side = re.match(r"[a-z]+", node_id)
            return f"{GATEWAY_SIDE.get(side.group(0), side.group(0)) if side else node_id} gateway"
        return "the outskirts"

    def street(self, edge_id: str) -> str:
        """The street an edge lies on, without its direction."""
        e = self.edges.loc[edge_id]
        a, b = e.from_node, e.to_node
        if self.net == "maple_falls":
            if frozenset((a, b)) in MAPLE_STREETS:
                return MAPLE_STREETS[frozenset((a, b))]
            j = a if a in self.junction else b
            return f"{self.junction.get(j, j)} local streets"
        if self.net == "westbrook":
            ja = a if a in self.junction else b                     # a junction end, and the other end
            jb = b if ja == a else a
            if ja[0] == jb[0] or self.kind[jb] == "boundary" and re.match(r"(top|bottom)", jb):
                return WEST_AVENUES.get(ja[0], ja)
            return WEST_STREETS.get(ja[1:], ja)
        return f"{self.node(a)} – {self.node(b)}"

    def label(self, node_id: str) -> str:
        """Short junction name for the map: 'Ash & Station'."""
        return re.sub(r" (Avenue|Street)\b", "", self.node(node_id))

    def short_road(self, edge_id: str) -> str:
        """Road name for tight spaces: 'Station St eastbound to Birch Ave'."""
        s = self.road(edge_id).replace(" Avenue", " Ave").replace(" Street", " St").replace(" Boulevard", " Blvd")
        return re.sub(r", (to|from) (north|south|east|west) gateway", lambda m: f" ({m.group(1)} {m.group(2)})", s)

    def heading(self, edge_id: str) -> str:
        e = self.edges.loc[edge_id]
        (x0, y0), (x1, y1) = self.xy[e.from_node], self.xy[e.to_node]
        if abs(x1 - x0) >= abs(y1 - y0):
            return "eastbound" if x1 > x0 else "westbound"
        return "northbound" if y1 > y0 else "southbound"

    def road(self, edge_id: str) -> str:
        """Street plus the way it goes: 'Oakwood Avenue → Downtown', 'Station Street eastbound to Birch
        Avenue'; parallel roads with the same name get a number (2), (3)."""
        if not hasattr(self, "_labels"):
            self._labels, seen = {}, {}
            for e in self.edges.index:
                base = self._road(e)
                seen[base] = seen.get(base, 0) + 1
                self._labels[e] = base if seen[base] == 1 else f"{base} ({seen[base]})"
        return self._labels[edge_id]

    def _road(self, edge_id: str) -> str:
        e = self.edges.loc[edge_id]
        to_out, from_out = self.kind.get(e.to_node) == "boundary", self.kind.get(e.from_node) == "boundary"
        if self.net == "maple_falls" and (to_out or from_out):
            return f"{self.street(edge_id)} ({'outbound' if to_out else 'inbound'})"
        if self.net == "westbrook":
            street = self.street(edge_id)
            if to_out:
                end = f", to {self.node(e.to_node)}"
            elif from_out:
                end = f", from {self.node(e.from_node)}"
            else:                                                     # the cross street it leads to
                end = " to " + next(p for p in self.node(e.to_node).split(" & ") if p != street)
            return f"{street} {self.heading(edge_id)}{end}"
        if self.net == "harbor_city" or self.net not in ("maple_falls",):
            return f"{self.node(e.from_node)} → {self.node(e.to_node)}"
        return f"{self.street(edge_id)} → {self.node(e.to_node)}"


SCENARIO_TITLES = {
    # as defined under whatif.scenarios in sim/maple_falls.yaml
    "close_J1_J2_am": ("Oakwood Avenue closed, morning peak", "Road into Downtown shut 07:30–09:00"),
    "close_J2_J3_pm": ("River Road closed, evening peak", "Road to Riverside shut 17:00–18:30"),
    "centre_growth": ("Downtown grows by 30%", "30% more trips to and from the centre, all day"),
}


def scenario_title(name: str) -> tuple[str, str]:
    return SCENARIO_TITLES.get(name, (name.replace("_", " ").capitalize(), ""))


# ---------------------------------------------------------------------- delay in plain words
# The model counts delay in vehicle-hours (veh-h: every vehicle's lost time, added up). Customer pages say it as
# the time drivers lose together, and a road change's impact as a word; the technical view keeps veh-h.
IMPACT = ((2.5, "Low"), (5.0, "Moderate"))         # extra veh-h of a road change up to each limit; above: High


def time_lost(veh_h: float) -> str:
    """Drivers' time lost, all together: '45 min', '2.4 h', '13 h'."""
    h = abs(float(veh_h))
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h" if h < 10 else f"{h:.0f} h"


def impact_word(veh_h: float) -> str:
    """Low / Moderate / High: a road change's extra delay against the limits of the Predict verdict."""
    return next((w for lim, w in IMPACT if veh_h <= lim), "High")
