"""A real town's road network from an OpenStreetMap extract, without SUMO (netstudy onboard).

  load(path)                 the .osm XML (openstreetmap.org Export, Overpass, JOSM): nodes and ways
  fetch(bbox, path)          download the extract for a bounding box from the Overpass API
  build(osm_path, cfg)       the graph FlowCast uses, in the same tables as a simulated town:
      nodes   node_id, kind (junction | boundary), x, y (metres east / north of the map's centre), signal, lat, lon
      edges   one row per driving direction between two nodes: edge_id <from>_<to>, kind (in | out | road),
              length_m, lanes, speed_kmh, capacity_pcu, camera_id (set by netstudy cameras), street
      shapes  each edge's centre line in driving order, metres
      names   readable node names: the streets that meet there ("High Street & Mill Lane")

Ways are kept by `network.osm.highways` (main roads by default). A node where three or more road ends meet is a
junction; where a way stops (a dead end, or the extract's edge) it is a boundary point, so traffic enters and
leaves the town there (edge kinds in / out, as in the simulated towns). Junction nodes closer than `merge_m` are
one junction (dual carriageways, small roundabouts). A junction is signalised when a traffic_signals node lies
within `signal_m` of it. Lanes and speed come from the lanes / maxspeed tags, else per highway type.
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

EARTH_M = 6_371_000.0
JAM_M_PER_PCU = 6.0                        # same rule as the simulated towns' cameras (graph.JAM_M_PER_PCU)
HIGHWAYS = ["motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link",
            "secondary_link", "tertiary_link", "unclassified", "residential"]
DEFAULT_HIGHWAYS = ["trunk", "primary", "secondary", "tertiary", "trunk_link", "primary_link", "secondary_link",
                    "tertiary_link"]
RANK = {h: i for i, h in enumerate(HIGHWAYS)}
LANES = {"motorway": 2, "trunk": 2, "primary": 2}          # per direction, when the way has no lanes tag; else 1
SPEED_KMH = {"motorway": 100, "trunk": 80, "primary": 50, "secondary": 50, "tertiary": 40, "unclassified": 40,
             "residential": 30}
OVERPASS = "https://overpass-api.de/api/interpreter"


def settings(net_cfg: dict) -> dict:
    o = net_cfg.get("osm") or {}
    return {"file": o.get("file"), "bbox": o.get("bbox"), "highways": list(o.get("highways") or DEFAULT_HIGHWAYS),
            "merge_m": float(o.get("merge_m", 25)), "signal_m": float(o.get("signal_m", 30)),
            "names": {str(k): str(v) for k, v in (net_cfg.get("names") or {}).items()}}


def fetch(bbox: list[float], path: Path, highways: list[str] = DEFAULT_HIGHWAYS, timeout: int = 180) -> Path:
    """The roads (and their nodes) inside bbox [south, west, north, east] from the Overpass API, as .osm XML."""
    s, w, n, e = bbox
    query = (f'[out:xml][timeout:{timeout}];(way["highway"~"^({"|".join(highways)})$"]({s},{w},{n},{e});'
             f'node["highway"="traffic_signals"]({s},{w},{n},{e}););(._;>;);out body;')
    req = urllib.request.Request(OVERPASS, urllib.parse.urlencode({"data": query}).encode(),
                                 {"User-Agent": "FlowCast town onboarding"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(urllib.request.urlopen(req, timeout=timeout + 30).read())
    return path


def load(path: Path) -> tuple[dict, list]:
    """({osm node id: (lat, lon, tags)}, [(way id, [node ids], tags)])."""
    nodes, ways = {}, []
    for _, el in ET.iterparse(str(path), events=("end",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("lat")), float(el.get("lon")),
                                   {t.get("k"): t.get("v") for t in el.findall("tag")})
            el.clear()
        elif el.tag == "way":
            ways.append((el.get("id"), [nd.get("ref") for nd in el.findall("nd")],
                         {t.get("k"): t.get("v") for t in el.findall("tag")}))
            el.clear()
    return nodes, ways


def _speed(tags: dict, highway: str) -> float:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mph)?", tags.get("maxspeed", ""))
    if m:
        return round(float(m.group(1)) * (1.609 if m.group(2) else 1.0), 1)
    return float(SPEED_KMH.get(highway.removesuffix("_link"), 40))


def _lanes(tags: dict, highway: str, oneway: bool, forward: bool) -> int:
    for key in (("lanes:forward" if forward else "lanes:backward"),):
        if tags.get(key, "").isdigit():
            return max(1, int(tags[key]))
    if tags.get("lanes", "").isdigit():
        n = int(tags["lanes"])
        return max(1, n if oneway else n // 2)
    return LANES.get(highway.removesuffix("_link"), 1)


def _direction(tags: dict) -> str:
    """both | forward | backward"""
    ow = tags.get("oneway", "")
    if ow in ("yes", "true", "1") or tags.get("junction") in ("roundabout", "circular") or \
            tags.get("highway") == "motorway":
        return "forward"
    return "backward" if ow == "-1" else "both"


def build(osm_path: Path, net_cfg: dict) -> dict:
    """The town graph from the extract (see the module doc): {'nodes', 'edges', 'shapes', 'names', 'osm', 'centre'}."""
    cfg = settings(net_cfg)
    raw_nodes, raw_ways = load(osm_path)
    keep = set(cfg["highways"])
    ways = [(wid, [n for n in refs if n in raw_nodes], tags) for wid, refs, tags in raw_ways
            if tags.get("highway") in keep and tags.get("area") != "yes" and tags.get("access") not in ("no", "private")]
    ways = [w for w in ways if len(w[1]) >= 2]
    if not ways:
        raise SystemExit(f"{osm_path}: no roads of the kinds {', '.join(sorted(keep))} (network.osm.highways)")
    if cfg["bbox"]:                                    # crop: a way leaving the box ends at a boundary point
        s, w, n, e = cfg["bbox"]
        inside = {k for k, (la, lo, _) in raw_nodes.items() if s <= la <= n and w <= lo <= e}
        cut = []
        for wid, refs, tags in ways:
            run = []
            for r in refs + [None]:
                if r is not None and r in inside:
                    run.append(r)
                else:
                    if len(run) >= 2:
                        cut.append((wid, run, tags))
                    run = []
        ways = cut
    used = {r for _, refs, _ in ways for r in refs}
    lat0 = sum(raw_nodes[r][0] for r in used) / len(used)
    lon0 = sum(raw_nodes[r][1] for r in used) / len(used)
    kx = math.radians(1) * EARTH_M * math.cos(math.radians(lat0))
    ky = math.radians(1) * EARTH_M
    xy = {r: ((raw_nodes[r][1] - lon0) * kx, (raw_nodes[r][0] - lat0) * ky) for r in used}

    # vertices: ends of road pieces (three or more neighbours, or one), and where two ways of different direction meet
    nbrs: dict[str, set] = {}
    ends: dict[str, list] = {}
    for _, refs, tags in ways:
        for a, b in zip(refs, refs[1:]):
            nbrs.setdefault(a, set()).add(b)
            nbrs.setdefault(b, set()).add(a)
        for r in (refs[0], refs[-1]):
            ends.setdefault(r, []).append(_direction(tags))
    vertex = {r for r in used if len(nbrs.get(r, ())) != 2 or len(set(ends.get(r, []))) > 1}
    # chains: way pieces between vertices
    chains = []
    for wid, refs, tags in ways:
        start = 0
        for i in range(1, len(refs)):
            if refs[i] in vertex or i == len(refs) - 1:
                piece = refs[start:i + 1]
                if len(piece) >= 2 and piece[0] != piece[-1]:
                    chains.append((piece, tags, wid))
                start = i
    for piece, _, _ in chains:                         # a chain end is a vertex even if its way just continues
        vertex.update((piece[0], piece[-1]))

    # merge junction vertices joined by a short chain (one junction drawn as several nodes)
    parent = {v: v for v in vertex}

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def length(piece):
        return sum(math.dist(xy[a], xy[b]) for a, b in zip(piece, piece[1:]))

    for piece, _, _ in chains:
        a, b = piece[0], piece[-1]
        if len(nbrs[a]) >= 3 and len(nbrs[b]) >= 3 and length(piece) < cfg["merge_m"]:
            parent[root(a)] = root(b)
    groups: dict[str, list] = {}
    for v in vertex:
        groups.setdefault(root(v), []).append(v)

    # nodes: junctions numbered from the centre out (J1 ..), boundary points (B1 ..)
    degree = {}
    for piece, _, _ in chains:
        a, b = root(piece[0]), root(piece[-1])
        if a != b:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
    gxy = {g: (sum(xy[v][0] for v in vs) / len(vs), sum(xy[v][1] for v in vs) / len(vs)) for g, vs in groups.items()}
    signal_pts = [xy.get(k) or ((lo - lon0) * kx, (la - lat0) * ky) for k, (la, lo, t) in raw_nodes.items()
                  if t.get("highway") == "traffic_signals"]
    order = sorted((g for g in groups if degree.get(g, 0) > 0), key=lambda g: (math.hypot(*gxy[g]), g))
    ids, nj, nb = {}, 0, 0
    for g in order:
        if degree[g] == 1:
            nb += 1
            ids[g] = f"B{nb}"
        else:
            nj += 1
            ids[g] = f"J{nj}"
    nodes = pd.DataFrame([{
        "node_id": ids[g], "kind": "boundary" if ids[g].startswith("B") else "junction",
        "x": round(gxy[g][0], 1), "y": round(gxy[g][1], 1),
        "signal": int(ids[g].startswith("J") and any(math.dist(p, gxy[g]) <= cfg["signal_m"] for p in signal_pts)),
        "lat": round(lat0 + gxy[g][1] / ky, 7), "lon": round(lon0 + gxy[g][0] / kx, 7),
    } for g in order])

    # edges: each chain in its driving directions
    rows, shapes, streets_at = [], {}, {}
    for piece, tags, wid in chains:
        a, b = root(piece[0]), root(piece[-1])
        if a == b:
            continue
        hw, name = tags.get("highway", ""), tags.get("name") or tags.get("ref") or ""
        for g in (a, b):
            if name:
                streets_at.setdefault(g, {}).setdefault(name, RANK.get(hw, 99))
        d = _direction(tags)
        for fwd in ((True, False) if d == "both" else ((d == "forward"),)):
            u, v = (a, b) if fwd else (b, a)
            pts = [xy[r] for r in (piece if fwd else piece[::-1])]
            pts = [gxy[u]] + pts[1:-1] + [gxy[v]]            # ends at the (merged) junction centres
            eid = f"{ids[u]}_{ids[v]}"
            k = 2
            while eid in shapes:
                eid = f"{ids[u]}_{ids[v]}_{k}"
                k += 1
            lanes = _lanes(tags, hw, d != "both", fwd)
            ln = length(piece)
            kind = "in" if ids[u].startswith("B") else "out" if ids[v].startswith("B") else "road"
            shapes[eid] = [[round(x, 2), round(y, 2)] for x, y in pts]
            rows.append({"edge_id": eid, "from_node": ids[u], "to_node": ids[v], "kind": kind,
                         "length_m": round(ln, 1), "lanes": lanes, "speed_kmh": _speed(tags, hw),
                         "capacity_pcu": round(lanes * ln / JAM_M_PER_PCU, 1), "camera_id": None, "street": name,
                         "osm_way": wid, "highway": hw})
    edges = pd.DataFrame(rows)
    names = {}
    for g in order:
        nid = ids[g]
        if nid in cfg["names"]:
            names[nid] = cfg["names"][nid]
            continue
        streets = [s for s, _ in sorted(streets_at.get(g, {}).items(), key=lambda kv: (kv[1], kv[0]))]
        if nid.startswith("B"):
            names[nid] = f"{streets[0]} (edge of map)" if streets else f"Edge of map {nid}"
        else:
            names[nid] = " & ".join(streets[:2]) if len(streets) >= 2 else (f"{streets[0]} ({nid})" if streets
                                                                            else f"Junction {nid}")
    seen = {}
    for nid, nm in names.items():                      # two junctions of the same two streets
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            names[nid] = f"{nm} ({seen[nm]})"
    osm_ids = {ids[g]: sorted(groups[g]) for g in order}
    return {"nodes": nodes, "edges": edges, "shapes": shapes, "names": names, "osm": osm_ids,
            "centre": {"lat": lat0, "lon": lon0}}


def to_xy(centre: dict, lat: float, lon: float) -> tuple[float, float]:
    """A map point -> the town's metres (east, north of the centre the extract was built around)."""
    kx = math.radians(1) * EARTH_M * math.cos(math.radians(centre["lat"]))
    return (lon - centre["lon"]) * kx, (lat - centre["lat"]) * math.radians(1) * EARTH_M


def save(town: dict, out_dir: Path) -> None:
    """network/shapes.json, names.json, osm.json (+ centre), the nodes / edges tables, graph.json."""
    from .graph import save_graph

    net_dir = out_dir / "network"
    net_dir.mkdir(parents=True, exist_ok=True)
    (net_dir / "shapes.json").write_text(json.dumps(town["shapes"]), encoding="utf-8")
    (net_dir / "names.json").write_text(json.dumps(town["names"], indent=1, ensure_ascii=False), encoding="utf-8")
    (net_dir / "osm.json").write_text(json.dumps({"centre": town["centre"], "nodes": town["osm"]}), encoding="utf-8")
    save_graph(out_dir / "traffic.db", out_dir / "graph.json", town["nodes"], town["edges"])


def load_shapes(out_dir: Path) -> dict[str, list[tuple[float, float]]]:
    f = out_dir / "network" / "shapes.json"
    return {k: [tuple(p) for p in v] for k, v in json.loads(f.read_text(encoding="utf-8")).items()} if f.exists() else {}


def centre(out_dir: Path) -> dict:
    return json.loads((out_dir / "network" / "osm.json").read_text(encoding="utf-8"))["centre"]


def draw(town: dict, cams: list[dict], title: str, png: Path) -> Path:
    """map.png for review: roads by kind, junction ids and names, cameras."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 11))
    kinds = town["edges"].set_index("edge_id").kind
    for eid, pts in town["shapes"].items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color={"road": "#4b5563", "in": "#9ca3af", "out": "#9ca3af"}[kinds[eid]], lw=1.4)
    n = town["nodes"]
    j = n[n.kind == "junction"]
    ax.scatter(j.x, j.y, s=[40 if s else 18 for s in j.signal], c=["#c5221f" if s else "#111827" for s in j.signal],
               zorder=3)
    for r in j.itertuples():
        ax.annotate(f"{r.node_id} {town['names'].get(r.node_id, '')}", (r.x, r.y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    for c in cams:
        ax.scatter([c["stop"][0]], [c["stop"][1]], marker="s", c="#0f766e", s=30, zorder=4)
        ax.annotate(c["camera_id"], c["stop"], fontsize=7, color="#0f766e", xytext=(4, -10), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_title(f"{title}: {len(j)} junctions ({int(j.signal.sum())} signalised), {len(town['edges'])} road "
                 f"directions, {len(cams)} cameras")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png
