"""Road network of connected junctions (SUMO), and the cameras watching it.

A town is hand-drawn in the sim config: signalised junctions joined by two-way roads, and each
junction with its own neighbourhood road (edges in_<J> towards the junction, out_<J> away from it)
where that neighbourhood's trips start and end. The road points away from the town centre unless
`town.neighbourhoods.<J>` sets its `dir` [dx, dy] and `length_m`. Every road direction is its own edge,
and one camera watches exactly one edge (one direction). With `network.opendrive: true` the network is
also written as OpenDRIVE (net.xodr), from which CARLA builds the same roads as a 3-D world.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from xml.sax.saxutils import quoteattr


def ensure_sumo() -> Path:
    """Make SUMO_HOME point at a SUMO install (pip install eclipse-sumo sets none by itself)."""
    home = os.environ.get("SUMO_HOME")
    if not home:
        try:
            import sumo  # eclipse-sumo wheel

            home = sumo.SUMO_HOME
        except (ImportError, AttributeError):
            raise SystemExit("SUMO not found: pip install -e . (eclipse-sumo) or install SUMO and set SUMO_HOME")
        os.environ["SUMO_HOME"] = str(home)
    return Path(home)


def sumo_binary(name: str) -> str:
    ensure_sumo()
    import sumolib

    return sumolib.checkBinary(name)


def build_town(net_cfg: dict, out_dir: Path) -> Path:
    """Write the town's nodes and edges, convert them with netconvert, return the .net.xml path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t = net_cfg["town"]
    junctions = {j: (float(x), float(y)) for j, (x, y) in t["junctions"].items()}
    lanes, speed = int(t.get("lanes", 2)), float(t.get("speed_kmh", 40)) / 3.6
    fringe = float(t.get("fringe_m", 250))
    cx = sum(x for x, _ in junctions.values()) / len(junctions)
    cy = sum(y for _, y in junctions.values()) / len(junctions)
    nodes = [f'  <node id={quoteattr(j)} x="{x}" y="{y}" type="traffic_light"/>' for j, (x, y) in junctions.items()]
    edges = []
    hoods = t.get("neighbourhoods") or {}
    for j, (x, y) in junctions.items():
        h = hoods.get(j) or {}
        dx, dy = h.get("dir", (x - cx, y - cy))       # by default the neighbourhood lies away from the town centre
        d = math.hypot(dx, dy)
        ux, uy = (dx / d, dy / d) if d > 1e-9 else (0.0, -1.0)
        f = float(h.get("length_m", fringe))
        nodes.append(f'  <node id="F_{j}" x="{x + ux * f:.1f}" y="{y + uy * f:.1f}" type="priority"/>')
        edges.append(f'  <edge id="in_{j}" from="F_{j}" to={quoteattr(j)} numLanes="{lanes}" speed="{speed:.2f}"/>')
        edges.append(f'  <edge id="out_{j}" from={quoteattr(j)} to="F_{j}" numLanes="{lanes}" speed="{speed:.2f}"/>')
    for a, b in t["roads"]:
        for u, v in ((a, b), (b, a)):
            edges.append(f'  <edge id="{u}_{v}" from={quoteattr(u)} to={quoteattr(v)} '
                         f'numLanes="{lanes}" speed="{speed:.2f}"/>')
    nod, edg, net_file = out_dir / "town.nod.xml", out_dir / "town.edg.xml", out_dir / "net.net.xml"
    nod.write_text("<nodes>\n" + "\n".join(nodes) + "\n</nodes>\n", encoding="utf-8")
    edg.write_text("<edges>\n" + "\n".join(edges) + "\n</edges>\n", encoding="utf-8")
    convert = [sumo_binary("netconvert"), "--node-files", str(nod), "--edge-files", str(edg),
               "--tls.default-type", "static", "--no-turnarounds", "true",
               "--lefthand", str(bool(net_cfg.get("lefthand", True))).lower(), "--output-file", str(net_file)]
    subprocess.run(convert, check=True, capture_output=True)
    if t.get("signals"):                 # the timing sheet: link numbers come from the first pass, then the same call
        tll = out_dir / "town.tll.xml"
        write_signal_file(tll, signal_states(net_file, t["signals"]))
        subprocess.run(convert + ["--tllogic-files", str(tll)], check=True, capture_output=True)
    if net_cfg.get("opendrive"):
        subprocess.run([sumo_binary("netconvert"), "--sumo-net-file", str(net_file),
                        "--opendrive-output", str(out_dir / "net.xodr")], check=True, capture_output=True)
    return net_file


def movement_edges(junction: str, movement: str) -> tuple[str, str]:
    """'J2>hood' at junction C -> ('J2_C', 'out_C'): the road in and the road out, hood = in_/out_<J>."""
    a, b = (s.strip() for s in movement.split(">"))
    return (f"in_{junction}" if a == "hood" else f"{a}_{junction}",
            f"out_{junction}" if b == "hood" else f"{junction}_{b}")


def signal_states(net_file: Path, signals: dict) -> dict[str, list[tuple[float, str]]]:
    """The timing sheet as SUMO phases per junction: (duration, one state letter per link). A green phase gives
    G to its `go` movements and g to its `yield` ones; its amber turns those to y; all-red is every link r."""
    ensure_sumo()
    import sumolib

    net = sumolib.net.readNet(str(net_file), withPrograms=True)
    out = {}
    for j, plan in signals.items():
        links: dict[tuple[str, str], list[int]] = {}
        for a, b, i in net.getTLS(j).getConnections():
            links.setdefault((a.getEdge().getID(), b.getEdge().getID()), []).append(i)
        n = 1 + max(i for ix in links.values() for i in ix)
        phases = []
        for p in plan["phases"]:
            state = ["r"] * n
            for letter, key in (("G", "go"), ("g", "yield")):
                for m in p.get(key, []):
                    ix = links.get(movement_edges(j, m))
                    if ix is None:
                        raise SystemExit(f"signals.{j}: movement {m!r} is not a turn at {j}")
                    for i in ix:
                        state[i] = letter
            phases.append((float(p["green_s"]), "".join(state)))
            if p.get("amber_s"):
                phases.append((float(p["amber_s"]), "".join("y" if c in "Gg" else "r" for c in state)))
            if p.get("all_red_s"):
                phases.append((float(p["all_red_s"]), "r" * n))
        out[j] = phases
    return out


def write_signal_file(path: Path, states: dict[str, list[tuple[float, str]]]):
    lines = ["<tlLogics>"]
    for j, phases in states.items():
        lines.append(f'  <tlLogic id={quoteattr(j)} type="static" programID="0" offset="0">')
        lines += [f'    <phase duration="{d:g}" state="{s}"/>' for d, s in phases]
        lines.append("  </tlLogic>")
    path.write_text("\n".join(lines + ["</tlLogics>"]) + "\n", encoding="utf-8")


def camera_edges(net_cfg: dict) -> list[str]:
    """Edges that get a camera, from `network.cameras`:
      all         every edge: in_<J> first (cam01..), then the roads between junctions, then out_<J>
      approaches  edges ending at a junction (in_<J> and the roads between junctions)
      [ids]       exactly these edges
    """
    t = net_cfg["town"]
    choice = net_cfg.get("cameras", "all")
    if isinstance(choice, list):
        return [str(e) for e in choice]
    js = list(t["junctions"])
    edges = [f"in_{j}" for j in js] + [f"{u}_{v}" for a, b in t["roads"] for u, v in ((a, b), (b, a))]
    if choice == "approaches":
        return edges
    if choice == "all":
        return edges + [f"out_{j}" for j in js]
    raise SystemExit(f"network.cameras: expected all, approaches or a list of edge ids, got {choice!r}")


def cameras_on(net_file: Path, edge_ids: list[str], view_m: float) -> list[dict]:
    """One camera per given edge (one direction), watching it up from its stop line."""
    ensure_sumo()
    import sumolib

    net = sumolib.net.readNet(str(net_file))
    cams = []
    for i, e in enumerate(edge_ids, start=1):
        edge = net.getEdge(e)
        nd = edge.getToNode()
        lanes = [ln for ln in edge.getLanes() if ln.allows("passenger")]
        (x0, y0), (x1, y1) = edge.getShape()[-2], edge.getShape()[-1]
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        cams.append({
            "camera_id": f"cam{i:02d}",
            "name": f"Junction {nd.getID()} - approach {edge.getID()}",
            "junction": nd.getID(),
            "edge": edge.getID(),
            "lanes": [ln.getID() for ln in lanes],
            "x": nd.getCoord()[0], "y": nd.getCoord()[1],
            "stop": [x1, y1],                                         # stop line: end of the approach
            "upstream": [(x0 - x1) / length, (y0 - y1) / length],   # unit vector junction -> upstream
            # Jam density ~ one PCU per 6 m of lane (two-wheelers 0.5 PCU share lanes).
            "capacity_pcu": round(len(lanes) * min(view_m, edge.getLength()) / 6.0, 1),
        })
    return cams


def save_cameras(cams: list[dict], path: Path):
    path.write_text(json.dumps(cams, indent=2), encoding="utf-8")


def load_cameras(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found; run `netstudy network` first")
    return json.loads(path.read_text(encoding="utf-8"))
