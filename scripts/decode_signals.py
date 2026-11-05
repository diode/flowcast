"""Decode a town's SUMO signal plans (net.net.xml tlLogic, netconvert's defaults) into the timing-sheet format of
network.town.signals (docs/architecture-next.md 5.1), once; after that the town config is the source and
`netstudy network` gives it back to SUMO.

  .venv\\Scripts\\python scripts\\decode_signals.py --sim-config towns/brookhaven.yaml      # prints YAML to paste

A green phase (any G or g) starts a sheet phase: movements all of whose links are G -> go, any g -> yield. The
following phases with y become its amber_s, an all-red one its all_red_s. A link still g during the amber (a turn
that keeps going) becomes amber like the rest, and a partial amber inside a green phase (some links y while others
stay green, as netconvert makes for larger junctions) becomes red: the sheet's simplifications. Checked by a round
trip through network.signal_states: every green phase gives back the same G/g links (y read as r).
"""
from __future__ import annotations

import argparse
from xml.etree import ElementTree as ET

import yaml

from netstudy.cli import load_sim_config
from netstudy.network import signal_states


def movement(junction: str, a: str, b: str) -> str:
    fa = "hood" if a == f"in_{junction}" else a.split("_")[0]
    tb = "hood" if b == f"out_{junction}" else b.split("_")[1]
    return f"{fa}>{tb}"


def decode(net_file) -> dict:
    root = ET.parse(net_file).getroot()
    links: dict[str, dict[int, str]] = {}
    for c in root.iter("connection"):
        if c.get("tl"):
            links.setdefault(c.get("tl"), {})[int(c.get("linkIndex"))] = movement(c.get("tl"), c.get("from"), c.get("to"))
    out = {}
    for tl in root.iter("tlLogic"):
        j, phases = tl.get("id"), []
        for ph in tl.iter("phase"):
            s, d = ph.get("state"), float(ph.get("duration"))
            if "G" in s or ("g" in s and "y" not in s):
                by: dict[str, set] = {}
                for i, ch in enumerate(s):
                    by.setdefault(links[j][i], set()).add(ch)
                go = sorted(m for m, cs in by.items() if cs == {"G"})
                yld = sorted(m for m, cs in by.items() if "g" in cs and cs <= {"G", "g"})
                phases.append({"green_s": d, "go": go, **({"yield": yld} if yld else {})})
            elif "y" in s and phases:
                phases[-1]["amber_s"] = phases[-1].get("amber_s", 0) + d
            elif set(s) == {"r"} and phases:
                phases[-1]["all_red_s"] = phases[-1].get("all_red_s", 0) + d
        out[j] = {"phases": [{k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in p.items()}
                             for p in phases]}
    return out


def check(net_file, signals: dict):
    """Round trip: the sheet's green phases give back the plan's G/g links."""
    root = ET.parse(net_file).getroot()
    orig = {tl.get("id"): [p.get("state").replace("y", "r") for p in tl.iter("phase")
                           if "G" in p.get("state") or ("g" in p.get("state") and "y" not in p.get("state"))]
            for tl in root.iter("tlLogic")}
    back = signal_states(net_file, signals)
    for j, states in back.items():
        greens = [s for _, s in states if "G" in s or ("g" in s and "y" not in s)]
        assert greens == orig[j], f"{j}: {greens} != {orig[j]}"
    return {j: sum(d for d, _ in s) for j, s in back.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", required=True)
    args = p.parse_args(argv)
    sim, out_dir = load_sim_config(args.sim_config)
    net_file = out_dir / "network" / "net.net.xml"
    signals = decode(net_file)
    cycles = check(net_file, signals)
    print(f"# decoded from {net_file.name} by scripts/decode_signals.py; cycles (s): "
          + ", ".join(f"{j} {c:g}" for j, c in cycles.items()))
    print(yaml.safe_dump({"signals": signals}, sort_keys=False, default_flow_style=None, width=118))


if __name__ == "__main__":
    main()
