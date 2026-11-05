"""A picture of a town for review: junctions with their names, every road direction with its id, length and
camera, and the neighbourhood roads. `netstudy network` writes it as <out_dir>/map.png.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROAD, HOOD, CAM, JUNCTION, SIDE = "#1f4e79", "#2e7d32", "#c62828", "#212121", 22.0


def _point(shape, frac: float):
    """Point and unit direction at `frac` of a polyline's length."""
    from sumolib.geomhelper import polyLength, positionAtShapeOffset

    total = polyLength(shape)
    a = positionAtShapeOffset(shape, total * frac)
    b = positionAtShapeOffset(shape, min(total, total * frac + 1.0))
    d = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    return a, ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def draw_map(net_file: Path, cams: list[dict], town: dict, title: str, out_png: Path, lefthand: bool = False):
    import sumolib

    net = sumolib.net.readNet(str(net_file))
    junctions = town["junctions"]
    names = town.get("names") or {}
    cam_of = {c["edge"]: c["camera_id"] for c in cams}
    pad = 120
    xs = [n.getCoord()[0] for n in net.getNodes()]
    ys = [n.getCoord()[1] for n in net.getNodes()]
    w, h = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
    width = min(24.0, max(10.0, w / 110))                    # about 110 m per inch, so labels stay readable
    fig, ax = plt.subplots(figsize=(width, width * h / w + 1.0))
    n_roads = 0
    for e in net.getEdges():
        inner = e.getFromNode().getID() in junctions and e.getToNode().getID() in junctions
        n_roads += inner
        colour = ROAD if inner else HOOD
        for ln in e.getLanes():
            lx, ly = zip(*ln.getShape())
            ax.plot(lx, ly, color=colour, lw=1.6, alpha=0.85, solid_capstyle="butt")
        shape = e.getLanes()[0].getShape()
        (x, y), (ux, uy) = _point(shape, 0.5)
        ax.annotate("", xy=(x + ux * 18, y + uy * 18), xytext=(x - ux * 18, y - uy * 18),
                    arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.6, "mutation_scale": 16})
        (x, y), (ux, uy) = _point(shape, 0.5 if inner else 0.6)
        rx, ry = uy, -ux                                     # right of the direction of travel
        label = f"{e.getID()}  {e.getLength():.0f} m" + (f"\n{cam_of[e.getID()]}" if e.getID() in cam_of else "")
        ax.text(x + rx * SIDE, y + ry * SIDE, label, fontsize=7.5, color=colour, ha="center", va="center",
                rotation=(math.degrees(math.atan2(uy, ux)) + 90) % 180 - 90, rotation_mode="anchor",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
    for c in cams:
        ax.plot(*c["stop"], marker="s", ms=5, color=CAM, zorder=4)
    for n in net.getNodes():
        x, y = n.getCoord()
        nid = n.getID()
        if nid in junctions:
            ax.plot(x, y, "o", ms=26, color="white", mec=JUNCTION, mew=2.2, zorder=5)
            ax.text(x, y, nid, fontsize=10, fontweight="bold", ha="center", va="center", zorder=6)
            if names.get(nid):
                ax.text(x, y - 45, names[nid], fontsize=11, fontweight="bold", ha="center", va="top", zorder=6,
                        bbox={"boxstyle": "round,pad=0.25", "fc": "#fff8e1", "ec": JUNCTION, "lw": 0.8})
        else:
            ax.plot(x, y, "D", ms=7, color=HOOD, zorder=5)
            ax.text(x, y + 22, f"{names.get(nid[2:], nid[2:])} neighbourhood", fontsize=8, color=HOOD,
                    ha="center", va="bottom", style="italic")
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    x0, y0 = min(xs) - pad + 30, min(ys) - pad + 30
    ax.plot([x0, x0 + 200], [y0, y0], color="black", lw=3)
    ax.text(x0 + 100, y0 + 12, "200 m", ha="center", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], color=ROAD, lw=2, label="road between junctions (one line per lane)"),
                       Line2D([], [], color=HOOD, lw=2, label="neighbourhood road (trips start / end)"),
                       Line2D([], [], marker="s", ls="", color=CAM, label="camera at the stop line it watches"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=JUNCTION, ms=12, label="junction (signalised)")],
              loc="upper left", fontsize=9, framealpha=0.9)
    side = "left" if lefthand else "right"
    ax.set_title(f"{title}: {len(junctions)} junctions, {n_roads} road directions between them, "
                 f"{len(net.getEdges())} edges, {len(cams)} cameras ({side}-hand traffic)", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
