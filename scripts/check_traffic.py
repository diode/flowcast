"""Step 2: does the simulated traffic look right?

  .venv\\Scripts\\python scripts\\check_traffic.py [--sim-config sim/maple_falls.yaml]

Reads data/maple_falls/sumo_series.csv (loop `flow` per camera per 5 minutes) and checks:
  1. routine zones: weekday hourly flow on in_<J> follows the zone's generation slot table
  2. day-to-day consistency: correlation of each day's hourly profile with the other days
     (high for routine zones, low for random ones)
  3. surges on random zones: runs of >= 15 minutes above 2x the day's median
  4. direction: morning and evening peaks load opposite directions between residential and centre
  5. balance: trips planned vs vehicles counted entering (in_<J>) and leaving (out_<J>) town
Writes a summary to stdout and charts to data/maple_falls/check/.
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from netstudy.cli import load_sim_config
from netstudy.zones import GRID, slot_levels

INK, MUTED, GRIDC = "#222222", "#6b6b6b", "#e6e6e6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed categorical order


def load(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    cams = {c["camera_id"]: c["edge"] for c in json.loads((out_dir / "cameras.json").read_text())}
    s = pd.read_csv(out_dir / "sumo_series.csv", usecols=["camera_id", "ts", "flow", "total", "level", "incident"])
    start = pd.to_datetime(s.ts) - pd.Timedelta(minutes=5)     # ts marks the end of its slot
    s["edge"] = s.camera_id.map(cams)
    s["day"], s["hour"], s["slot"] = start.dt.normalize(), start.dt.hour, start.dt.hour * 12 + start.dt.minute // 5
    s["weekday"] = start.dt.weekday < 5
    return s, cams


def hourly(s: pd.DataFrame, edge: str) -> pd.DataFrame:
    """Vehicles per hour: rows = days, columns = hour of day."""
    return s[s.edge == edge].pivot_table(index="day", columns="hour", values="flow", aggfunc="sum")


def expected_hourly(zone: dict, weekday: int) -> np.ndarray:
    lv = slot_levels(zone.get("generation", []), weekday, float(zone.get("base_level", 0.05)),
                     int(zone.get("ramp_minutes", 30)))
    return lv.reshape(24, -1).mean(axis=1) * float(zone["peak_veh_per_hour"])


def day_corr(h: pd.DataFrame) -> float:
    return float(np.mean([np.corrcoef(h.loc[a], h.loc[b])[0, 1] for a, b in combinations(h.index, 2)]))


def surges(s: pd.DataFrame, edge: str, factor: float = 2.0, min_slots: int = 3) -> pd.Series:
    """Per day: number of runs of >= min_slots slots with flow above factor x the day's median."""
    out = {}
    for day, d in s[s.edge == edge].sort_values("slot").groupby("day"):
        high = (d.flow > factor * max(d.flow.median(), 1)).to_numpy()
        runs, n = 0, 0
        for h in np.append(high, False):
            if h:
                n += 1
            else:
                runs += n >= min_slots
                n = 0
        out[day] = runs
    return pd.Series(out)


def planned_trips(out_dir: Path) -> pd.DataFrame:
    """Trips per origin zone per day, from the route files SUMO ran."""
    rows = []
    for f in sorted((out_dir / "sumo_days").glob("trips_*.rou.xml")):
        day = pd.Timestamp(f.name.split("_")[1].split(".")[0])
        for zone, n in pd.Series(re.findall(r'from="in_(\w+)"', f.read_text())).value_counts().items():
            rows.append({"day": day, "zone": zone, "trips": n})
    return pd.DataFrame(rows)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 23)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="sim/maple_falls.yaml")
    args = p.parse_args(argv)
    sim, out_dir = load_sim_config(args.sim_config)
    zones = sim["demand"]["zones"]
    s, cams = load(out_dir)
    chart_dir = out_dir / "check"
    chart_dir.mkdir(exist_ok=True)
    wd = s[s.weekday]

    # 1 + 2 + 3: neighbourhood cameras (in_<J>)
    print("\n1-3. Neighbourhood approaches in_<J>: profile, day-to-day consistency, surges")
    rows = []
    for j, z in zones.items():
        h_all, h_wd = hourly(s, f"in_{j}"), hourly(wd, f"in_{j}")
        row = {"camera": next(c for c, e in cams.items() if e == f"in_{j}"), "zone": f"{j} {z['name']}",
               "kind": z["kind"], "day_corr_weekdays": day_corr(h_wd), "day_corr_all": day_corr(h_all)}
        if z["kind"] == "routine":
            exp = expected_hourly(z, weekday=2)
            obs = h_wd.mean().to_numpy()
            row |= {"fit_r": np.corrcoef(obs, exp)[0, 1], "peak_obs": f"{obs.argmax():02d}h",
                    "peak_table": f"{exp.argmax():02d}h", "obs/table": obs.sum() / exp.sum()}
        else:
            sg = surges(s, f"in_{j}")
            row |= {"surges_per_day": sg.mean(), "days_with_surge": f"{(sg > 0).mean():.0%}"}
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # 4: direction between residential (J1) and centre (J2)
    print("\n4. Direction: residential J1 <-> centre J2 (weekday mean vehicles per hour)")
    dirs = []
    for edge, label in (("J1_J2", "J1 -> J2 (towards centre)"), ("J2_J1", "J2 -> J1 (towards homes)")):
        h = hourly(wd, edge).mean()
        dirs.append({"camera": next(c for c, e in cams.items() if e == edge), "direction": label,
                     "morning 07-10": h.loc[7:9].mean(), "evening 17-20": h.loc[17:19].mean(),
                     "morning/evening": h.loc[7:9].mean() / h.loc[17:19].mean()})
    print(pd.DataFrame(dirs).round(2).to_string(index=False))

    # 5: balance of trips planned, entering and leaving
    print("\n5. Balance per day (mean over 28 days)")
    plan = planned_trips(out_dir).groupby("day").trips.sum()
    daily = s.groupby(["day", s.edge.str.extract(r"^(in|out)_", expand=False)]).flow.sum().unstack()
    bal = pd.DataFrame({"planned": plan, "entered (in_ loops)": daily["in"], "left (out_ loops)": daily["out"]})
    print(bal.mean().round(0).to_string())
    print(f"entered/planned {bal.iloc[:, 1].sum() / bal.planned.sum():.1%}, "
          f"left/entered {bal.iloc[:, 2].sum() / bal.iloc[:, 1].sum():.1%}")
    by_hour = s.groupby(["hour", s.edge.str.extract(r"^(in|out)_", expand=False)]).flow.sum().unstack() / s.day.nunique()
    late = by_hour.loc[[22, 23, 0]]
    print("vehicles per hour, entered vs left, late evening:\n" + late.round(0).to_string())

    # charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, j in zip(axes[0], [j for j, z in zones.items() if z["kind"] == "routine"]):
        z, edge = zones[j], f"in_{j}"
        h = hourly(wd, edge)
        for _, r in h.iterrows():
            ax.plot(range(24), r.to_numpy(), color=SERIES[0], alpha=0.12, lw=1)
        ax.plot(range(24), h.mean().to_numpy(), color=SERIES[0], lw=2, label="simulated, weekday mean")
        ax.plot(range(24), expected_hourly(z, 2), color=SERIES[1], lw=2, ls="--", label="slot table (generation)")
        style(ax, f"{next(c for c, e in cams.items() if e == edge)} · {j} {z['name']}: vehicles per hour, weekdays")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[1][0]
    for i, (j, z) in enumerate([(j, z) for j, z in zones.items() if z["kind"] == "random"]):
        h = hourly(s, f"in_{j}")
        for k, (_, r) in enumerate(h.iterrows()):
            ax.plot(range(24), r.to_numpy(), color=SERIES[2 + i], alpha=0.25, lw=1,
                    label=f"in_{j} {z['name']}" if k == 0 else None)
    style(ax, "Random zones: every day's hourly profile (no shared shape)")
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    for line in leg.get_lines():
        line.set_alpha(1)
    ax = axes[1][1]
    for i, (edge, label) in enumerate((("J1_J2", "J1 → J2 towards centre"), ("J2_J1", "J2 → J1 towards homes"))):
        ax.plot(range(24), hourly(wd, edge).mean().to_numpy(), color=SERIES[i], lw=2, label=label)
    style(ax, "Spoke J1 ↔ J2: weekday vehicles per hour by direction")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    for ax in axes.flat:
        ax.set_xlabel("hour of day", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(chart_dir / "profiles.png", dpi=120)
    print(f"\nchart: {chart_dir / 'profiles.png'}")


if __name__ == "__main__":
    main()
