"""Command line interface.

A real town (towns/<town>.yaml with source.kind: real; no SUMO, no CARLA):
  netstudy onboard  --sim-config towns/<town>.yaml [--fetch]   # OpenStreetMap roads -> graph, cameras, map.png
  netstudy cameras  --sim-config towns/<town>.yaml             # cameras + calibration again, after editing them
  netstudy receive  --sim-config towns/<town>.yaml [--port 8765]   # HTTP drop for recorders that push
  netstudy observe  --sim-config towns/<town>.yaml [--follow]  # inbox -> SAM 3.1 -> observations -> state -> model
  netstudy status   --sim-config towns/<town>.yaml             # the onboarding checklist

A simulated town (SUMO + CARLA sample data):
  netstudy network  --sim-config sim/maple_falls.yaml    # build the town network + cameras (one per edge) + graph
  netstudy simulate --sim-config sim/maple_falls.yaml    # weeks of per-camera series + true edge state (headless SUMO)
  netstudy state    --sim-config sim/maple_falls.yaml [--hide one_direction]   # observations -> edge_state
  netstudy kpis     --sim-config sim/maple_falls.yaml [--at 2026-10-06T18:00 | --day 2026-10-06]
  netstudy incidents --sim-config sim/maple_falls.yaml [--footprints-only]   # paired incident runs -> footprints
  netstudy whatif   --sim-config sim/maple_falls.yaml --scenario close_J1_J2_am [--list]   # what-if impact
  netstudy predict  --sim-config sim/maple_falls.yaml --date 2026-11-27 --time 18:00 [--edge J1_J2 --level jammed]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml


def load_sim_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        sim = yaml.safe_load(f) or {}
    out_dir = Path(sim.get("out_dir", "data/run"))
    if not out_dir.is_absolute():
        out_dir = (path.parent / out_dir).resolve()
    sim["_config_dir"] = str(path.parent)             # where the config's own relative paths start (map, inbox)
    return sim, out_dir


def _simulated_only(sim: dict, what: str):
    from .realtown import is_real

    if is_real(sim):
        raise SystemExit(f"netstudy {what} is for simulated towns; a real town uses netstudy onboard / observe")


def cmd_network(args):
    from .graph import build_graph, save_graph
    from .network import build_town, camera_edges, cameras_on, save_cameras
    from .townmap import draw_map

    sim, out_dir = load_sim_config(args.sim_config)
    _simulated_only(sim, "network")
    net_cfg = sim["network"]
    net_file = build_town(net_cfg, out_dir / "network")
    cams = cameras_on(net_file, camera_edges(net_cfg), float(net_cfg.get("view_m", 60)))
    save_cameras(cams, out_dir / "cameras.json")
    print(f"Network: {net_file}")
    for c in cams:
        print(f"  {c['camera_id']}  junction {c['junction']}  approach {c['edge']}  "
              f"{len(c['lanes'])} lanes  capacity {c['capacity_pcu']} PCU")
    nodes, edges = build_graph(net_file, cams)
    save_graph(out_dir / "traffic.db", out_dir / "graph.json", nodes, edges)
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges ({edges.camera_id.notna().sum()} with a camera) "
          f"-> traffic.db (nodes, edges) and graph.json")
    png = draw_map(net_file, cams, net_cfg["town"], out_dir.resolve().name.replace("_", " ").title(),
                   out_dir / "map.png", lefthand=bool(net_cfg.get("lefthand", True)))
    print(f"Map: {png}")


def cmd_state(args):
    from .graph import build_state, save_evaluation

    sim, out_dir = load_sim_config(args.sim_config)
    hidden = None
    if args.hide:
        scenarios = (sim.get("state") or {}).get("scenarios") or {}
        if args.hide not in scenarios:
            raise SystemExit(f"unknown scenario {args.hide!r}; sim config state.scenarios has: {', '.join(scenarios)}")
        hidden = scenarios[args.hide]
    db = out_dir / "traffic.db"
    state, evaluation = build_state(db, out_dir / "sumo_series.csv", hidden, args.hide,
                                    float(sim.get("network", {}).get("view_m", 60)))
    print(f"edge_state: {len(state)} rows ({state.edge_id.nunique()} edges x {state.ts.nunique()} slots), "
          f"{state.observed.mean():.0%} observed")
    if evaluation is not None:
        save_evaluation(db, evaluation)
        summary = evaluation.groupby("method").agg(flow_wape=("flow_wape", "mean"), load_mae=("load_mae", "mean"),
                                                   level_acc=("level_acc", "mean"),
                                                   congested_recall=("congested_recall", "mean"))
        print(f"hidden ({args.hide}): {', '.join(hidden)}; mean over hidden edges:\n{summary.round(3).to_string()}")


def cmd_kpis(args):
    import pandas as pd

    from .kpis import build_kpis, day_by_hour, snapshot

    _, out_dir = load_sim_config(args.sim_config)
    db = out_dir / "traffic.db"
    k = build_kpis(db)
    print(f"network_kpi: {len(k)} slots")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        if args.at:
            ts = pd.Timestamp(args.at).isoformat()
            kpi, edges = snapshot(db, ts)
            print(f"\nSlot ending {ts}\n" + kpi.drop("ts").to_string())
            print("\nEdges (the state the system knows: from its camera, or estimated; cam_* = edge_state):")
            print(edges.to_string(index=False))
        if args.day:
            print(f"\n{args.day} by hour\n" + day_by_hour(db, args.day).to_string())


def cmd_simulate(args):
    from .network import load_cameras
    from .simulate import run_days

    sim, out_dir = load_sim_config(args.sim_config)
    cams = load_cameras(out_dir / "cameras.json")
    db = out_dir / "traffic.db"
    series = run_days(sim, out_dir / "network" / "net.net.xml", cams, out_dir, db, workers=args.workers)
    print(f"{len(series)} samples -> {db} and {out_dir / 'sumo_series.csv'}")


def cmd_incidents(args):
    import pandas as pd

    from .network import load_cameras
    from .propagation import build_footprints, run_incidents

    sim, out_dir = load_sim_config(args.sim_config)
    inc_dir = out_dir / "incidents"
    if not args.footprints_only:
        incs = run_incidents(sim, out_dir / "network" / "net.net.xml", load_cameras(out_dir / "cameras.json"),
                             out_dir / "traffic.db", inc_dir, workers=args.workers, fresh=args.fresh)
        print(f"{len(incs)} runs -> {inc_dir / 'incidents.db'}")
    fp, _ = build_footprints(out_dir / "traffic.db", inc_dir)
    with pd.option_context("display.width", 200):
        print(fp.groupby(["kind", "incident_id"], sort=False).agg(
            edges=("edge_id", "count"), max_hop=("hop", "max"),
            extra_delay_veh_h=("extra_delay_veh_h", "sum")).to_string())


def cmd_whatif(args):
    import pandas as pd

    from .network import load_cameras
    from .whatif import impact, run_scenario

    sim, out_dir = load_sim_config(args.sim_config)
    scenarios = (sim.get("whatif") or {}).get("scenarios") or {}
    if args.list or not args.scenario:
        for name, sc in scenarios.items():
            print(f"  {name:16s} {sc}")
        return
    if args.scenario not in scenarios:
        raise SystemExit(f"unknown scenario {args.scenario!r}; whatif.scenarios has: {', '.join(scenarios)}")
    wdir = out_dir / "whatif"
    if not args.impact_only:
        run_scenario(sim, args.scenario, out_dir / "network" / "net.net.xml", load_cameras(out_dir / "cameras.json"),
                     out_dir / "traffic.db", wdir, workers=args.workers)
    junction, edges = impact(args.scenario, out_dir / "traffic.db", wdir)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\n{args.scenario}: {scenarios[args.scenario]}  (mean over {int(junction.days.max())} days)")
        print("Junctions (affected: extra delay on its roads in beyond 2x the placebo's largest):")
        print(junction.drop(columns="scenario").to_string(index=False))
        print("\nEdges with the largest change:")
        top = edges[edges.affected == 1].drop(columns="scenario")
        print(top.round(1).to_string(index=False) if len(top) else "  none beyond the placebo level")


def cmd_predict(args):
    from datetime import date

    from .predict_data import predict_settings

    sim, out_dir = load_sim_config(args.sim_config)
    predict_settings(sim)                             # refuses predict.engine: sumo
    # this command is the SUMO engine (predict.py); with engine: graph it does not run
    raise SystemExit("netstudy predict runs SUMO (predict.engine: sumo), which is disabled. The graph model answers: "
                     f"netstudy graph-model --sim-config {args.sim_config} --change EDGE LEVEL DATE HH:MM MINUTES, "
                     "or the dashboard's Predict.")
    from .predict import case_spec, impact, precompute, run_case  # noqa: E402  (kept for engine: sumo later)

    town = out_dir.name
    if args.precompute:
        precompute(town, sim, out_dir, workers=args.workers, log=lambda m: print(m, flush=True))
        return
    change = {"edge": args.edge, "level": args.level, "minutes": args.minutes} if args.edge else None
    spec = case_spec(town, date.fromisoformat(args.date), args.time, change, out_dir / "traffic.db")
    print(f"{spec['date']} ({spec['kind']}{', ' + spec['event'] if spec['event'] else ''}) {spec['time']}, "
          f"from the studied day {spec['base_day']}", flush=True)
    run_case(spec, sim, out_dir, workers=args.workers, log=lambda m: print(m, flush=True))
    if change:
        junction, _, _ = impact(spec, out_dir / "traffic.db", out_dir)
        print(junction.to_string(index=False))


def cmd_record(args):
    from datetime import date

    from .network import load_cameras
    from .record import record_day

    sim, out_dir = load_sim_config(args.sim_config)
    _simulated_only(sim, "record")
    windows = None
    if args.window:                                   # period=HH:MM-HH:MM; the period names the light
        windows = []
        for w in args.window:
            period, span = w.split("=") if "=" in w else ("morning", w)
            a, b = span.split("-")
            windows.append({"period": period, "start": a, "end": b})
    stats = record_day(date.fromisoformat(args.day), sim, out_dir, load_cameras(out_dir / "cameras.json"), windows,
                       port=args.port)
    for w in stats["windows"]:
        print(f"{stats['day']} {w['period']} {w['start']}-{w['end']}: {w['wall_s']} s, {w['x_real_time']} x real time")
    print(f"{stats['segments']} segments, {stats['frames']} frames -> {out_dir / 'recordings' / stats['day']}")


def cmd_ingest(args):
    from .ingest import Ingest
    from .network import load_cameras

    sim, out_dir = load_sim_config(args.sim_config)
    cams = load_cameras(out_dir / "cameras.json")
    if args.cameras:
        cams = [c for c in cams if c["camera_id"] in args.cameras.split(",")]
    n = Ingest(sim, out_dir, cams).run(args.day, follow=args.follow, log_fn=lambda m: print(m, flush=True))
    print(f"{n} segments ingested -> {out_dir / 'traffic.db'} (vision_samples, observations)")


def cmd_labels(args):
    from .labels import from_sim

    sim, out_dir = load_sim_config(args.sim_config)
    if not args.from_sim:
        raise SystemExit("labels: people label with the label tool (not built yet); use --from-sim for SUMO's labels")
    n = from_sim(sim, out_dir, args.day)
    print(f"SUMO labels: {n['segments']} segments, {n['frames']} frame counts, {n['speeds']} speeds, "
          f"{n['incidents']} incidents -> {out_dir / 'labels.db'}")


def cmd_score(args):
    import pandas as pd

    from .score import vision

    _, out_dir = load_sim_config(args.sim_config)
    summ, _ = vision(out_dir, args.labels)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"Vision against {args.labels} labels (flow: vehicles per 5 min; WAPE and bias over the slots):")
        print(summ.drop(columns=["label_source", "scored_at"]).round(2).to_string(index=False))


def cmd_run(args):
    from datetime import date, datetime

    from .pipeline import finish, run

    sim, out_dir = load_sim_config(args.sim_config)
    _simulated_only(sim, "run")
    log = lambda m: print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)  # noqa: E731
    if args.finish_only:
        finish(sim, out_dir, log)
    else:
        run(sim, out_dir, date.fromisoformat(args.first), args.days, args.redo, log=log, render_days=args.render_days)


def cmd_frames(args):
    from .frames import extract

    _, out_dir = load_sim_config(args.sim_config)
    n = extract(out_dir / "recordings", out_dir / "frames", args.every, args.day)
    print(f"{n} images -> {out_dir / 'frames'}")


def cmd_graph_model(args):
    from datetime import date

    from .graph_model import GraphModel, model_path

    sim, out_dir = load_sim_config(args.sim_config)
    until = date.fromisoformat(args.until) if args.until else None
    m = GraphModel.fit(sim, out_dir, until=until)
    m.save(model_path(out_dir))
    e = m.edges
    print(f"{m.town}: fitted on {len(m.days)} days ({m.days[0]} .. {m.days[-1]}), {len(m.signals)} signal plans, "
          f"saturation flow fitted on {(e.sat_source == 'fitted').sum()} of {int(e.signal.sum())} approaches "
          f"-> {model_path(out_dir)}")
    if args.change:
        edge, level, day, start, minutes = args.change
        r = m.road_change(edge, level, date.fromisoformat(day), start, int(minutes), signals=not args.no_signals)
        on = r["edges"][r["edges"].change_on].groupby("edge_id")[["flow_usual", "flow", "speed_usual", "speed"]].mean()
        print(r["summary"])
        print(on[(on.flow - on.flow_usual).abs() >= 0.5].round(1).to_string())


def cmd_whatif_labels(args):
    from .network import load_cameras
    from .whatif_labels import make

    sim, out_dir = load_sim_config(args.sim_config)
    cases = make(sim, out_dir, load_cameras(out_dir / "cameras.json"), workers=args.workers, only=args.case,
                 log=lambda m: print(m, flush=True))
    print(cases.drop(columns="made_at").to_string(index=False))


def _stamp(m):
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def _real(args):
    from .realtown import is_real

    sim, out_dir = load_sim_config(args.sim_config)
    if not is_real(sim):
        raise SystemExit(f"{args.sim_config} is a simulated town (source.kind: simulation); real-town commands need "
                         "source.kind: real and network.source: osm")
    return sim, out_dir


def cmd_onboard(args):
    from .realtown import onboard

    sim, out_dir = _real(args)
    onboard(sim, out_dir, fetch=args.fetch, log=_stamp)


def cmd_cameras(args):
    from .osm import draw
    from .realtown import place_cameras

    sim, out_dir = _real(args)
    cams = place_cameras(sim, out_dir)
    for c in cams:
        cal = c.get("calibration")
        print(f"  {c['camera_id']}  {c['edge']} -> {c['junction']}  {len(c['lanes'])} lanes  "
              + (f"calibrated, fit {cal['rms_m']:.2f} m" if cal else f"NOT calibrated {c.get('calibration_error', '')}"))
    import json
    import sqlite3

    import pandas as pd
    with sqlite3.connect(out_dir / "traffic.db") as c:
        town = {"nodes": pd.read_sql("SELECT * FROM nodes", c), "edges": pd.read_sql("SELECT * FROM edges", c)}
    town["shapes"] = json.loads((out_dir / "network" / "shapes.json").read_text(encoding="utf-8"))
    town["names"] = json.loads((out_dir / "network" / "names.json").read_text(encoding="utf-8"))
    draw(town, cams, sim.get("name") or out_dir.name.title(), out_dir / "map.png")
    print(f"{len(cams)} cameras -> {out_dir / 'cameras.json'}, map {out_dir / 'map.png'}")


def cmd_observe(args):
    from .observe import observe

    sim, out_dir = _real(args)
    t = observe(sim, out_dir, follow=args.follow, poll_s=args.poll, log=_stamp, check_gpu=not args.no_gpu_check)
    print(f"stored {t['stored']}, rejected {t['rejected']}, segments read {t['ingested']}")


def cmd_receive(args):
    from .inbox import receive

    sim, out_dir = _real(args)
    receive(sim, out_dir, host=args.host, port=args.port, log=_stamp)


def cmd_status(args):
    from .observe import status

    sim, out_dir = _real(args)
    for s in status(sim, out_dir, args.sim_config):
        print(f"[{'x' if s['done'] else ' '}] {s['step']}: {s['found']}")
        if not s["done"]:
            print(f"      next: {s['action']}" + (f"  ->  {s['run']}" if s["run"] else ""))


def main(argv=None):
    p = argparse.ArgumentParser(prog="netstudy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("network", help="build the town network and its cameras")
    s.add_argument("--sim-config", default="sim/maple_falls.yaml")
    s.set_defaults(func=cmd_network)
    s = sub.add_parser("simulate", help="simulate days of traffic (headless SUMO)")
    s.add_argument("--sim-config", default="sim/maple_falls.yaml")
    s.add_argument("--workers", type=int, help="days simulated in parallel")
    s.set_defaults(func=cmd_simulate)
    s = sub.add_parser("state", help="build edge_state from the camera observations")
    s.add_argument("--sim-config", default="sim/maple_falls.yaml")
    s.add_argument("--hide", help="scenario from state.scenarios: hide those cameras and estimate their edges")
    s.set_defaults(func=cmd_state)
    s = sub.add_parser("kpis", help="network KPIs per slot; print a snapshot or a day")
    s.add_argument("--sim-config", default="sim/maple_falls.yaml")
    s.add_argument("--at", help="slot end time, e.g. 2026-10-06T18:00")
    s.add_argument("--day", help="a day hour by hour, e.g. 2026-10-06")
    s.set_defaults(func=cmd_kpis)
    s = sub.add_parser("incidents", help="phase 2: paired incident runs and their footprints")
    s.add_argument("--sim-config", default="sim/maple_falls.yaml")
    s.add_argument("--workers", type=int, help="runs simulated in parallel")
    s.add_argument("--footprints-only", action="store_true", help="recompute footprints from stored runs")
    s.add_argument("--fresh", action="store_true", help="simulate every run again, not only missing ones")
    s.set_defaults(func=cmd_incidents)
    s = sub.add_parser("whatif", help="phase 4: what-if scenario runs and their impact")
    s.add_argument("--sim-config", default="sim/maple_falls.yaml")
    s.add_argument("--scenario", help="name from whatif.scenarios")
    s.add_argument("--list", action="store_true", help="list the scenarios")
    s.add_argument("--impact-only", action="store_true", help="recompute the impact from stored runs")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_whatif)
    s = sub.add_parser("predict", help="simulate a future date / time, optionally with one road's traffic level set")
    s.add_argument("--sim-config", default="sim/maple_falls.yaml")
    s.add_argument("--date", help="e.g. 2026-11-27 (see predict.EVENTS for the mock festivals and holidays)")
    s.add_argument("--time", default="18:00")
    s.add_argument("--edge", help="road to change (edge id)")
    s.add_argument("--level", choices=["busy", "slow", "jammed", "closed"], default="jammed")
    s.add_argument("--minutes", type=int, default=60)
    s.add_argument("--precompute", action="store_true", help="base runs of every holiday and festival")
    s.add_argument("--workers", type=int, default=9)
    s.set_defaults(func=cmd_predict)
    s = sub.add_parser("record", help="film a day with SUMO + CARLA (CARLA server must be running)")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--day", required=True, help="e.g. 2026-10-07")
    s.add_argument("--window", action="append",
                   help="instead of recording.windows: period=HH:MM-HH:MM, repeatable, e.g. night=21:00-21:05 "
                        "(period: morning, noon, evening, night; it sets the light)")
    s.add_argument("--port", type=int, default=2000)
    s.set_defaults(func=cmd_record)
    s = sub.add_parser("ingest", help="recorded video -> observations (SAM 3.1), every segment once")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--day", help="only this day, e.g. 2026-10-07")
    s.add_argument("--cameras", help="comma-separated camera ids (default: all)")
    s.add_argument("--follow", action="store_true", help="live: keep waiting for new segments")
    s.set_defaults(func=cmd_ingest)
    s = sub.add_parser("labels", help="label store: --from-sim loads SUMO's labels (skip_render_read)")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--from-sim", action="store_true")
    s.add_argument("--day")
    s.set_defaults(func=cmd_labels)
    s = sub.add_parser("score", help="score vision against labels (labels.db)")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--labels", choices=["sumo", "people"], default="sumo")
    s.set_defaults(func=cmd_score)
    s = sub.add_parser("run", help="a city end to end: record, labels, ingest per day; then state, KPIs, scores")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--first", default="2026-10-05", help="first day (a Monday)")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--redo", help="run this stage again for every day (record, labels, ingest)")
    s.add_argument("--render-days", type=int, help="simulation only: render the first N days (CARLA + vision), "
                                                     "the rest in SUMO alone (default: source.render_days)")
    s.add_argument("--finish-only", action="store_true", help="only state, KPIs and scores")
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("frames", help="still images from the recorded video")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--day", help="only this day, e.g. 2026-10-07")
    s.add_argument("--every", type=float, default=300, help="seconds between images (300: one per 5-minute slot)")
    s.set_defaults(func=cmd_frames)
    s = sub.add_parser("graph-model", help="fit the graph model on edge_state; optionally try one road change")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--until", help="fit only on days before this date (backtests)")
    s.add_argument("--change", nargs=5, metavar=("EDGE", "LEVEL", "DATE", "HH:MM", "MINUTES"),
                   help="e.g. J1_C closed 2026-11-04 08:00 60")
    s.add_argument("--no-signals", action="store_true", help="implicit capacity, no signal plans")
    s.set_defaults(func=cmd_graph_model)
    s = sub.add_parser("whatif-labels", help="simulation only: paired SUMO runs -> labels.db whatif_labels (scoring)")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--case", help="only this case id, e.g. J1_C_closed_0800_60")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_whatif_labels)
    s = sub.add_parser("onboard", help="real town: OpenStreetMap roads -> graph, cameras, map.png (no SUMO)")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--fetch", action="store_true", help="download the roads in network.osm.bbox (Overpass API)")
    s.set_defaults(func=cmd_onboard)
    s = sub.add_parser("cameras", help="real town: place and calibrate the cameras of the town config again")
    s.add_argument("--sim-config", required=True)
    s.set_defaults(func=cmd_cameras)
    s = sub.add_parser("observe", help="real town: inbox -> SAM 3.1 -> observations -> state, KPIs, graph model")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--follow", action="store_true", help="keep going: pick up and read new recordings as they come")
    s.add_argument("--poll", type=float, default=30.0, help="seconds between inbox checks with --follow")
    s.add_argument("--no-gpu-check", action="store_true", help="do not wait for the other GPU to be idle")
    s.set_defaults(func=cmd_observe)
    s = sub.add_parser("receive", help="real town: HTTP receiver, PUT /recordings/<camera>/<YYYYMMDDTHHMMSS>.mp4")
    s.add_argument("--sim-config", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_receive)
    s = sub.add_parser("status", help="real town: the onboarding checklist")
    s.add_argument("--sim-config", required=True)
    s.set_defaults(func=cmd_status)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
