"""End-to-end status check: every stage of the flow (docs/implementation-next.md), per study city, OK or GAP.

  .venv\\Scripts\\python scripts\\check_e2e.py                 # all cities
  .venv\\Scripts\\python scripts\\check_e2e.py --city ashford

CPU only: reads files and tables, fits the graph model, answers one road change and calls the Predict tools
offline. Starts no GPU job, no CARLA, no llama-server. A GAP line says what is missing and which step fills it.
Output also written to data/e2e_status.md.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CITIES = ["ashford", "brookhaven", "clearwater"]
rows: list[tuple[str, str, str, str, str]] = []          # city, stage, OK/GAP, detail, fills


def add(city, stage, ok, detail, fills=""):
    rows.append((city, stage, "OK" if ok else "GAP", detail, "" if ok else fills))


def table_count(db: Path, sql: str):
    if not db.exists():
        return None
    try:
        with sqlite3.connect(db) as c:
            return c.execute(sql).fetchone()[0]
    except sqlite3.Error:
        return None


def check_city(name: str):
    import yaml
    cfg_path = ROOT / "towns" / f"{name}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    d = ROOT / "data" / name
    db, lab = d / "traffic.db", d / "labels.db"
    town = (cfg.get("network") or {}).get("town") or {}

    # 1 config
    add(name, "config: source switches", bool(cfg.get("source")), "source.kind / render_days" if cfg.get("source") else
        "no source block", "step 5: copy Ashford's source block")
    add(name, "config: demand", bool(cfg.get("demand")), "zones defined" if cfg.get("demand") else "no demand block",
        "step 5: zones per junction")
    add(name, "config: recording windows", bool((cfg.get("recording") or {}).get("windows")),
        "windows set" if (cfg.get("recording") or {}).get("windows") else "no recording windows",
        "step 5: size from Ashford's render speed")
    add(name, "config: signal plans", bool(town.get("signals")),
        f"{len(town.get('signals') or {})} junctions" if town.get("signals") else "no network.town.signals",
        "step 5: timing sheet per junction (decode net.net.xml once, as for Ashford)")
    add(name, "config: whatif_labels cases", bool((cfg.get("whatif_labels") or {}).get("cases")),
        f"{len((cfg.get('whatif_labels') or {}).get('cases') or [])} cases" if cfg.get("whatif_labels") else "none",
        "step 5: test road changes")

    # 2 network
    net_ok = all((d / "network" / f).exists() for f in ("net.net.xml", "net.xodr")) and (d / "cameras.json").exists()
    add(name, "network, OpenDRIVE, cameras, map", net_ok and (d / "map.png").exists(), "built" if net_ok else "missing",
        "netstudy network")

    # 3 data
    vis = table_count(db, "SELECT COUNT(*) FROM observations WHERE source = 'vision'")
    sumo = table_count(db, "SELECT COUNT(*) FROM observations WHERE source = 'sumo'")
    days = table_count(db, "SELECT COUNT(DISTINCT substr(ts, 1, 10)) FROM observations WHERE source = 'sumo'")
    add(name, "rendered day: recorded + ingested (vision)", bool(vis), f"{vis or 0} rows", "netstudy run (vision GPU)")
    add(name, "direct days (sumo)", bool(sumo), f"{sumo or 0} rows over {days or 0} days", "netstudy run (CPU)")
    add(name, "labels (from sim)", bool(table_count(lab, "SELECT COUNT(*) FROM crossing_labels")),
        f"{table_count(lab, 'SELECT COUNT(*) FROM crossing_labels') or 0} crossing labels", "netstudy labels --from-sim")
    st = table_count(db, "SELECT COUNT(*) FROM edge_state")
    add(name, "state + KPIs", bool(st) and bool(table_count(db, "SELECT COUNT(*) FROM network_kpi")),
        f"edge_state {st or 0} rows", "netstudy run --finish-only")
    add(name, "vision scores + direct_check", bool(table_count(db, "SELECT COUNT(*) FROM vision_eval")) and
        bool(table_count(db, "SELECT COUNT(*) FROM direct_check_summary")),
        "vision_eval, direct_check_summary", "netstudy run --finish-only")
    add(name, "detection on camera data (N3)", bool(table_count(db, "SELECT COUNT(*) FROM detect_events")),
        "detector events", "netstudy detect on edge_state (not built for camera data yet)")
    add(name, "forecast on camera data (N3)", bool(table_count(db, "SELECT COUNT(*) FROM forecast_eval")),
        "forecast_eval", "forecast.py on vision/direct observations (not re-run yet)")

    # 4 graph model
    if not st:
        add(name, "graph model fit + road change", False, "no edge_state", "after state + KPIs")
        return
    try:
        from netstudy.cli import load_sim_config
        from netstudy.graph_model import GraphModel
        sim, out = load_sim_config(cfg_path)
        t = time.time()
        m = GraphModel.fit(sim, out, town=name)
        fit_s = time.time() - t
        e = m.edges[m.edges.kind == "road"].edge_id.iloc[0]
        t = time.time()
        r = m.road_change(e, "closed", date(2026, 11, 4), "08:00", 60)
        add(name, "graph model fit + road change", True,
            f"fit {fit_s:.1f} s on {len(m.days)} days; {e} closed: {len(r['summary']['affected_edges'])} roads "
            f"affected, {time.time() - t:.2f} s; signals {'yes' if m.signals else 'no (implicit capacity)'}")
    except Exception as ex:                                             # noqa: BLE001
        add(name, "graph model fit + road change", False, f"{type(ex).__name__}: {ex}"[:150], "graph_model.py")
        return
    nwl = table_count(lab, "SELECT COUNT(DISTINCT case_id) FROM whatif_cases")
    add(name, "answer key (whatif_labels)", bool(nwl), f"{nwl or 0} cases", "netstudy whatif-labels (CPU)")
    pe = table_count(db, "SELECT COUNT(*) FROM predict_eval_summary WHERE kind = 'road_change'")
    add(name, "prediction scores (predict_eval)", bool(pe), f"{pe or 0} road-change rows", "scripts/check_predict.py")
    add(name, "Predict scores: logged incidents, forecast", bool(table_count(
        db, "SELECT COUNT(*) FROM predict_eval WHERE kind IN ('incident', 'forecast')")),
        "kinds incident / forecast in predict_eval", "check_predict.py: add 5.5's incident backtest and forecast")

    # 5 Predict tools and Glimmer
    try:
        from netstudy.assist import Tools
        tools = Tools.load(name, sim, out)
        c = tools.call("set_road", {"edge": e, "level": "closed", "minutes": 60, "date": "2026-11-04", "time": "09:00"})
        ok = "case" in c and "roads" in tools.call("impact", {"case": c["case"]}) and \
            "actions" in tools.call("actions", {"case": c["case"]})
        names = sorted(set(tools.places.junction.values()))[:3]
        add(name, "Predict tools (offline)", ok, f"set_road / impact / actions; names {names}", "assist.py")
    except Exception as ex:                                             # noqa: BLE001
        add(name, "Predict tools (offline)", False, f"{type(ex).__name__}: {ex}"[:150], "assist.py")
    with sqlite3.connect(out / "traffic.db") as c:                     # the overview's "What our studies found"
        n = c.execute("SELECT count(*) FROM sqlite_master WHERE name = 'study_closures'").fetchone()[0]
        n = c.execute("SELECT count(*) FROM study_closures").fetchone()[0] if n else 0
    add(name, "closure study (overview findings)", n > 0, f"{n} closures in study_closures",
        "scripts/study_closures.py")
    na = table_count(db, "SELECT SUM(match) FROM predict_eval_assist")
    tot = table_count(db, "SELECT COUNT(*) FROM predict_eval_assist")
    add(name, "Glimmer test questions", bool(tot) and na == tot, f"{na or 0} / {tot or 0} match",
        "tests/assist_questions*.yaml for this city + scripts/check_assist.py (Glimmer session)")


def check_shared():
    pv = (ROOT / "dashboard" / "predict_view.py").read_text(encoding="utf-8")
    sumo_view = bool(re.search(r"_launch\(|predict\.base_truth|predict\.run_case|predict\.impact\(", pv))
    add("all", "Predict view on the graph model", (not sumo_view) and ("graph_model" in pv or "assist" in pv),
        "predict_view.py still launches SUMO runs (_launch, predict.impact)" if sumo_view else "graph engine",
        "step 4: swap the engine to assist.Tools / graph_model")
    add("all", "Predict view: Glimmer question box", "Assistant" in pv, "question box",
        "step 4: question box, tool results as they come, numbers check fallback")
    add("all", "Predict view: source badges", "source_badges(" in pv and "accuracy_badge(" in pv,
        "observed / estimated / assumed + accuracy badge",
        "step 4")
    import yaml

    from netstudy.predict_data import predict_settings
    towns = sorted((ROOT / "towns").glob("*.yaml"))
    eng = all(((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("predict") or {}).get("engine") == "graph"
              for p in towns)
    try:
        predict_settings({"predict": {"engine": "sumo"}})
        refuse = False
    except SystemExit:
        refuse = True
    add("all", "predict.engine switch + refusal (1.6)", eng and refuse,
        f"engine: graph in {len(towns)} towns, sumo refused" if eng and refuse else "config + refusal",
        "step 1 / 4: predict.engine: graph in towns/*.yaml; refuse sumo")
    pipe = (ROOT / "src" / "netstudy" / "pipeline.py").read_text(encoding="utf-8")
    add("all", "CARLA adapter looked up (not fixed 0)", "-graphicsadapter=0" not in pipe,
        "pipeline.py passes -graphicsadapter=0 (the adapter order changes)" if "-graphicsadapter=0" in pipe else "looked up",
        "before step 5 recording: use scripts/dxgi_list.py's index")
    td = (ROOT / "tests" / "test_dashboard.py").read_text(encoding="utf-8")
    add("all", "dashboard tests on a study city", "ashford" in td, "tests target maple_falls (skipped)" if "ashford"
        not in td else "ashford", "step 4: test_dashboard.py on Ashford")
    add("all", "package layout (implementation-next 1.1)", (ROOT / "src" / "netstudy" / "predict").is_dir(),
        "flat modules", "later")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    add("all", "test suite", r.returncode == 0, last, "fix failing tests")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=CITIES)
    a = p.parse_args(argv)
    for c in [a.city] if a.city else CITIES:
        check_city(c)
    check_shared()
    lines = [f"# End-to-end status ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "| City | Stage | Status | Detail | Fills the gap |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {s} | {'**GAP**' if o == 'GAP' else 'OK'} | {d} | {f} |" for c, s, o, d, f in rows]
    gaps = sum(1 for r in rows if r[2] == "GAP")
    lines += ["", f"{len(rows) - gaps} OK, {gaps} gaps."]
    text = "\n".join(lines)
    (ROOT / "data" / "e2e_status.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
