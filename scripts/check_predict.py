"""Scoring prediction (docs/architecture-next.md section 5.5): the graph model, fitted on the days before the test
days, against what happened.

  .venv\\Scripts\\python scripts\\check_predict.py --sim-config towns/ashford.yaml

  1. expected day: each test day's 15-minute flow and speed against its day kind's expected day (fitted on the
     days before): WAPE, bias, share of observed values inside the 10-90% range. Reference: the cameras.
  2. road change: each case of labels.db whatif_cases against its paired SUMO runs (whatif_labels, label_source
     sumo, simulation only), the true change being change run - base run, mean over the test days, per road and
     15 minutes from the change's start to 2 hours after it ends. Per case: WAPE of the change in flow, sign right
     on the roads that truly changed, roads hit / missed / false (|mean change| >= 2 veh per 5 min or 5 km/h while
     the change is on), when it was over (from which slot every changed road stays quiet), in both graph-model variants
     (signal plans, implicit capacity).

Results: traffic.db predict_eval (one row per kind, case, variant, road, metric) and predict_eval_summary.
Simulation scores are method accuracy in simulation, never the real town's.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

from netstudy.cli import load_sim_config
from netstudy.graph_model import GraphModel

FLOW_CHANGED = 2.0      # veh per 5 minutes (24 veh/h)
SPEED_CHANGED = 5.0     # km/h


def slot15(ts: pd.Series) -> pd.Series:
    """Observation stamps are slot ends; 15-minute slots by their start."""
    return (pd.to_datetime(ts) - pd.Timedelta(seconds=1)).dt.floor("15min")


def expected_day(m: GraphModel, out_dir, days: list[date]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(out_dir / "traffic.db") as c:
        st = pd.read_sql("SELECT edge_id, ts, flow, speed FROM edge_state", c)
    st["slot_start"] = slot15(st.ts)
    st = st[st.slot_start.dt.date.isin(days)]
    q = st.groupby(["edge_id", "slot_start"])[["flow", "speed"]].mean().reset_index()
    q["slot"], q["day"] = q.slot_start.dt.strftime("%H:%M"), q.slot_start.dt.date
    rows = []
    for d, g in q.groupby("day"):
        e = m.expected_day(d)
        rows.append(g.merge(e, on=["edge_id", "slot"]))
    x = pd.concat(rows)
    per = []
    for (edge,), g in x.groupby(["edge_id"]):
        f = g.dropna(subset=["flow"])
        s = g.dropna(subset=["speed", "speed_p50"])
        per.append({"edge_id": edge, "flow_wape": (f.flow - f.flow_p50).abs().sum() / f.flow.sum(),
                    "flow_bias": (f.flow_p50.sum() - f.flow.sum()) / f.flow.sum(),
                    "flow_in_range": ((f.flow >= f.flow_p10) & (f.flow <= f.flow_p90)).mean(),
                    "speed_mae": (s.speed - s.speed_p50).abs().mean(),
                    "speed_in_range": ((s.speed >= s.speed_p10) & (s.speed <= s.speed_p90)).mean()})
    per = pd.DataFrame(per)
    f = x.dropna(subset=["flow"])
    total = {"flow_wape": (f.flow - f.flow_p50).abs().sum() / f.flow.sum(),
             "flow_bias": (f.flow_p50.sum() - f.flow.sum()) / f.flow.sum(),
             "flow_in_range": ((f.flow >= f.flow_p10) & (f.flow <= f.flow_p90)).mean(),
             "speed_in_range": per.speed_in_range.mean()}
    return per, pd.DataFrame([total])


def true_change(out_dir, case_id: str) -> pd.DataFrame:
    from netstudy.predict_data import paired_truth          # shared with the technical view

    return paired_truth(out_dir, case_id)


def score_case(m: GraphModel, case: pd.Series, days: list[str], truth: pd.DataFrame, signals: bool) -> tuple:
    preds = []
    for d in days:
        r = m.road_change(case.edge, case.level, date.fromisoformat(d), case.start, int(case.minutes), signals=signals)
        e = r["edges"].assign(day=d)
        preds.append(e)
    p = pd.concat(preds).groupby(["edge_id", "slot"]).agg(p_flow=("flow_change", "mean"),
                                                          p_speed=("speed_change", "mean"),
                                                          on=("change_on", "first")).reset_index()
    start = datetime.strptime(case.start, "%H:%M")
    end_on = (start + pd.Timedelta(minutes=int(case.minutes))).strftime("%H:%M")
    last = (start + pd.Timedelta(minutes=int(case.minutes) + 120)).strftime("%H:%M")
    start_slot = f"{start.hour:02d}:{start.minute // 15 * 15:02d}"
    t = truth[(truth.slot >= start_slot) & (truth.slot < last)]
    x = t.merge(p, on=["edge_id", "slot"], how="left").fillna({"p_flow": 0.0})
    x["on"] = x.slot < end_on
    wape = (x.p_flow - x.d_flow).abs().sum() / max(1e-9, x.d_flow.abs().sum())
    during = x[x.on].groupby("edge_id")[["d_flow", "p_flow", "d_speed", "p_speed"]].mean()
    changed_t = (during.d_flow.abs() >= FLOW_CHANGED) | (during.d_speed.abs() >= SPEED_CHANGED)
    changed_p = (during.p_flow.abs() >= FLOW_CHANGED) | (during.p_speed.abs() >= SPEED_CHANGED)
    sign = np.sign(during.d_flow[changed_t]) == np.sign(during.p_flow[changed_t])

    def over(col, roads):
        """The first slot from which every changed road stays under the threshold (paired runs drift apart by about
        that much on every road once the change has started, so roads that never changed are not looked at)."""
        g = x[x.edge_id.isin(roads)].groupby("slot")[col].apply(lambda s: s.abs().max()).sort_index()
        loud = g[g >= FLOW_CHANGED]
        if loud.empty:
            return None
        after = g.index[g.index > loud.index[-1]]
        return after[0] if len(after) else last

    before = truth[truth.slot < start_slot].d_flow.abs().max()
    summary = {"case": case.case_id, "variant": "signals" if signals else "implicit", "wape": wape,
               "sign_ok": sign.mean() if len(sign) else np.nan,
               "hit": int((changed_t & changed_p).sum()), "missed": int((changed_t & ~changed_p).sum()),
               "false": int((~changed_t & changed_p).sum()),
               "missed_edges": " ".join(during.index[changed_t & ~changed_p]),
               "false_edges": " ".join(during.index[~changed_t & changed_p]),
               "over_true": over("d_flow", during.index[changed_t]), "over_pred": over("p_flow", during.index[changed_p]),
               "noise_before": before if pd.notna(before) else 0.0}
    rows = [{"kind": "road_change", "case": case.case_id, "variant": summary["variant"], "edge_id": e,
             "metric": metric, "predicted": float(r[f"p_{metric}"]), "observed": float(r[f"d_{metric}"]),
             "sign_ok": bool(np.sign(r[f"p_{metric}"]) == np.sign(r[f"d_{metric}"])), "label_source": "sumo"}
            for e, r in during.iterrows() for metric in ("flow", "speed") if pd.notna(r[f"d_{metric}"])]
    return summary, rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="towns/ashford.yaml")
    args = p.parse_args(argv)
    sim, out_dir = load_sim_config(args.sim_config)
    with sqlite3.connect(out_dir / "labels.db") as c:
        cases = pd.read_sql("SELECT * FROM whatif_cases", c)
    if cases.empty:
        raise SystemExit("no whatif_cases: run `netstudy whatif-labels` first")
    days = sorted(cases.day.unique())
    first = date.fromisoformat(days[0])
    m = GraphModel.fit(sim, out_dir, until=first)
    print(f"Graph model fitted on {len(m.days)} days before {first} ({m.days[0]} .. {m.days[-1]}); "
          f"test days {', '.join(days)}\n")
    now = datetime.now().isoformat(timespec="seconds")
    eval_rows, summaries = [], []

    print("1. Expected day (reference: the cameras on the test days)")
    per, total = expected_day(m, out_dir, [date.fromisoformat(d) for d in days])
    print(per.round(2).to_string(index=False))
    print("   all roads:", {k: round(float(v), 3) for k, v in total.iloc[0].items()}, "\n")
    for r in per.itertuples():
        eval_rows.append({"kind": "expected_day", "case": "test_days", "variant": "-", "edge_id": r.edge_id,
                          "metric": "flow_wape", "predicted": np.nan, "observed": r.flow_wape, "sign_ok": None,
                          "label_source": "camera"})
    summaries.append({"kind": "expected_day", "case": "test_days", "variant": "-",
                      "wape": float(total.flow_wape[0]), "in_range": float(total.flow_in_range[0])})

    print("2. Road change against paired SUMO runs (method accuracy in simulation, label_source sumo)")
    for cid, g in cases.groupby("case_id", sort=False):
        case = g.iloc[0]
        truth = true_change(out_dir, cid)
        for signals in (True, False):
            s, rows = score_case(m, case, sorted(g.day), truth, signals)
            s["rerouted"] = int(g.rerouted.mean())
            s["teleports_extra"] = int((g.teleports_change - g.teleports_base).mean())
            summaries.append({"kind": "road_change", **s})
            eval_rows += rows
    rc = pd.DataFrame([s for s in summaries if s["kind"] == "road_change"])
    show = ["case", "variant", "wape", "sign_ok", "hit", "missed", "false", "over_true", "over_pred", "noise_before",
            "rerouted", "teleports_extra", "missed_edges", "false_edges"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(rc[show].round(2).to_string(index=False))
    for v, g in rc.groupby("variant"):
        print(f"   {v}: mean WAPE {g.wape.mean():.2f}, sign right {g.sign_ok.mean():.0%}, "
              f"roads hit {g.hit.sum()} / missed {g.missed.sum()} / false {g['false'].sum()}")

    with sqlite3.connect(out_dir / "traffic.db") as c:
        pd.DataFrame(eval_rows).assign(scored_at=now).to_sql("predict_eval", c, if_exists="replace", index=False)
        pd.DataFrame(summaries).assign(scored_at=now).to_sql("predict_eval_summary", c, if_exists="replace", index=False)
    print(f"\n-> {out_dir / 'traffic.db'} predict_eval ({len(eval_rows)} rows), predict_eval_summary")


if __name__ == "__main__":
    main()
