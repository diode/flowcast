"""Glimmer's 20 test questions (docs/architecture-next.md 4.3): the tool calls it makes against the ones a person
wrote (tests/assist_questions.yaml). Needs the llama-server running (docs/setup.md).

  .venv\\Scripts\\python scripts\\check_assist.py [--sim-config towns/ashford.yaml] [--only 3] [--fast]

--fast: the Predict page's path (Assistant.predict: Glimmer's plan, code's tool calls, Glimmer's insight); code
finds the roads there, so find_road / find_junction are not expected. -> check/assist_fast.json.

Per question: match (every expected call made with its arguments, no unexpected road change), numbers check
(every number in the answer from a tool result), rounds, seconds. -> data/<town>/check/assist.json and
traffic.db predict_eval_assist.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

from netstudy.assist import Assistant
from netstudy.cli import load_sim_config

QUESTIONS = Path(__file__).resolve().parents[1] / "tests" / "assist_questions.yaml"


def norm(name: str, args: dict) -> dict:
    """Comparable arguments; a road change's roads as one set ("roads": the road and its `also` roads, any order;
    a change of two or three roads is not checked for each road's level)."""
    args, out = dict(args), {}
    if name == "set_road" and "edge" in args:
        also = args.pop("also", None) or []
        also = also.split(",") if isinstance(also, str) else [x["edge"] if isinstance(x, dict) else x for x in also]
        out["roads"] = ",".join(sorted({str(args.pop("edge")), *also}))
        if also:
            args.pop("level", None)
    for k, v in args.items():
        v = str(v).strip()
        if k == "time" and len(v) == 4:
            v = "0" + v
        out[k] = v.lower() if k == "level" else v
    return out


def matches(spec: dict, calls: list[dict], fast: bool = False) -> tuple[bool, list[str]]:
    ok_changes = [c for c in calls if c["name"] == "set_road" and "error" not in c["result"]]
    why = []
    if spec.get("no_change"):
        if ok_changes:
            why.append(f"a road change ran: {ok_changes[0]['args']}")
        return not why, why
    expected = [(n, norm(n, a or {})) for e in spec["expect"] for n, a in e.items()
                if not (fast and n in ("find_road", "find_junction"))]
    for name, args in expected:
        if not any(c["name"] == name and all(norm(name, c["args"]).get(k) == v for k, v in args.items())
                   for c in calls):
            why.append(f"missing {name}({args})")
    want = [a for n, a in expected if n == "set_road"]
    for c in ok_changes:
        a = norm("set_road", c["args"])
        if not any(all(a.get(k) == v for k, v in w.items()) for w in want):
            why.append(f"unexpected set_road({a})")
    return not why, why


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sim-config", default="towns/ashford.yaml")
    p.add_argument("--only", type=int, help="only question N (1-based)")
    p.add_argument("--questions", help="question file (default: tests/assist_questions_<town>.yaml, Ashford's "
                                       "tests/assist_questions.yaml)")
    p.add_argument("--fast", action="store_true", help="the Predict page's path (Assistant.predict)")
    args = p.parse_args(argv)
    sim, out_dir = load_sim_config(args.sim_config)
    own = QUESTIONS.with_name(f"assist_questions_{out_dir.name}.yaml")
    qfile = Path(args.questions) if args.questions else own if own.exists() else QUESTIONS
    print(f"questions: {qfile.name}", flush=True)
    spec = yaml.safe_load(qfile.read_text(encoding="utf-8"))
    today = spec["today"] if isinstance(spec["today"], date) else date.fromisoformat(spec["today"])
    a = Assistant(out_dir.name, sim, out_dir, today=today)
    rows = []
    for i, qs in enumerate(spec["questions"], start=1):
        if args.only and i != args.only:
            continue
        ans = a.predict(qs["q"]) if args.fast else a.ask(qs["q"])
        ok, why = matches(qs, ans.calls, args.fast)
        calls = " ".join(f"{c['name']}({','.join(f'{k}={v}' for k, v in c['args'].items())})"
                         + ("!" if "error" in c["result"] else "") for c in ans.calls)
        print(f"{i:2d} {'OK ' if ok else 'NO '} {ans.seconds:5.1f}s r{ans.rounds} "
              f"numbers {'ok' if ans.numbers_ok else 'NOT ok ' + ','.join(ans.unknown_numbers)}\n"
              f"   Q: {qs['q']}\n   calls: {calls}\n" + "".join(f"   why: {w}\n" for w in why)
              + f"   A: {ans.text.strip()[:300]}", flush=True)
        rows.append({"n": i, "question": qs["q"], "match": ok, "why": "; ".join(why), "calls": calls,
                     "numbers_ok": ans.numbers_ok, "unknown_numbers": ",".join(ans.unknown_numbers),
                     "rounds": ans.rounds, "seconds": round(ans.seconds, 1), "answer": ans.text,
                     "tool_calls": [{"name": c["name"], "args": c["args"], "result": c["result"]} for c in ans.calls]})
    r = pd.DataFrame(rows)
    print(f"\nmatch {r.match.sum()} / {len(r)}, numbers ok {r.numbers_ok.sum()} / {len(r)}, "
          f"median {r.seconds.median():.0f} s per question")
    (out_dir / "check").mkdir(exist_ok=True)
    (out_dir / "check" / f"assist{'_fast' if args.fast else ''}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.only:
        with sqlite3.connect(out_dir / "traffic.db") as c:
            r.drop(columns="tool_calls").assign(scored_at=datetime.now().isoformat(timespec="seconds")).to_sql(
                "predict_eval_assist" + ("_fast" if args.fast else ""), c, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
