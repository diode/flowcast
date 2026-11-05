"""Glimmer drives Predict (docs/architecture-next.md 4.3): a question in words -> tool calls over the graph model ->
an answer. Glimmer (llama.cpp server, OpenAI-style chat with tools) plans and explains; code decides.

  tools   find_road, find_junction, day_kind, expected_state, set_road, impact, detour, actions, history: plain
          functions over the graph model and the stored tables; read only (a road change lives in memory only)
  checks  every call before it runs: road and junction ids (Places), levels (days.LEVELS), dates in the calendar's
          range, 30-120 minutes, HH:MM; a failed check goes back to Glimmer as an error, never to the model
  numbers every number in the answer must appear in a tool result of that answer (or the question); otherwise
          the answer is marked and the tool tables are what to show
Flows are vehicles per hour here (the graph model's veh per 5 minutes x 12), speeds km/h.

  Assistant(town, sim, out_dir).ask("Elm Park to Ashford Centre jammed tomorrow 8am for an hour")
  Assistant(...).predict(question, when=(page date, "08:00"))   the Predict page's fast path (about 6 s, not a
          minute): Glimmer only reads the question into a plan and words the insight; code runs the tools

The llama-server runs on the workspace's Traffic Assistant GPU (glimmer_server, workspace.glimmer_gpu), --jinja.
"""
from __future__ import annotations

import difflib
import json
import re
import sqlite3
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import actions as rules
from .days import CAL_FROM, CAL_TO, LEVELS, day_kind, event_on
from .graph_model import GraphModel, model_path
from .places import IMPACT, Places, city_name, impact_word, time_lost

SERVER = "http://127.0.0.1:8080"
MINUTES = (30, 120)
MAX_ROUNDS = 8
TOP_ROADS = 10
TOOLS = [
    {"name": "find_road", "description": "Find a road (one direction) by place names, e.g. 'Elm Park to Ashford "
     "Centre'. Returns candidate edge ids with their names; use the edge id in other tools.",
     "parameters": {"text": {"type": "string"}}, "required": ["text"]},
    {"name": "find_junction", "description": "Find a junction by name. Returns candidate junction ids.",
     "parameters": {"text": {"type": "string"}}, "required": ["text"]},
    {"name": "day_kind", "description": "What kind of day a date is: working, weekend, holiday or festival, and its "
     "event.", "parameters": {"date": {"type": "string", "description": "YYYY-MM-DD"}}, "required": ["date"]},
    {"name": "expected_state", "description": "The usual traffic on every road (or one road) at a date and time, no "
     "road change: flow veh/h and speed km/h, median and 10-90% range.",
     "parameters": {"date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM"},
                    "edge": {"type": "string", "description": "optional edge id"}}, "required": ["date", "time"]},
    {"name": "set_road", "description": "Predict a road change: the road set to busy, slow, jammed or closed for "
     "30-120 minutes from a date and time. Returns a case id for impact and actions.",
     "parameters": {"edge": {"type": "string"}, "level": {"type": "string", "enum": list(LEVELS)},
                    "minutes": {"type": "integer"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM"},
                    "also": {"type": "array", "description": "more roads changed at the same time (at most 2)",
                             "items": {"type": "object", "properties": {"edge": {"type": "string"},
                                                                        "level": {"type": "string",
                                                                                  "enum": list(LEVELS)}}}}},
     "required": ["edge", "level", "minutes", "date", "time"]},
    {"name": "impact", "description": "The affected roads and junctions of a case from set_road: change in flow and "
     "speed while it lasts, queue, extra delay, when it is over.",
     "parameters": {"case": {"type": "string"}}, "required": ["case"]},
    {"name": "detour", "description": "The shortest way round a road (by length), junction by junction.",
     "parameters": {"edge": {"type": "string"}}, "required": ["edge"]},
    {"name": "actions", "description": "What to do for a case from set_road: diversions, signal retiming, officers, "
     "message signs (rules, not the model).", "parameters": {"case": {"type": "string"}}, "required": ["case"]},
    {"name": "history", "description": "Past logged incidents on a road or at a junction.",
     "parameters": {"target": {"type": "string", "description": "edge id or junction id"}}, "required": ["target"]},
]


class CheckError(ValueError):
    """A tool call that does not pass the checks: told to Glimmer, nothing runs."""


# ---------------------------------------------------------------------- filmed windows
def windows(sim: dict | None) -> list[dict]:
    """The town's filmed windows (recording.windows); Predict answers only inside them (implementation-next step 4).
    None in the config: the whole day is filmed."""
    return list(((sim or {}).get("recording") or {}).get("windows") or [])


def in_windows(ws: list[dict], hhmm: str) -> bool:
    return not ws or any(w["start"] <= hhmm < w["end"] for w in ws)


def covered(ws: list[dict]) -> str:
    return ", ".join(f"{w['period']} {w['start']}–{w['end']}" for w in ws) or "the whole day"


def _date(s: str) -> date:
    try:
        d = date.fromisoformat(str(s))
    except ValueError:
        raise CheckError(f"date {s!r}: expected YYYY-MM-DD")
    if not CAL_FROM - timedelta(days=31) <= d <= CAL_TO:
        raise CheckError(f"date {d}: outside the calendar ({CAL_FROM - timedelta(days=31)} .. {CAL_TO})")
    return d


def _time(s: str) -> str:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(s).strip())
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise CheckError(f"time {s!r}: expected HH:MM")
    return f"{int(m.group(1)):02d}:{m.group(2)}"


@dataclass
class Tools:
    town: str
    model: GraphModel
    places: Places
    nodes: pd.DataFrame
    edges: pd.DataFrame
    labels_db: Path
    cases: dict = field(default_factory=dict)
    windows: list = field(default_factory=list)          # filmed windows; times outside them are refused

    @classmethod
    def load(cls, town: str, sim: dict, out_dir: Path) -> "Tools":
        path = model_path(out_dir)
        model = GraphModel.load(path) if path.exists() else GraphModel.fit(sim, out_dir, town=town)
        with sqlite3.connect(out_dir / "traffic.db") as c:
            nodes = pd.read_sql("SELECT * FROM nodes", c)
            edges = pd.read_sql("SELECT * FROM edges", c)
        edges = edges.merge(model.edges[["edge_id", "free_kmh"]], on="edge_id", how="left")
        return cls(town, model, Places(town, nodes, edges), nodes, edges, out_dir / "labels.db",
                   windows=windows(sim))

    # ------------------------------------------------------------------ checks
    def _edge(self, e) -> str:
        if str(e) not in set(self.edges.edge_id):
            raise CheckError(f"edge {e!r} is not a road in {self.town}: use find_road first")
        return str(e)

    @property
    def _cams(self) -> set:
        return set(self.edges.edge_id[self.edges.camera_id.notna()])

    def _filmed(self, t: str) -> str:
        if not in_windows(self.windows, t):
            raise CheckError(f"time {t}: outside the filmed windows ({covered(self.windows)}); there is no camera "
                             "flow to predict from, so this time is refused. Say so; do not move the time")
        return t

    def _case(self, c) -> dict:
        if str(c) not in self.cases:
            raise CheckError(f"case {c!r} not known: call set_road first")
        return self.cases[str(c)]

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"no tool {name!r}"}
        try:
            return fn(**args)
        except CheckError as e:
            return {"error": str(e)}
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}

    # ------------------------------------------------------------------ tools
    def t_find_road(self, text: str) -> dict:
        names = {e: self.places.road(e) for e in self.edges.edge_id}
        q = _norm(text)
        parts = re.split(r"\s+(?:to|towards|into|->|→)\s+", q)
        scored = []
        for e, n in names.items():
            a, b = (_norm(x) for x in n.split(" → "))
            if len(parts) == 2:
                s = (difflib.SequenceMatcher(None, parts[0], a).ratio() + difflib.SequenceMatcher(None, parts[1], b).ratio()) / 2
            else:
                s = difflib.SequenceMatcher(None, q, _norm(n)).ratio()
            scored.append((s, e, n))
        scored.sort(reverse=True)
        found = [{"edge": e, "name": n, "match": round(s, 2)} for s, e, n in scored[:3] if s >= 0.6]
        return {"candidates": found} if found else {"candidates": [], "note": f"no road matches {text!r}"}

    def t_find_junction(self, text: str) -> dict:
        q = _norm(text)
        scored = sorted(((difflib.SequenceMatcher(None, q, _norm(n)).ratio(), j, n)
                         for j, n in self.places.junction.items()), reverse=True)
        found = [{"junction": j, "name": n, "match": round(s, 2)} for s, j, n in scored[:3] if s >= 0.6]
        return {"candidates": found} if found else {"candidates": [], "note": f"no junction matches {text!r}"}

    def t_day_kind(self, date: str) -> dict:
        d = _date(date)
        kind, assumed = self.model.kind_for(d)
        e = event_on(d) or {}
        return {"date": d.isoformat(), "weekday": d.strftime("%A"), "kind": day_kind(d), "event": e.get("name"),
                "model_uses": kind + (" (assumed: no recorded day of this kind)" if assumed else "")}

    def t_expected_state(self, date: str, time: str, edge: str | None = None) -> dict:
        d, t = _date(date), self._filmed(_time(time))
        u = self.model.expected_day(d)
        u = u[u.slot == f"{t[:2]}:{int(t[3:]) // 15 * 15:02d}"]
        if edge:
            u = u[u.edge_id == self._edge(edge)]
        rows = [{"edge": r.edge_id, "road": self.places.road(r.edge_id), "flow_vph": round(r.flow_p50 * 12),
                 "flow_range_vph": [round(r.flow_p10 * 12), round(r.flow_p90 * 12)],
                 "speed_kmh": None if pd.isna(r.speed_p50) else round(r.speed_p50, 1),
                 "source": "assumed" if r.assumed else ("observed" if r.edge_id in self._cams else "estimated")}
                for r in u.itertuples()]
        return {"date": d.isoformat(), "time": t, "day_kind": day_kind(d), "assumed": bool(u.assumed.any()),
                "roads": rows}

    def t_set_road(self, edge: str, level: str, minutes, date: str, time: str, also=None) -> dict:
        """`also`: more roads changed at the same time, [{"edge", "level"}] (two or three roads closed together)."""
        e = self._edge(edge)
        if level not in LEVELS:
            raise CheckError(f"level {level!r}: one of {', '.join(LEVELS)}")
        more = {}
        for x in also or []:
            k, lv = self._edge(x.get("edge")), x.get("level")
            if lv not in LEVELS:
                raise CheckError(f"level {lv!r}: one of {', '.join(LEVELS)}")
            if k != e:
                more[k] = lv
        if len(more) > MAX_ALSO:
            raise CheckError(f"{len(more) + 1} roads: at most {MAX_ALSO + 1} roads changed together")
        try:
            minutes = int(minutes)
        except (TypeError, ValueError):
            raise CheckError(f"minutes {minutes!r}: a whole number")
        if not MINUTES[0] <= minutes <= MINUTES[1]:
            raise CheckError(f"minutes {minutes}: {MINUTES[0]}-{MINUTES[1]} only")
        d, t = _date(date), self._filmed(_time(time))
        r = self.model.road_change(e, level, d, t, minutes, also=more)
        cid = f"c{len(self.cases) + 1}"
        self.cases[cid] = {"edge": e, "level": level, "also": more, "minutes": minutes, "date": d.isoformat(),
                           "time": t, "result": r}
        s = r["summary"]
        road, lv = change_label(self.places, e, level, more)
        return {"case": cid, "road": road, "level": lv, "date": d.isoformat(), "from": t,
                "minutes": minutes, "day_kind": s["day_kind"], "assumed": s["assumed"],
                "roads_affected": len(s["affected_edges"]), "over_at": (s["cleared"] or "")[-5:]}

    def t_impact(self, case: str) -> dict:
        c = self._case(case)
        e = c["result"]["edges"]
        on = e[e.change_on]
        g = on.groupby("edge_id").agg(flow_usual=("flow_usual", "mean"), flow=("flow", "mean"),
                                      speed_usual=("speed_usual", "mean"), speed=("speed", "mean"),
                                      queue=("queue_veh", "max"), affected=("affected", "any"),
                                      delay=("extra_delay_veh_h", "sum")).reset_index()
        g["change"] = (g.flow - g.flow_usual) * 12
        g = g[g.affected]
        g = g.loc[g.change.abs().sort_values(ascending=False).index].head(TOP_ROADS)
        roads = [{"edge": r.edge_id, "road": self.places.road(r.edge_id), "flow_usual_vph": round(r.flow_usual * 12),
                  "flow_vph": round(r.flow * 12), "change_vph": round(r.change),
                  "speed_usual_kmh": None if pd.isna(r.speed_usual) else round(r.speed_usual),
                  "speed_kmh": None if pd.isna(r.speed) else round(r.speed),
                  "queue_veh": round(r.queue), "extra_delay_veh_h": None if pd.isna(r.delay) else round(r.delay, 1)}
                 for r in g.itertuples()]
        j = c["result"]["junctions"]
        js = j.groupby("junction").agg(delay=("extra_delay_veh_h", "sum"), affected=("affected", "any")).reset_index()
        juncs = [{"junction": r.junction, "name": self.places.node(r.junction),
                  "extra_delay_veh_h": round(r.delay, 1)} for r in js[js.affected].itertuples()]
        s = c["result"]["summary"]
        total = float(js.delay[js.affected].sum()) if len(js) else 0.0
        road, lv = change_label(self.places, c["edge"], c["level"], c.get("also"))
        return {"case": case, "road": road, "level": lv, "roads": roads,
                "junctions": juncs, "total_extra_delay_veh_h": round(total, 1), "max_queue_veh": round(s["max_queue_veh"]),
                "over_at": (s["cleared"] or "")[-5:],
                "note": "flows veh/h, mean while the change lasts; affected = outside the usual 10-90% range; "
                        "a junction's extra delay already includes its roads in, and total_extra_delay_veh_h is the sum "
                        "over the affected junctions: compare cases by it, do not add roads and junctions"}

    def t_detour(self, edge: str) -> dict:
        e = self._edge(edge)
        kinds = dict(zip(self.nodes.node_id, self.nodes.kind))
        path = rules.detour(self.edges, kinds, e)
        if not path:
            return {"edge": e, "road": self.places.road(e), "detour": None, "note": "no way round on the town's roads"}
        ed = self.edges.set_index("edge_id")
        return {"edge": e, "road": self.places.road(e), "detour": [{"edge": p, "road": self.places.road(p)} for p in path],
                "length_m": round(float(ed.loc[path, "length_m"].sum())), "direct_m": round(float(ed.at[e, "length_m"]))}

    def t_actions(self, case: str) -> dict:
        acts = self.case_actions(self._case(case))
        return {"case": case, "actions": [{k: a[k] for k in ("priority", "when", "action", "why")} | {
            "where": self.places.node(a["junction"])} for a in acts]}

    def case_actions(self, c: dict) -> list[dict]:
        """actions.for_change on a case (set_road's record): the rule cards, junction ids kept (the Predict form)."""
        r = c["result"]
        t0 = datetime.fromisoformat(f"{c['date']}T{c['time']}")
        t1 = t0 + timedelta(minutes=c["minutes"])
        j = r["junctions"]
        jt = (j.groupby("junction").agg(extra_delay_veh_h=("extra_delay_veh_h", "sum"), affected=("affected", "any"),
                                        from_ts=("ts", "min"), to_ts=("ts", "max")).reset_index())
        jt["affected"] = jt.affected.astype(int)
        jt["from_ts"], jt["to_ts"] = jt.from_ts.str[-5:], jt.to_ts.str[-5:]
        e = r["edges"]
        on = e[e.change_on]
        worst = on.groupby("slot").extra_delay_veh_h.sum().idxmax() if len(on) else None
        state = on[on.slot == worst][["edge_id", "speed", "flow"]].rename(columns={"flow": "total"})
        ch = {c["edge"]: c["level"], **(c.get("also") or {})}
        blocked = {k for k, lv in ch.items() if lv in ("closed", "jammed")}     # diversions keep off all of them
        acts = []
        for i, (k, lv) in enumerate(ch.items()):            # the other roads: their own actions (diversions, crews)
            change = {"edge": k, "level": lv, "from": t0.strftime("%H:%M"), "to": t1.strftime("%H:%M"),
                      "avoid": blocked - {k}}
            acts += rules.for_change(self.places, self.edges, self.nodes, change, jt if i == 0 else jt.iloc[0:0], state)
        seen = set()
        return [a for a in acts if not (a["action"] in seen or seen.add(a["action"]))]

    def t_history(self, target: str) -> dict:
        t = str(target)
        if t in set(self.edges.edge_id):
            where, name = "edge = ?", self.places.road(t)
            params: tuple = (t,)
        elif t in self.places.junction:
            ins = self.edges.edge_id[self.edges.to_node == t].tolist()
            where, name = f"edge IN ({','.join('?' * len(ins))})", self.places.node(t)
            params = tuple(ins)
        else:
            raise CheckError(f"{t!r} is neither a road nor a junction id")
        with sqlite3.connect(self.labels_db) as c:
            inc = pd.read_sql(f"SELECT edge, kind, start, end FROM incident_log WHERE {where} AND label_source = ? "
                              "ORDER BY start", c, params=params + (self.model.label_source,))
        rows = [{"road": self.places.road(r.edge), "kind": r.kind, "start": r.start[:16], "minutes":
                 round((pd.Timestamp(r.end) - pd.Timestamp(r.start)).total_seconds() / 60)} for r in inc.itertuples()]
        return {"target": t, "name": name, "incidents": rows, "count": len(rows),
                "days_on_record": f"{self.model.days[0]} .. {self.model.days[-1]}"}


MAX_ALSO = 2          # up to three roads changed together


def change_label(pl: Places, edge: str, level: str, also: dict | None = None) -> tuple[str, str]:
    """The road(s) and level of a change in words: ('A → B', 'closed'); two roads at one level ('A → B and C → D',
    'closed'); mixed levels ('A → B closed and C → D jammed', 'changed')."""
    ch = {edge: level, **(also or {})}
    names = [pl.road(k) for k in ch]
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]  # noqa: E731
    if len(set(ch.values())) == 1:
        return join(names), level
    return join([f"{n} {lv}" for n, lv in zip(names, ch.values())]), "changed"


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower().replace("→", " to "))
    s = re.sub(r"\b(the|road|from|street|junction)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------- Glimmer
def system_prompt(town_name: str, today: date, places: Places, ws: list[dict] | None = None) -> str:
    js = ", ".join(f"{n} ({j})" for j, n in places.junction.items())
    filmed = (f"Predictions are possible only at times inside the filmed windows ({covered(ws)}); for another time "
              "say so instead of moving it. ") if ws else ""
    return (f"You answer questions about traffic in {town_name}. Today is {today:%A} {today.isoformat()}. {filmed}"
            f"Junctions: {js}. Every road runs one way between two junctions, or between a junction and its own "
            "neighbourhood ('<junction> neighbourhood'). Use the tools: find a road's edge id with find_road before "
            "using it; turn relative dates (tomorrow, Monday) into YYYY-MM-DD from today; times as HH:MM, 24 h. "
            "A road change lasts 30 to 120 minutes and its level is busy, slow, jammed or closed; if the question "
            "asks for something else, say so instead of changing it. Never make up numbers: every number you write "
            "must come from a tool result. Answer briefly, in plain words.")


def _numbers(text: str) -> set[str]:
    text = re.sub(r"(?<=\d)[    ,](?=\d{3}(?!\d))", "", text)     # 1 322 / 1,322 -> 1322
    return {n.rstrip(".").lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", text)}


def numbers_ok(answer: str, question: str, results: list[dict], given: str = "") -> tuple[bool, list[str]]:
    """Every number in the answer appears in a tool result, the question or `given` (the system prompt and the tool
    descriptions: our own text); dates and times count by their parts."""
    seen = _numbers(question) | _numbers(given)
    for r in results:
        blob = json.dumps(r, ensure_ascii=False)
        seen |= _numbers(blob)
        for x in re.findall(r"-?\d+\.\d+", blob):           # an answer may round 23.8 to 24
            seen.add(str(round(float(x))).lstrip("-").lstrip("0") or "0")
    bad = sorted(n for n in _numbers(answer) if n not in seen)
    return not bad, bad


@dataclass
class Answer:
    question: str
    text: str
    calls: list[dict]           # {"name", "args", "result"}
    numbers_ok: bool
    unknown_numbers: list[str]
    seconds: float
    rounds: int
    plan: dict | None = None    # the fast path: Glimmer's reading of the question
    facts: list[str] = field(default_factory=list)
    plan_seconds: float = 0.0
    insights: list[str] = field(default_factory=list)   # the fast path: Glimmer's quick insights, one line each
    actions: list[dict] = field(default_factory=list)   # and the rules' action items, by code
    verdicts: list[dict] = field(default_factory=list)  # recommended / not recommended per road change, by code


class Assistant:
    def __init__(self, town: str, sim: dict, out_dir: Path, today: date | None = None, server: str = SERVER,
                 tools: Tools | None = None, backend: str | None = None):
        from . import assist_backend

        self.tools = tools or Tools.load(town, sim, out_dir)
        self.today = today or date.today()
        self.server = server
        self.town_name = city_name(town)
        self.backend = backend or assist_backend.current()     # glimmer (local) | muse_spark (Meta Model API)

    def _chat_completions(self, body: dict, timeout: int = 600) -> dict:
        """POST an OpenAI-style chat completion: to the local llama-server, or to the Meta Model API with the key."""
        from . import assist_backend as ab

        if self.backend == "muse_spark":
            key = ab.api_key()
            if not key:
                raise OSError(f"no Meta Model API key: set {ab.KEY_ENV}")
            url, head = f"{ab.API_URL}/chat/completions", {"Authorization": f"Bearer {key}"}
            body = {"model": ab.API_MODEL, **body}
        else:
            url, head = f"{self.server}/v1/chat/completions", {}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **head})
        return json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]

    def _chat(self, messages: list[dict]) -> dict:
        return self._chat_completions({"messages": messages, "temperature": 0, "tools": [
            {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": {"type": "object", "properties": t["parameters"],
                                                             "required": t["required"]}}} for t in TOOLS]})

    def ask(self, question: str, on_call=None) -> Answer:
        """on_call(call) after each tool call ({"name", "args", "result"}): the view shows results as they come."""
        self.tools.cases.clear()
        t0 = time.perf_counter()
        messages = [{"role": "system", "content": system_prompt(self.town_name, self.today, self.tools.places,
                                                                self.tools.windows)},
                    {"role": "user", "content": question}]
        calls: list[dict] = []
        text, rounds = "", 0
        for rounds in range(1, MAX_ROUNDS + 1):
            msg = self._chat(messages)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                text = msg.get("content") or ""
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args, result = {}, {"error": "arguments are not JSON"}
                else:
                    result = self.tools.call(name, args)
                calls.append({"name": name, "args": args, "result": result})
                if on_call:
                    on_call(calls[-1])
                messages.append({"role": "tool", "tool_call_id": tc.get("id", name),
                                 "content": json.dumps(result, ensure_ascii=False)})
        given = messages[0]["content"] + json.dumps(TOOLS, ensure_ascii=False)
        # the call arguments count too: dates, times and minutes Glimmer took from the question (checked by the tools)
        ok, bad = numbers_ok(text, question, [c["result"] for c in calls] + [c["args"] for c in calls], given)
        return Answer(question, text, calls, ok, bad, time.perf_counter() - t0, rounds)

    def _complete(self, messages: list[dict], schema: dict | None = None, n_predict: int = 400) -> str:
        """One reply straight to the user, without Glimmer's reasoning turn (its template's 'assistant to=user'
        channel): the template applied by the server, the reply started by us, held to `schema` when given. Muse
        Spark: a plain chat completion, held to `schema` by its structured output."""
        if self.backend == "muse_spark":
            body = {"messages": messages, "temperature": 0, "max_tokens": n_predict}
            if schema:
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
            else:
                body["stop"] = ["\n\n"]
            out = (self._chat_completions(body, timeout=120).get("content") or "").strip()
            if out.startswith("```"):                  # a fenced JSON block
                out = out.strip("`").removeprefix("json").strip()
            return out

        def post(path, body):
            req = urllib.request.Request(f"{self.server}{path}", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=300))

        prompt = post("/apply-template", {"messages": messages})["prompt"] + " to=user<|message|>"
        body = {"prompt": prompt, "temperature": 0, "n_predict": n_predict, "cache_prompt": True,
                **({} if schema else {"stop": ["\n\n"]})}             # the insight: one paragraph
        if schema:
            body["json_schema"] = schema
        return post("/completion", body)["content"].strip()

    def predict(self, question: str, when: tuple[date, str] | None = None) -> Answer:
        """The Predict page's path: Glimmer reads the question into a plan, code predicts, Glimmer words the insight
        from the fact sheet. `when`: the page's date and time, for a question that gives none."""
        t0 = time.perf_counter()
        sys_p = plan_prompt(self.town_name, self.today, self.tools, when)
        raw = self._complete([{"role": "system", "content": sys_p}, {"role": "user", "content": question}],
                             plan_schema(self.tools))
        plan = json.loads(raw)
        t_plan = time.perf_counter() - t0
        calls = run_plan(self.tools, plan)
        vs = verdicts(calls)
        fs = facts(calls, plan) + [f"Verdict for {v['road']} {v['level']} {v['minutes']} min: "
                                   + VERDICT_WORDS[v["verdict"]]
                                   + ((", the better option" if v["verdict"] != "not" else
                                       ", the less harmful of the options") if v["best"] else "")
                                   + f" ({v['why']})." for v in vs]
        if not fs:
            text = ("This is not a question I can predict: ask about a road change, the usual traffic, the way round "
                    "a road, past incidents or a kind of day.")
            return Answer(question, text, calls, True, [], time.perf_counter() - t0, 1, plan, fs, t_plan, [text])
        raw = self._complete([{"role": "system", "content": INSIGHT_PROMPT},
                              {"role": "user", "content": f"Question: {question}\n\nFacts:\n"
                               + "\n".join(f for f in fs if not f.startswith("Actions:"))}],
                             INSIGHT_SCHEMA, n_predict=INSIGHT_TOKENS)
        ins = [x.strip() for x in json.loads(raw)["insights"] if x.strip()][:INSIGHTS]
        text = "\n".join(f"- {x}" for x in ins)
        ok, bad = numbers_ok(text, question, [fs, plan])
        return Answer(question, text, calls, ok, bad, time.perf_counter() - t0, 2, plan, fs, t_plan, ins,
                      action_items(calls), vs)


# ---------------------------------------------------------------------- the fast path (the Predict page)
# Glimmer reads the question into a plan (one call, JSON held to the plan's schema, no reasoning turn); code runs
# the plan on the tools; Glimmer words the insight from a short fact sheet (one call). About 10 s, not a minute.
INSIGHT_TOKENS = 220
TOP_FACTS = 3


def plan_schema(tools: Tools) -> dict:
    edges = list(tools.edges.edge_id) + ["none"]
    when = {"date": {"type": "string"}, "time": {"type": "string"}}
    arr = lambda items: {"type": "array", "items": items}  # noqa: E731
    keys = ["unknown", "day", "history", "detour", "usual", "changes"]      # written in this order: the plain asks
    return {"type": "object", "properties": {                               # before the road changes
        "unknown": {"type": "string"},
        "day": arr({"type": "string"}),
        "history": arr({"type": "string", "enum": edges + list(tools.places.junction)}),
        "detour": arr({"type": "string", "enum": edges}),
        "usual": arr({"type": "object", "properties": {**when, "edge": {"type": "string", "enum": edges}},
                      "required": ["date", "time", "edge"]}),
        "changes": arr({"type": "object", "properties": {
            "edge": {"type": "string", "enum": edges}, "level": {"type": "string", "enum": list(LEVELS) + ["other"]},
            "with": arr({"type": "object", "properties": {"edge": {"type": "string", "enum": edges},
                                                          "level": {"type": "string", "enum": list(LEVELS)}},
                         "required": ["edge", "level"]}),
            "minutes": {"type": "integer"}, **when}, "required": ["edge", "level", "with", "minutes", "date", "time"]})},
        "required": keys}


def plan_prompt(town_name: str, today: date, tools: Tools, when: tuple[date, str] | None = None) -> str:
    """Everything Glimmer needs to read the question, so it looks nothing up: the days ahead with their weekdays,
    the roads and junctions with their ids, the page's date and time for a question that gives none."""
    pl = tools.places
    cal = ", ".join(f"{d:%a} {d.isoformat()}" for d in (today + timedelta(days=k) for k in range(15)))
    roads = "\n".join(f"{e}: {pl.road(e)}" for e in tools.edges.edge_id)
    juncs = ", ".join(f"{j}: {n}" for j, n in pl.junction.items())
    default = (f"A question with no date means {when[0]:%A} {when[0].isoformat()}, with no time {when[1]}. "
               if when else "A question with no date means today. ")
    return (f"You read traffic questions about {town_name} into a JSON plan; code predicts and answers. "
            f"Today is {today:%A} {today.isoformat()}. Days: {cal}; a weekday name is the next one in this list. "
            f"{default}Times HH:MM, 24 h (8am, at 8: 08:00; 6pm: 18:00; noon: 12:00).\n"
            "changes: every road change to predict (what if, closed, jammed, ...), each with its level (busy, slow, "
            "jammed or closed; other for anything else) and its own minutes as asked (an hour: 60, 2 hours: 120; 'jammed for 60 minutes or closed for 30': 60 "
            "and 30). "
            "Heavy, heavier or more traffic on a road is level busy. Roads changed at the same time ('A and B "
            "closed') are one change: the first road, the others in with (each with its level); roads compared "
            "('A or B', 'which is worse') are separate changes with with empty.\n"
            "usual: the usual or expected traffic asked about, no road change (edge none: the whole town).\n"
            "detour: roads whose way round is asked ('the way round if A to B is closed?': detour [the A → B id], "
            "no changes). history: roads or junctions whose past incidents are asked ('incidents around B?': "
            "history [B's junction id]). day: dates whose kind of day is asked.\n"
            "Leave a list empty when the question does not ask for it.\n"
            "unknown: a road or place the question names that is not in the lists below (else empty); use edge "
            "none only then.\n"
            f"Roads (id: from → to):\n{roads}\nJunctions: {juncs}")


def _when(d: str, t: str) -> str:
    try:
        return f"{date.fromisoformat(d):%a %d %b} {t}"
    except ValueError:
        return f"{d} {t}"


def run_plan(tools: Tools, plan: dict) -> list[dict]:
    """The plan's tool calls, made by code ({"name", "args", "result"}), for the fact sheet and the checks."""
    tools.cases.clear()
    calls: list[dict] = []
    u = plan.get("unknown")
    if u and (tools.t_find_road(u)["candidates"] or tools.t_find_junction(u)["candidates"]):
        plan["unknown"] = ""                       # Glimmer's "unknown" names a road or junction the town has

    def do(name, **args):
        calls.append({"name": name, "args": args, "result": tools.call(name, args)})
        return calls[-1]["result"]

    together = {(w.get("edge"), c.get("date"), c.get("time")) for c in plan.get("changes", [])
                for w in c.get("with", [])}                # roads already part of a change of two or three
    for c in plan.get("changes", []):
        if c.get("edge", "none") == "none" or c.get("level") == "other":
            continue
        if (c["edge"], c.get("date"), c.get("time")) in together and not c.get("with"):
            continue                                       # the same road again, on its own: Glimmer's repeat
        also = [w for w in c.get("with", []) if w.get("edge", "none") not in ("none", c["edge"])]
        r = do("set_road", **{k: c.get(k) for k in ("edge", "level", "minutes", "date", "time")},
               **({"also": also} if also else {}))
        if "case" in r:
            do("impact", case=r["case"])
            do("actions", case=r["case"])
    for u in plan.get("usual", []):
        do("expected_state", date=u.get("date"), time=u.get("time"),
           **({"edge": u["edge"]} if u.get("edge", "none") != "none" else {}))
    for e in plan.get("detour", []):
        if e != "none":
            do("detour", edge=e)
    for t in plan.get("history", []):
        if t != "none":
            do("history", target=t)
    for d in dict.fromkeys(plan.get("day", []) + [c["args"]["date"] for c in calls if c["name"] == "set_road"
                                                  and "case" in c["result"]]):
        do("day_kind", date=d)
    return calls


def facts(calls: list[dict], plan: dict) -> list[str]:
    """The tool results in a few plain lines: what the insight is written from (and shown when its numbers fail)."""
    out = []
    if plan.get("unknown"):
        out.append(f"Not a road or place in the town: {plan['unknown']}.")
    if any(c.get("level") == "other" for c in plan.get("changes", [])):
        out.append("A road can only be set to busy, slow, jammed or closed.")
    for c in calls:
        r, n = c["result"], c["name"]
        if "error" in r:
            out.append(f"Refused ({n}): {r['error'].split('. Say so')[0].split(': use find_road')[0]}.")
        elif n == "impact":
            worst = "; ".join(f"{x['road']} {x['flow_usual_vph']}→{x['flow_vph']} vehicles/h"
                              + (f", {x['speed_usual_kmh']}→{x['speed_kmh']} km/h"
                                 if x["speed_kmh"] is not None and x["speed_usual_kmh"] is not None else "")
                              for x in r["roads"][:TOP_FACTS])
            sr = next(x["result"] for x in calls if x["name"] == "set_road" and x["result"].get("case") == r["case"])
            out.append(f"{r['road']} {r['level']} for {sr['minutes']} min from {_when(sr['date'], sr['from'])}: "
                       f"{sr['roads_affected']} roads affected, drivers lose {time_lost(r['total_extra_delay_veh_h'])} "
                       f"in total ({impact_word(r['total_extra_delay_veh_h']).lower()} impact), "
                       + (f"longest queue {r['max_queue_veh']} vehicles, " if r["max_queue_veh"] else "")
                       + f"usual again at {r['over_at'] or 'the end'}"
                       + (f". Most changed: {worst}" if worst else "") + ".")
        elif n == "actions" and r["actions"]:
            out.append("Actions: " + "; ".join(f"{a['action']} ({a['where']}, {a['priority']})"
                                               for a in r["actions"][:TOP_FACTS]) + ".")
        elif n == "expected_state":
            rows = sorted(r["roads"], key=lambda x: -x["flow_vph"])
            one = len(rows) == 1
            out.append(f"Usual traffic {_when(r['date'], r['time'])} ({r['day_kind']}"
                       + (", assumed" if r["assumed"] else "") + "): "
                       + "; ".join(f"{x['road']} {x['flow_vph']} vehicles/h ({x['flow_range_vph'][0]}–"
                                   f"{x['flow_range_vph'][1]})" + (f", {x['speed_kmh']} km/h" if x["speed_kmh"] else "")
                                   for x in rows[:1 if one else TOP_FACTS])
                       + ("" if one else f" (the busiest {TOP_FACTS} of {len(rows)} roads)") + ".")
        elif n == "detour":
            legs = [x["road"].split(" → ") for x in r["detour"] or []]
            out.append(f"Way round {r['road']}: " + ("none on the town's roads." if not legs else
                       "via " + " → ".join([legs[0][0]] + [x[-1] for x in legs])
                       + f", {r['length_m']} m instead of {r['direct_m']} m ({r['length_m'] - r['direct_m']} m "
                       "longer)."))
        elif n == "history":
            out.append(f"{r['name']}: {r['count']} past incidents ({r['days_on_record']})"
                       + (": " + "; ".join(f"{x['road']} {x['kind']} {x['start']} {x['minutes']} min"
                                           for x in r["incidents"][-TOP_FACTS:]) if r["incidents"] else "") + ".")
        elif n == "day_kind":
            kind, _, assumed = r["model_uses"].partition(" (assumed")
            out.append(f"{r['weekday']} {r['date']}: a {r['kind']} day" + (f" ({r['event']})" if r["event"] else "")
                       + f"; its traffic is predicted from recorded {kind} days"
                       + (f", as no {r['kind']} day is recorded" if assumed else "") + ".")
    return out


INSIGHTS = 3
INSIGHT_SCHEMA = {"type": "object", "properties": {"insights": {"type": "array", "items": {"type": "string"},
                                                                "minItems": 1, "maxItems": INSIGHTS}},
                  "required": ["insights"]}
INSIGHT_PROMPT = ("You tell a traffic operator what a prediction means. From the facts only, give one to three quick "
                  "insights, each one short line of at most 20 words: the answer to the question first with its main "
                  "number, then what stands out (the worst road, how long it lasts, the kind of day); keep to the "
                  "verdict when there is one. Every number "
                  "you write must be in the facts. If something was refused, say why in plain words. No actions: "
                  "they are listed separately. In your own words: never copy a fact line, never say 'the facts'.")


VERDICT_DELAY_VEH_H = IMPACT[-1][0]   # extra delay over this (5 veh-h), or a queue this long: not recommended;
VERDICT_QUEUE_VEH = 20                # from half of either: with caution; below both halves: recommended
VERDICT_WORDS = {"recommended": "recommended", "caution": "with caution", "not": "not recommended"}


def verdicts(calls: list[dict]) -> list[dict]:
    """Recommended / with caution / not recommended for each road change, by code from its impact: extra delay and
    longest queue against the limits (VERDICT_*); with two or more, the one with the least extra delay is best."""
    out = []
    for c in calls:
        if c["name"] != "impact" or "error" in c["result"]:
            continue
        r = c["result"]
        sr = next(x["result"] for x in calls if x["name"] == "set_road" and x["result"].get("case") == r["case"])
        delay, queue = r["total_extra_delay_veh_h"], r["max_queue_veh"]
        if delay > VERDICT_DELAY_VEH_H or queue >= VERDICT_QUEUE_VEH:
            verdict = "not"
        elif delay > VERDICT_DELAY_VEH_H / 2 or queue >= VERDICT_QUEUE_VEH / 2:
            verdict = "caution"
        else:
            verdict = "recommended"
        why = (f"drivers lose {time_lost(delay)} in total (limit {VERDICT_DELAY_VEH_H:g} h)"
               + (f", queue of {queue} vehicles (limit {VERDICT_QUEUE_VEH})" if queue else ""))
        out.append({"case": r["case"], "road": r["road"], "level": r["level"], "minutes": sr["minutes"],
                    "when": _when(sr["date"], sr["from"]), "delay": delay, "verdict": verdict, "why": why,
                    "best": False})
    if len(out) > 1:
        min(out, key=lambda v: v["delay"])["best"] = True      # the least harm of the options compared
    return out


def action_items(calls: list[dict]) -> list[dict]:
    """The rules' actions of each road change (actions.for_change), highest priority first, with the case's road."""
    road = {c["result"]["case"]: c["result"]["road"] for c in calls if c["name"] == "set_road" and "case" in c["result"]}
    rank = {"high": 0, "medium": 1, "low": 2}
    items = [a | {"road": road.get(c["args"]["case"], "")} for c in calls
             if c["name"] == "actions" and "actions" in c["result"] for a in c["result"]["actions"]
             if not a["action"].startswith("No action")]                 # a rule card that says nothing to do
    return sorted(items, key=lambda a: rank.get(a["priority"], 3))
