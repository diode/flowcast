# FlowCast — Architecture

How FlowCast is built end to end:
- the **Phase 1 proof of concept** (the current application), on a development machine
- the **cloud production architecture**
- the later phases: **signal control**, **traffic control** and **infrastructure planning**
- how **simulation** builds and tests the system without real traffic data
- how **Meta's models** run: through the Meta Model API and cloud GPU pools in production, and locally at low cost
  during development

**Status:** section 3 describes what exists today. Sections 4, 5 and the production column of section 7 are designs: none
of it is built or benchmarked. The proof of concept has been exercised on sample data, not on a real city's recordings.

The problem statement and the case for automation are in *Global Increasing Traffic — FlowCast — Build for Your
World.md*. Commands and operation are in the [README](../README.md).

---

## 1. Principles

1. **Camera-first.** The only input is CCTV video that the city's systems already record, with no loops, probes or new
   roadside hardware. Measurement covers only the road directions a calibrated camera watches.
2. **One network graph.** Every road direction is an edge between junctions. Observation, state, forecast and
   control all work on the same graph and the same ids.
3. **Numbers from models, words from language models.**
   - Counts and speeds come from vision (SAM 3.1); forecasts come from the network model.
   - The language model (Muse Spark in production; Muse Glimmer in development) only turns a question into a
     structured plan and the result
     into words. Every number it writes is checked against the facts and the question; if the check fails, the app shows
     the facts instead. The check confirms a number appears there, not that the wording around it is right.
4. **Simulation makes sample data only.** SUMO and CARLA build sample towns for development and scoring. A real
   town's pipeline has no simulator in it, and forecasts never run a simulator.
5. **People approve actions.** Until a later phase is proven, FlowCast recommends and people decide. Automated
   control stays inside explicit safety limits.
6. **Measure forecasts.** Forecasts are compared with what then happens. Today this is scored on the sample towns, and
   the app shows an accuracy badge. Continuous scoring on a real city is a production goal.

---

## 2. Functional view

```
 ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
 │  ONBOARD    │   │   INGEST     │   │   OBSERVE       │   │   NETWORK STATE  │   │   FORECAST            │
 │ map (OSM)   │   │ recordings   │   │ SAM 3.1 on      │   │ road directions  │   │ graph model:          │
 │ cameras     │──►│ pushed by    │──►│ frame pairs →   │──►│ with cameras,KPIs│──►│ usual day, road       │
 │ calibration │   │ city systems │   │ per-camera      │   │ usual pattern    │   │ changes, junction     │
 └─────────────┘   └──────────────┘   │ 5-min measures  │   └──────────────────┘   │ delay, actions        │
                                      └─────────────────┘                          └──────────┬───────────┘
                                                                                              │
              ┌──────────────────────────┐            ┌───────────────────────────────────────┴──────────┐
              │ TRAFFIC ASSISTANT         │ plan/words │ DASHBOARD                                         │
              │ Muse Spark (Meta API);    │◄──────────►│ City overview: Observed · Forecast                │
              │ dev: Muse Glimmer (local) │            │ Technical view: network, onboarding & ingest      │
              └──────────────────────────┘            └──────────────────────────────────────────────────┘
   Later:  SIGNAL CONTROL (2) ── TRAFFIC CONTROL (3) ── INFRASTRUCTURE PLANNING (4), all on the same graph
```

---

## 3. Phase 1 — proof of concept on a development set-up (the current application)

### 3.1 Deployment

One workspace machine (the development set-up): any developer's computer with one or more NVIDIA GPUs. Every component is a Python module or process on it. Which GPU runs what is the machine's own setting (`workspace.yaml`), not part of the design.

```
 City / test systems                       FlowCast workspace machine
 ───────────────────                       ──────────────────────────────────────────────────────────────────
  recorders, VMS export ──file copy──►  inbox/ (folder or share)
  or HTTP PUT  ───────────────────────► netstudy receive (HTTP, bearer token) ──► inbox/
                                              │ netstudy observe  (once, or --follow)
                                              ▼
                                        pickup: validate name, camera, video ──► recordings/<date>/<period>/
                                              ▼
                                        ingest ── SAM 3.1 ──────────────────────────────── vision GPU
                                              ▼
                                        traffic.db (SQLite): observations → edge_state / edge_view → network_kpi
                                              ▼
                                        graph model (fit, pickle) ── forecast tools
                                              ▼
                                        Streamlit dashboard ◄─► Traffic Assistant
                                                                 ├─ Muse Spark via Meta Model API (HTTPS): production model
                                                                 └─ Muse Glimmer via llama-server ── assistant GPU: dev, low cost
```

| Workspace setting | Role |
|---|---|
| `vision_gpu` | SAM 3.1 ingest (about 6 GB in use, measured on a 12 GB card) |
| `glimmer_gpu` | the development Traffic Assistant, Muse Glimmer (about 14 GB in use, measured); on one card, stop it while ingest runs, since a 16 GB card cannot hold both; none when developing against Muse Spark. The single-card set-up is untested. |
| `one_gpu_at_a_time` | for machines whose power supply cannot carry two GPUs under full load: ingest waits while another GPU is busy, and Glimmer does not start while one is |

Defaults: one GPU (0) for everything, with no load limit. None of this exists in production, where SAM 3.1 runs on
GPU pools and the language model is Muse Spark (section 4).

### 3.2 Components

| Component | Module(s) | Responsibility |
|---|---|---|
| Map import | `osm.py` | OpenStreetMap extract → junctions (merged within 25 m), boundary points, one edge per driving direction, lanes, speed limits, shapes, street-based names; no SUMO |
| Camera onboarding | `realtown.py` | Cameras snapped to their edge by position and heading. Ground-plane homography from at least 4 pixel-to-map point pairs. Measured zone from the edge shape, lanes and stop line |
| Recording intake | `inbox.py` | Inbox contract (file names, sidecar, completion rule), validation, filing, refusals with reasons; HTTP receiver |
| Vision | `vision.py`, `ingest.py` | SAM 3.1 text-prompted detection on frame pairs every 12 s; ground placement, zone filter, lane and distance, matched speeds; 5-minute observations |
| Network state | `graph.py`, `kpis.py` | `edge_state` / `edge_view` per road direction; network KPIs (vehicles in town, share congested, speed index, delay) |
| Forecast engine | `graph_model.py`, `predict_data.py`, `actions.py`, `days.py` | Usual flow and speed per edge, slot and day kind; turn shares; capacity (signal plan or observed); road-change propagation with rerouting; junction delay; rule-based actions |
| Traffic Assistant | `assist.py`, `assist_backend.py`, `glimmer_server.py` | Question → plan (JSON schema) → tool calls by code → facts → insight; numbers check; backend switch (Muse Spark; Glimmer for development) |
| Orchestration | `observe.py`, `jobs.py`, `cli.py` | Pickup → ingest → state → model loop; onboarding checklist; background jobs for the dashboard |
| Dashboard | `dashboard/` | City overview (Observed, Forecast), Technical view (network, timeline, edge, onboarding & ingest), model menu |

### 3.3 Data model (per town, `traffic.db`)

```
 nodes(node_id, kind, x, y, signal, lat, lon)        edges(edge_id, from_node, to_node, kind, length_m, lanes,
                                                            speed_kmh, capacity_pcu, camera_id, street)
 received(file, camera_id, start, seconds, status, reason)      ingested(path, camera_id, start, samples)
 vision_samples(camera_id, t, vehicles[], speeds[])
 observations(camera_id, ts, total, counts, pcu, occupancy, load_ratio, level, flow, speed, samples, source)
 edge_state / edge_view(edge_id, ts, …, observed, method)       network_kpi(ts, veh_in_town, share_congested,
                                                                            speed_index, delay_veh_h, …)
 predict/graph_model.pkl     study_closures, predict_eval_*  (studies and accuracy)
```

### 3.4 Forecast path for one question (median 6.8–11.7 s by town, with Muse Glimmer on the sample towns)

```
 question ─► LLM: plan (JSON schema: roads by enum id, level, minutes, date, time; refusals for unknown places)
          ─► code: set_road → impact → actions, expected state, detour, history, day kind   (graph model, ms)
          ─► code: verdict per road change (delay and queue limits), action items
          ─► LLM: 1–3 quick insights from the fact sheet  ─► numbers check ─► dashboard
```

### 3.5 What the proof of concept does not cover

- It has not read a real city's recordings. The real-town path was tested on a made-up town with a stand-in for SAM 3.1,
  and SAM 3.1 has only read rendered video (flow error 25–37% per 5 minutes, speed error 5–7 km/h on the sample towns).
- The Traffic Assistant uses Muse Glimmer for local testing and Muse Spark (Meta Model API) for live use.
- Recordings are read in batches as they arrive, so the state is not real time.
- Roads without a camera have no measurement. The forecast estimates them.
- It runs on a single node: SQLite, local disk, and background jobs instead of a scheduler.
- There is no tenant isolation, single sign-on or audit trail.
- Signal plans of real towns are not imported. A real town's forecast uses each approach's observed capacity.
- There is no human labelling tool, so a real town's vision is checked visually through the calibration overlay,
  not scored.

---

## 4. Production architecture (cloud)

This section is a design. Nothing in it is built or benchmarked yet.

### 4.1 Overview

```
 CITY EDGE                         CLOUD (per region; one tenant per city)
 ─────────                         ─────────────────────────────────────────────────────────────────────────────────
 VMS / NVR exports ──HTTPS──► Ingestion API ──► Object storage (raw video, lifecycle: e.g. 7–30 days)
 camera recorders            (auth, quota,        │ event: segment stored
 optional edge box ─observations only─┐  checks)  ▼
   (SAM 3.1 on site)                  │     Segment queue ──► GPU inference pool (SAM 3.1, autoscaled, batched)
                                      │                              │ vision samples, observations
                                      └──────────────────────────────▼
                                            Observation store (time series: PostgreSQL + TimescaleDB or similar)
                                                   │ stream / micro-batch
                                                   ▼
                                            State & KPI service ──► network state per 5 min, KPIs, alerts
                                                   │
                                  nightly fit ─────┼────► Model registry (graph model per town, versioned)
                                                   ▼
                                            Forecast service (API): expected day, road changes, actions, verdicts
                                                   ▲                         ▲
                                                   ▼                         ▼ plan / words (tool calls by code)
                                      Web app (dashboard, operator console)   LLM gateway ◄──► Muse Spark (Meta Model API,
                                                                                               outside the tenant)
                                      SSO, roles, audit
 Cross-cutting: identity (SSO / OIDC), secrets, monitoring and tracing, data retention, per-tenant encryption
```

The queue carries references to stored segments, and the GPU workers read the video from object storage. An edge box takes
the same recordings on site and sends only observations. Only questions and model results go to Muse Spark, which runs
outside the tenant.

### 4.2 Mapping from the proof of concept

| POC | Production |
|---|---|
| `inbox/` folder, `netstudy receive` | Ingestion API (HTTPS, mutual TLS or signed tokens per sender), direct upload to object storage, the same name and sidecar contract |
| `pickup` | Ingestion validation: camera registry lookup, video probe, duplicate check, quarantine bucket for refusals |
| `ingest` on the vision GPU | GPU inference workers fed from a segment queue: horizontal scaling, batched frames, retry and dead-letter |
| `traffic.db` (SQLite) | Time-series database (observations, state, KPIs) plus relational tables (towns, cameras, graph, calibration) |
| `graph_model.pkl` | Model registry: versioned fits per town, nightly refit, promotion on accuracy checks |
| `assist_backend` switch (Glimmer in development) | LLM gateway to Muse Spark: rate limits, retries, per-tenant keys, logging of plans (not of video) |
| Streamlit | Web application with SSO and roles (viewer, operator, engineer, admin) and an audit trail |
| `jobs.py` | Workflow scheduler (onboarding, backfills, studies, refits) |

### 4.3 Inference sizing

From the proof of concept:
- SAM 3.1 reads about **2.5 to 3.5 frames per second** on a 12–16 GB consumer GPU, with four prompts (3.5 measured in
  the GPU check on rendered video).
- Each camera needs **2 frames every 12 s**, which is 1/6 of a frame per second.

| | Per GPU (12–16 GB consumer class, unbatched) | Example: 4,000 cameras, 24 h |
|---|---|---|
| Cameras in real time | about 15 to 21 | about 190 to 270 GPUs (before batching or larger GPUs) |

These figures are linear estimates from the proof of concept and must be benchmarked on the target GPUs. Ways to
reduce the GPU count:
- batched inference on data-centre GPUs
- a longer sampling interval at quiet hours
- prompt-specific crops
- edge inference (below)

### 4.4 Edge option

Where bandwidth or privacy rules out sending video, SAM 3.1 runs on an edge box at the city's data centre or
roadside cabinet. Only **observations** (counts, speeds, levels: a few kilobytes per camera per 5 minutes) go to
the cloud. The cloud side is unchanged from the observation store onwards.

### 4.5 Non-functional requirements

| Area | Design |
|---|---|
| Privacy | Video kept only as long as needed to re-run ingest. Observations hold counts and speeds, with no images. A privacy review per city is required; stored video can contain people and number plates. |
| Security | Per-sender credentials, TLS everywhere, tenant-scoped encryption keys, least-privilege service roles, audit of every action and approval |
| Reliability | Ingestion is idempotent (camera plus start time). Queue retries. Gaps are explicit (no row means no measurement, never zero traffic). The forecast service degrades to the usual day when live data is late. |
| Accuracy governance | Every model version is scored against held-out days. Accuracy badges appear in the app. Promotion needs no regression. |
| Data residency | Region per tenant. Only questions and model results reach Muse Spark, never video. Where a city's policy forbids any external API, an open-weights model self-hosted in the tenant (Muse Glimmer) is the fallback. |

---

## 5. Later phases

### 5.1 Phase 2: forecast-based traffic signalling

```
 forecast (next day / next period, per approach) ──► timing optimiser ──► candidate plans per junction and period
        ▲                                             (cycle, splits, offsets;      │
        │                                              Webster / HCM delay, as the  ▼
        │                                              graph model already uses)   safety check: minimum greens,
        │                                                                          pedestrian clearance, max cycle,
  measured result ◄── controller adapter ◄── approval (engineer) ◄─────────────── intergreens, change windows
  (observations)      NTCIP 1202 / UTMC /
                      vendor central system
```

- **Step 1: recommend.** Plans per period, with forecast delay saved, are reviewed and approved by engineers and
  applied through the existing central system.
- **Step 2: automated within limits.** Plans are applied automatically within an approved range, with automatic
  fallback to the last approved plan.
- **Closed loop:** observed delay after a change is compared with the forecast, and the model and the optimiser
  learn from both.

### 5.2 Phase 3: traffic control and incident response

- **Incident detection:** sudden loss of flow or speed on an approach while its neighbours change, with a target of
  flagging within minutes. Detection on camera data is not yet scored. A camera frame check could be added with a vision-capable model (third-party guides list image input for Muse Spark)
  to read the likely cause.
- **Response plans:** diversions and their effect forecast by the same road-change engine. Message signs,
  temporary signal plans and officer deployment are proposed for operator approval.
- **Planned events and roadworks:** the closure study ranks options by forecast harm and recommends the time
  window. The app's verdicts (*Recommended / With caution / Not recommended*) extend from single questions to
  permit workflows.

### 5.3 Phase 4: traffic infrastructure recommendation

- **Evidence base:** months of per-approach observations show which junctions limit the network, which roads have
  no alternative route (critical-road study), and where demand is growing.
- **Scenario testing:** capacity changes (lanes, turn pockets, signalisation, roundabouts, bus priority) are edited
  into the graph and forecast against the observed demand before design starts.
- **Output:** ranked interventions with forecast delay saved, the evidence behind each, and uncertainty bands.

---

## 6. Building without real traffic data: simulation

Real recordings take weeks to accumulate and have no ground truth. FlowCast therefore uses **sample towns** whose
traffic is simulated, so that every part of the pipeline can be built, tested and scored first.

```
 town config (junctions, roads, demand zones, signal timing sheets, incidents)
        │
        ├──► SUMO: microscopic traffic for each day ──────► ground truth per road and vehicle (labels)
        │                                                    │
        └──► CARLA: 3-D world from the same roads, CCTV      │
             cameras at the same poses, day/night light ──► rendered video ──► the SAME ingest (SAM 3.1)
                                                                                    │
                                     score vision vs labels ◄────────────────────────┤
                                     score forecasts vs paired SUMO runs ◄─── graph model forecasts
```

| Use | How |
|---|---|
| Develop the pipeline | Rendered CCTV video goes through the same pickup, SAM 3.1 ingest, state and dashboard code as a real town |
| Score vision | SUMO's exact counts and speeds are labels for SAM 3.1's observations (flow error, speed error per camera) |
| Score forecasts | Paired SUMO runs (with and without a road change) are the answer key for the graph model's road-change forecasts |
| Build months of history quickly | Days not rendered come straight from SUMO as the same observation table, which covers day kinds, events and incidents |
| Test edge cases safely | Closures, surges, festivals and incidents that a real city would not stage |

**Limits:**
- Simulated behaviour and rendered video are not reality.
- Every real town is checked on its own data: the calibration overlay, observation plausibility, and forecast
  accuracy as days accrue.
- Simulation never produces a real town's forecast.

---

## 7. Meta models: production and development

| Model | Purpose | Production | Development (low cost) |
|---|---|---|---|
| **SAM 3.1** | Vehicle detection in CCTV frames (text prompts: car, motorcycle, bus, truck) | Cloud GPU pool on data-centre GPUs, batched and autoscaled, or edge boxes where video must stay on site. It runs as FlowCast's own service. | Weights on the developer's GPU (`models/sam3`), PyTorch |
| **Muse Spark** | Traffic Assistant: question → plan; facts → insight; possible vision checks in Phase 3 | **Meta Model API**: Meta's hosted model, documented structured output, no GPUs for FlowCast to operate; key per tenant | Selectable in the menu; the live model (Muse Glimmer for local testing) |
| **Muse Glimmer** (open weights) | Stand-in for Muse Spark | – (fallback only where policy forbids any external API) | llama.cpp `llama-server` on the developer's GPU, quantised GGUF (about 14 GB): no API cost, no network |
| **DINOv3** (envisioned) | Camera health and scene change | Cloud GPU pool, with SAM 3.1 | Developer's GPU |

**Why Muse Spark in production, and Glimmer in development:**

| | Muse Spark (Meta Model API): production | Muse Glimmer (local): development |
|---|---|---|
| Capability | Meta's hosted model, with documented structured output; to be scored on FlowCast's question set | A quantised open-weights model; matched all 74 questions in the question set |
| Scale | Hosted by Meta, within the API's limits | One developer's GPU |
| Operations | No GPUs or model updates for FlowCast to run | The developer runs llama-server |
| Cost | Usage-based | No API cost |
| Data | Questions and model results only, never video | Stays on the developer's machine |

Both sit behind one interface (`assist_backend` in the proof of concept, the LLM gateway in production). Both get the
same plan schema, the same tool calls and the same numbers check. The same plan gives the same forecast figures, because
those come from the network model. A different model can read a question into a different plan, so the question set must
be re-scored when the model changes.

---

## 8. Roadmap

| Phase | Scope | Exit criteria |
|---|---|---|
| **1. Observe and forecast** (proof of concept on sample data; real city and production next) | Onboarding from OSM; pushed recordings; SAM 3.1 observations; network state and KPIs; usual-day and road-change forecasts; Traffic Assistant; studies | Vision accuracy per camera within the agreed limits on labelled samples; forecast accuracy reported per town; operators use the Forecast view in planning |
| **2. Forecast-based traffic signalling** | Timing optimiser, safety rules, approval workflow, controller adapters | Measured delay reduction on pilot corridors with no safety regressions |
| **3. Traffic control** | Incident detection, response plans, event and roadworks scheduling | Detection time and response quality measured against the operator log |
| **4. Traffic infrastructure recommendation** | Long-run evidence, scenario editing, ranked interventions | Recommendations accepted into the city's capital planning process |
