# FlowCast

City traffic, forecast. FlowCast turns a town's traffic camera recordings into a measured picture of the roads they
cover and a forecast of what a road change would do: a closure, roadworks, an event. It shows them in a dashboard for traffic
teams.

- **Observe:** recordings pushed by the town's camera systems are read with Meta's SAM 3.1. Each camera measures
  vehicles, flow and speed on its approach every 5 minutes.
- **Understand:** the measurements form the state of the road network: every road direction, the network KPIs, and
  the usual pattern per day kind.
- **Forecast:** a graph model of the network answers "what if this road is closed / jammed / busy at that time". The
  Traffic Assistant reads questions in plain words: Muse Spark through the Meta Model API (production), or Muse Glimmer
  on a local GPU as a low-cost development option.

The code package is `netstudy`, the project's earlier name.

### Documents

| | |
|---|---|
| **Demo video** | [FlowCast demo on YouTube](https://youtu.be/dYj9K5bcDKw) |
| **Problem statement** | [Global Increasing Traffic, I. The problem](docs/Global%20Increasing%20Traffic%20-%20FlowCast%20-%20Build%20for%20Your%20World.md#i-the-problem): a global issue, its growth, its scale, and why traditional teams and signal systems cannot keep up |
| **Solution** | [II. The solution](docs/Global%20Increasing%20Traffic%20-%20FlowCast%20-%20Build%20for%20Your%20World.md#ii-the-solution-a-fully-automated-ai-driven-traffic-system): a fully automated, AI-driven system, phase by phase; [III. Meta's models](docs/Global%20Increasing%20Traffic%20-%20FlowCast%20-%20Build%20for%20Your%20World.md#iii-metas-models-in-flowcast): how SAM 3.1, Muse Glimmer and Muse Spark deliver it |
| **Architecture** | [docs/Architecture.md](docs/Architecture.md): the proof of concept (development set-up), the cloud production architecture, signalling and infrastructure phases, simulation, Meta models in production and development |
| **Operating reference** | this README: setup, onboarding a town, recordings, ingest, dashboard, commands |

---

## Contents

1. [How it works](#how-it-works)
2. [Setup](#setup)
3. [Onboard a real town](#onboard-a-real-town): map, cameras, calibration
4. [Recordings: how outside systems deliver them](#recordings-how-outside-systems-deliver-them)
5. [Observations: SAM 3.1 ingest, no SUMO](#observations-sam-31-ingest-no-sumo)
6. [Dashboard](#dashboard)
7. [Forecast and the Traffic Assistant](#forecast-and-the-traffic-assistant)
8. [Command reference](#command-reference)
9. [Data and code layout](#data-and-code-layout)
10. [Workspace machine](#workspace-machine)
11. [Tests](#tests)
12. [Simulation (SUMO + CARLA sample towns)](#simulation-sumo--carla-sample-towns)
13. [Limits and open points](#limits-and-open-points)

---

## How it works

```
 OpenStreetMap extract ──► netstudy onboard ──► road graph (junctions, road directions, shapes, names)
 town config: cameras + calibration points ──►  cameras.json (each camera's road, lanes, ground-plane homography)

 camera systems ──push──► inbox/ (file copy, share, or HTTP PUT to `netstudy receive`)
                              │  netstudy observe  (once, or --follow)
                              ▼
            pickup: check name, camera, video ──► recordings/<date>/<period>/<camera>_<HHMMSS>.mp4
                              │                     (refused files ──► inbox/rejected/ + reason)
                              ▼
            SAM 3.1 on frame pairs ──► vehicles placed on the road (homography) and kept if in the camera's zone
                              ▼
            observations (per camera, 5 min): count, PCU, occupancy, load, level, flow, speed
                              ▼
            edge_state / edge_view (every road direction) ──► network_kpi ──► graph model (usual day, road changes)
                              ▼
                         dashboard: City overview (Observed · Forecast) · Technical view (incl. Onboarding & ingest)
```

A real town has no simulator anywhere: the roads come from the map, the traffic from the cameras. SUMO and CARLA
are only used to make sample towns ([Simulation](#simulation-sumo--carla-sample-towns)).

## Setup

**Quick try (no GPU, no model files):** the three sample towns' results (no recordings, 80 MB) are on Google Drive:
**[FlowCast sample data](https://drive.google.com/drive/folders/1Kd9xJtb4LJhg0LFC9RcvwPnlV4H-4whn?usp=drive_link)**.
Download `ashford.zip`, `brookhaven.zip` and `clearwater.zip` into a `sample-data\` folder in the project root, as
they are (do not unzip). They unpack into `data/` on the first test run or dashboard start:

```powershell
git clone https://github.com/diode/flowcast.git
cd flowcast
mkdir sample-data        # put the three zips from Google Drive here
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q                         # 103 pass; 8 skip (towns not shipped)
.venv\Scripts\python -m streamlit run dashboard\app.py   # http://localhost:8501
```

The Traffic Assistant needs Muse Spark (`MODEL_API_KEY`) or a local Glimmer; without one, the rest of the dashboard
works. Reading new recordings needs the SAM 3.1 checkpoint and a GPU, as below.

**Full setup:** any workspace machine with Python 3.12 and an NVIDIA GPU for SAM 3.1 (it used about 6 GB on the 12 GB card tested). The commands below are
for Windows PowerShell; on Linux or macOS use `.venv/bin/` instead of `.venv\Scripts\`. Tell FlowCast which GPU runs
what in `workspace.yaml` ([Workspace machine](#workspace-machine)).

```powershell
git clone https://github.com/diode/flowcast.git
cd flowcast
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build
.venv\Scripts\python -m pip install opencv-python-headless git+https://github.com/facebookresearch/sam3.git
```

| Needed for | What | Where |
|---|---|---|
| reading recordings | SAM 3.1 checkpoint `sam3.1_multiplex_fp16.safetensors` (facebook/sam3.1 on Hugging Face, access request) | `models\sam3\` |
| the Traffic Assistant, production model | a Meta Model API key for Muse Spark | `MODEL_API_KEY` ([Forecast](#forecast-and-the-traffic-assistant)) |
| the Traffic Assistant, low-cost development option | `Muse-Glimmer-30B-Q3_K_M.gguf` and llama.cpp's `llama-server.exe` | `models\`, `tools\llama.cpp\` |
| sample towns only | SUMO (installed by pip), CARLA 0.9.16 (`scripts\setup_tools.ps1`) | `tools\` |
| your machine's GPUs | `copy workspace.example.yaml workspace.yaml`, then edit | project root ([Workspace machine](#workspace-machine)) |

Check: `.venv\Scripts\python -m pytest -q`.

---

## Onboard a real town

A town is one config file, `towns/<town>.yaml`. Its file name is the town's id, and `out_dir` must point at
`data/<the same id>`. Start from the template:

```powershell
copy towns\templates\real_town.yaml towns\cambridge.yaml       # then edit it
.venv\Scripts\netstudy status --sim-config towns/cambridge.yaml # the checklist: what is done, what is next
```

The same checklist is in the dashboard: **Technical view → Onboarding & ingest**. It has a button for every step.

### 1. The map

FlowCast reads an OpenStreetMap extract (`.osm` XML). Get one in either of two ways:

- From **openstreetmap.org → Export** (draw the area), JOSM, or any Overpass query. Save it where
  `network.osm.file` points.
- Or set `network.osm.bbox: [south, west, north, east]` and let FlowCast download the roads:
  `netstudy onboard --sim-config towns/<town>.yaml --fetch` (Overpass API; keep areas town-sized).

```powershell
.venv\Scripts\netstudy onboard --sim-config towns/cambridge.yaml
```

This writes the graph (tables `nodes` and `edges` in `data/<town>/traffic.db`, plus `graph.json`), the road
shapes (`network/shapes.json`), readable names (`network/names.json`) and `map.png` for review. How the map is read:

| Setting | Default | Meaning |
|---|---|---|
| `network.osm.highways` | trunk, primary, secondary, tertiary (+ links) | which roads are kept; add `unclassified`, `residential` for a finer net |
| `network.osm.merge_m` | 25 | junction nodes closer than this are one junction (dual carriageways, small roundabouts) |
| `network.osm.signal_m` | 30 | a `highway=traffic_signals` node this close makes the junction signalised |
| `network.lefthand` | false | traffic keeps left (UK, India, Japan …): lanes are on the left of the road line |
| `network.names` | – | better junction names by id, e.g. `{J4: Market Square}` |

**Junctions** are where three or more road ends meet. They are numbered J1, J2 … from the centre out. **Boundary
points** (B1 …) are where a road leaves the map, and traffic enters and leaves the town there. Every driving
direction is its own edge (`J3_J1`, kinds `road`, `in`, `out`). Lanes come from `lanes` / `lanes:forward` and
speed from `maxspeed`; without those tags, a default per road type is used. Junction names come from the streets
that meet there ("High Street & Mill Lane").

### 2. The cameras

One camera watches one road direction. List them under `cameras` in the town config:

```yaml
cameras:
  - id: cam01                                   # as the recordings' file names use it
    name: High Street eastbound at Mill Lane
    at: {lat: 52.20510, lon: 0.12115}           # a point on the road it watches, near the stop line
    heading: 90                                 # the traffic's direction, degrees from north
    # edge: J7_J1                               # or name the edge from map.png instead of at / heading
    # stop_m: 8                                 # stop line this far before the junction centre (default 8)
    # view_m: 60                                # measured stretch up from the stop line (default network.view_m)
```

`netstudy onboard` places them. After you edit them, run `netstudy cameras --sim-config …` again. FlowCast snaps a
camera to the nearest road line. Of that road's two directions, it takes the one going the camera's `heading`, or
without a heading, the one that ends at the junction nearest the camera. `map.png` shows each camera's stop line.

### 3. Calibration

Each camera needs **four or more points** that are visible in its image and can be found on the map: corners of
road markings, kerb ends, manhole covers, stop-line ends. For each one, give the pixel and the map position (lat /
lon, or town metres `x` / `y`). FlowCast fits the ground-plane homography (pixel → road) from them.

```yaml
    calibration:
      image_size: [1920, 1080]
      points:
        - {px: [412, 980], lat: 52.20509, lon: 0.12108}
        - {px: [1510, 965], lat: 52.20503, lon: 0.12109}
        - {px: [1180, 420], lat: 52.20502, lon: 0.12061}
        - {px: [760, 430], lat: 52.20511, lon: 0.12060}
```

- Spread the points over the measured zone: near and far, left and right. Three on one line give no plane.
- The fit is reported in metres. Under about 0.5 m is good; more than 1 m usually means a point is mislocated.
- **Check it:** Technical view → Onboarding & ingest → *Calibration check* draws the camera's measured zone (its
  lanes, `view_m` up from the stop line) on its latest frame. It marks each point twice: as clicked (circles) and
  where its map position lands (crosses). The zone must cover exactly the approach's lanes.
- A camera without a calibration is not read: its recordings wait in `recordings/` until it has one.

## Recordings: how outside systems deliver them

FlowCast does not connect to cameras. The town's own systems push finished recordings to FlowCast: an NVR or VMS
export, a camera's recorder, or a cloud sync.

**The inbox** is a folder: `source.inbox`, by default `data/<town>/inbox`. It can be a network share
(`\\server\flowcast\<town>`) that the recorders write to. Accepted:

| | |
|---|---|
| names | `<camera>_<YYYYMMDD>T<HHMMSS>.mp4` (`cam03_20261105T081500.mp4`), or `<camera>/<YYYY-MM-DD>/<HHMMSS>.mp4` |
| other metadata | a sidecar `<same name>.json` `{"camera_id": "cam03", "start": "2026-11-05T08:15:00"}` overrides the name |
| time | the start of the recording, the town's local clock |
| video | H.264 `.mp4` (also `.mkv`, `.avi`, `.mov`), any length (1–15 min is usual), any frame rate (read from the file), at least 5 fps |
| complete when | unchanged for `source.inbox_quiet_s` seconds (default 10), or, with `source.inbox_marker: true`, when `<file>.done` appears beside it |

**Or push over HTTP:** `netstudy receive` runs a small receiver (port 8765). It needs the shared secret in
`FLOWCAST_INGEST_TOKEN`:

```powershell
$env:FLOWCAST_INGEST_TOKEN = "a-long-random-secret"
.venv\Scripts\netstudy receive --sim-config towns/cambridge.yaml --port 8765
# a sender:
curl -X PUT --data-binary @cam03.mp4 -H "Authorization: Bearer a-long-random-secret" `
     http://flowcast-host:8765/recordings/cam03/20261105T081500.mp4
```

It answers `201` with the stored name, `401` without the right token, `400` for a bad path. `GET /health` answers
`ok`. Files are written as `.partial` and renamed when complete. Put it behind HTTPS (a reverse proxy) if it is
reachable beyond the local network.

**Pickup**, the first step of `netstudy observe`, checks each closed file:
- the name or sidecar can be read
- the camera is in `cameras.json`
- the video opens and has frames
- the same camera and start time are not already stored

It then moves the file to `recordings/<date>/<period>/<camera>_<HHMMSS>.<ext>`. The period is the `recording.windows`
window it starts in, else `day`. A refused file goes to `inbox/rejected/` with a `.reason.txt`. Fix it and copy it
back. Every file is logged in the table `received`.

## Observations: SAM 3.1 ingest, no SUMO

```powershell
.venv\Scripts\netstudy observe --sim-config towns/cambridge.yaml            # everything new, once
.venv\Scripts\netstudy observe --sim-config towns/cambridge.yaml --follow   # keep going as recordings arrive
```

Each run does four things, in order:

1. **Pickup:** the inbox files go to `recordings/`, as above.
2. **Ingest:** every recorded segment is read once (table `ingested`).
   - Every `ingest.pair_every_s` seconds (12), SAM 3.1 looks at a pair of frames 0.6 s apart. It uses one text
     prompt per vehicle type: car, motorcycle (two-wheeler), bus, truck.
   - Each vehicle's ground point goes through the camera's homography. It is kept if it falls in the camera's zone,
     which gives its lane and its metres before the stop line.
   - Vehicles matched between the two frames give their speeds. The results go to table `vision_samples`.
3. **Observations:** per camera and 5-minute slot (table `observations`, `source = 'vision'`):

   | column | meaning |
   |---|---|
   | `total`, `counts_json`, `pcu` | vehicles in the zone at the sample nearest the slot's end, by type, in PCU |
   | `occupancy`, `load_ratio`, `level` | share of the zone's lane length occupied; PCU against the zone's capacity; free / moderate / heavy / jammed |
   | `flow` | vehicles per 5 min (Edie's definition, from the matched speeds; queued vehicles add nearly nothing) |
   | `speed` | mean speed between the speed lines (5–35 m before the stop line, `measure.speed_lines_m`), km/h |
   | `samples` | frame pairs in the slot. Fewer than `ingest.min_samples` (10) gives no row: a gap, not zero traffic |

4. **State, KPIs, model:**
   - `edge_state` / `edge_view`: every road direction, from its camera. A road without a camera stays empty here;
     the forecast estimates it from its neighbours.
   - `network_kpi`: vehicles in town, share congested, speed index, delay in vehicle-hours.
   - The graph model refits after new data (`data/<town>/predict/graph_model.pkl`; hourly with `--follow`).

The town appears in the dashboard's city list once `network_kpi` has rows. The forecast's usual day needs about a week of
recordings per day kind (working day, weekend).

## Dashboard

```powershell
.venv\Scripts\python -m streamlit run dashboard\app.py           # http://localhost:8501
```

- **City overview → Observed:**
  - the live road map, KPI tiles, the city flow score, the daily rhythm, the slowest roads, when delays happen
  - *What our studies found*: the most critical roads, the best time for roadworks, the slowest hour
- **City overview → Forecast:**
  - a future date and time: the expected traffic, recommended actions, the day's curve, the calendar
  - questions to the Traffic Assistant ("close High Street eastbound tomorrow 9am for an hour"). Its road changes
    become prediction cases, with a verdict (recommended / with caution / not recommended), quick insights and action
    items.
- **Technical view:**
  - network, timeline, per-edge charts, incidents, what-if, forecast
  - **Onboarding & ingest**, the operator's page for a real town:
    - the checklist, each step with a *Run* button (a background job, logged in `data/<town>/jobs/`)
    - live intake: start or stop the HTTP receiver and `observe --follow`
    - per-camera progress: stored, read, last observation
    - the inbox: waiting and refused files, with reasons
    - the calibration check
- **⋮ menu (top right):** the Traffic Assistant's model: Muse Spark (production) or Glimmer (local, development).

## Forecast and the Traffic Assistant

The forecast runs on the **graph model**, fitted on the town's observed days: the usual flow and speed per road, slot and
day kind, turn shares, capacity. Signal plans are not imported from the map, so a real town uses each road's
observed capacity. A road change spreads to neighbouring roads and reroutes traffic, and the result is the extra
delay at each affected junction. The numbers always come from the model. The language model only reads the question
and words the insight, and every number in its words is checked against the model's.

The assistant's model is chosen in the dashboard's ⋮ menu and saved in `data/assistant_backend.json`:

- **Muse Spark, Meta Model API: the production model.** Details below.
- **Glimmer, local: the low-cost development option** (no API cost, no network):
  - runs `llama-server` with `Muse-Glimmer-30B-Q3_K_M.gguf` (about 14 GB of GPU memory) on the workspace's
    `glimmer_gpu`. *Start* on the Forecast page, or picking it in the menu, starts it (about a minute).
  - by hand, with `N` = your `glimmer_gpu`:
    `$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"; tools\llama.cpp\llama-server.exe -m models\Muse-Glimmer-30B-Q3_K_M.gguf --jinja -ngl 99 -c 8192 --port 8080 --device CUDAN`
  - without a GPU that fits it, use Muse Spark.
- **Muse Spark, Meta Model API:**
  - picking it stops a local Glimmer. It needs a key from ai.developer.meta.com: `setx MODEL_API_KEY "your-key"`, or
    `MODEL_API_KEY = "your-key"` in `.streamlit\secrets.toml`, then restart the dashboard.
  - endpoint `https://api.meta.ai/v1/chat/completions`, model `muse-spark-1.1`; change them with `MUSE_SPARK_URL`
    and `MUSE_SPARK_MODEL`.
  - questions and model results are sent to Meta.

`predict.assist: off` in a town's config replaces the assistant with a simple road form.

## Command reference

`.venv\Scripts\netstudy <command> --sim-config towns/<town>.yaml`

| Command | Town | Does |
|---|---|---|
| `onboard [--fetch]` | real | map → graph, shapes, names; cameras; `map.png` |
| `cameras` | real | place and calibrate the cameras again (after editing them) |
| `receive [--host] [--port 8765]` | real | HTTP receiver into the inbox (`FLOWCAST_INGEST_TOKEN`) |
| `observe [--follow] [--poll 30] [--no-gpu-check]` | real | pickup → SAM 3.1 → observations → state, KPIs → graph model |
| `status` | real | the onboarding checklist |
| `graph-model [--until DATE] [--change EDGE LEVEL DATE HH:MM MIN]` | any | fit the graph model; try one road change |
| `ingest [--day] [--cameras] [--follow]` | any | read recorded segments only (no pickup, no state) |
| `state`, `kpis [--at | --day]` | any | network state / KPIs from the observations |
| `network`, `record`, `run`, `labels`, `score`, `frames`, `whatif-labels` | simulated | sample towns ([Simulation](#simulation-sumo--carla-sample-towns)) |

Scripts: `scripts\study_closures.py` (every road closed an hour per period; the overview's critical roads),
`scripts\check_assist.py` (the Traffic Assistant's question set), `scripts\check_e2e.py` (a town's status page).

## Data and code layout

```
towns/<town>.yaml            a town: map, cameras + calibration, inbox, windows, ingest, predict
towns/templates/real_town.yaml
workspace.example.yaml       which GPU runs what; copy to workspace.yaml (local to each machine)
data/<town>/
  inbox/ (rejected/)         pushed recordings, before pickup
  recordings/<date>/<period>/<camera>_<HHMMSS>.mp4
  network/                   shapes.json, names.json, osm.json (map centre, OSM ids) | a sample town: SUMO files
  cameras.json  graph.json  map.png
  traffic.db                 nodes, edges | received, ingested, vision_samples, observations | edge_state, edge_view,
                             edge_free_flow, network_kpi | study_closures, predict_eval_* (checks)
  predict/graph_model.pkl    the fitted graph model
  jobs/                      background jobs started from the dashboard (state + log)
models/                      SAM 3.1, Muse Glimmer (not in the repo: download them, see Setup)
sample-data/<town>.zip       the sample towns' results (no recordings; from Google Drive, see Setup), unpacked into
                             data/ on first use
src/netstudy/
  osm.py                     OpenStreetMap extract -> graph, shapes, names (no SUMO)
  realtown.py                real town onboarding: cameras on the map, homography calibration, camera zones
  inbox.py                   the inbox: pickup, the HTTP receiver
  observe.py                 pickup -> ingest -> state -> model; the onboarding checklist
  jobs.py                    background jobs for the dashboard
  workspace.py               the workspace machine's GPU roles (workspace.yaml, environment)
  vision.py                  SAM 3.1 detector
  ingest.py                  video -> vision_samples -> observations (real and simulated cameras alike)
  graph.py, kpis.py          network state and KPIs
  graph_model.py             the forecast engine; predict_data.py, actions.py, days.py
  assist.py, assist_backend.py, glimmer_server.py   the Traffic Assistant (Glimmer / Muse Spark)
  dashboard_data.py, overview.py, places.py          what the dashboard reads and how it names places
  network.py, simulate.py, zones.py, record.py, carla_world.py, carla_bridge.py, calib.py, direct.py, labels.py,
  score.py, pipeline.py      sample towns (SUMO + CARLA)
dashboard/                   app.py, overview.py, predict_view.py, technical.py, onboarding_view.py, shared.py
tests/                       pytest; tests/fixtures/millbrook.osm (a made-up town for the real-town tests)
```

## Workspace machine

Nothing about a particular machine is built into FlowCast. Each developer or deployment says which GPU runs what in
`workspace.yaml` in the project root. It is local to the machine and not shared: copy `workspace.example.yaml` and
edit it. Without the file, everything runs on GPU 0.

| Setting | Default | Meaning |
|---|---|---|
| `vision_gpu` | 0 | SAM 3.1: reads the recordings (`observe`, `ingest`). A town can override it with `ingest.gpu`. |
| `glimmer_gpu` | 0 | Muse Glimmer, the development Traffic Assistant (llama-server) |
| `carla_gpu` | the vision GPU | sample towns only: the card CARLA renders on, by part of its name |
| `one_gpu_at_a_time` | false | true when the power supply cannot carry two GPUs under full load: `observe` waits while another GPU is busy, and Glimmer will not start while one is |

Environment variables override the file: `FLOWCAST_VISION_GPU`, `FLOWCAST_GLIMMER_GPU`, `FLOWCAST_CARLA_GPU`,
`FLOWCAST_ONE_GPU_AT_A_TIME`. GPU numbers are nvidia-smi's (`nvidia-smi -L`). FlowCast sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` so that torch and llama.cpp number the cards the same way.

| Machine | Settings |
|---|---|
| one GPU, 16 GB or more | `vision_gpu: 0`, `glimmer_gpu: 0`. Stop Glimmer while `observe` runs, since 16 GB cannot hold both. Untested. |
| one GPU, under 16 GB | `vision_gpu: 0`, and Muse Spark in the dashboard menu |
| two GPUs | `vision_gpu` and `glimmer_gpu` on different cards. Set `one_gpu_at_a_time: true` if the power supply is tight. |
| no local GPU for language | Muse Spark; only SAM 3.1 needs a local GPU |
| production | not a workspace machine: SAM 3.1 on cloud GPU pools and Muse Spark through the Meta Model API ([docs/Architecture.md](docs/Architecture.md)) |

The GPU diagnostics (`scripts\check_gpus.py`, `gpu_test.py`, `gpu_series.py`) read the same settings.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

`tests/test_realtown.py` takes a real town through every step without SUMO, CARLA or a GPU:
- the map is the made-up town in `tests/fixtures/millbrook.osm`
- a calibrated camera
- recordings copied into the inbox and PUT over HTTP, some refused
- pickup, then ingest with a stand-in for SAM 3.1 that places cars at a known speed
- observations, state and KPIs, then what the dashboard reads, and the checklist

Other tests cover the graph model, the forecast, the Traffic Assistant (mocked), and the dashboard on the sample towns.

---

## Simulation (SUMO + CARLA sample towns)

Ashford, Brookhaven and Clearwater (`towns/ashford.yaml` …) are **sample towns**: hand-drawn junctions with demand
zones. SUMO drives their traffic and CARLA films it, so every part of FlowCast can be shown and scored before a real
town has a week of recordings. SUMO only makes this sample data: it is never used to predict, and a real town does
not use it at all.

```powershell
netstudy network --sim-config towns/ashford.yaml        # SUMO network, OpenDRIVE, cameras, graph, map.png
netstudy run     --sim-config towns/ashford.yaml --first 2026-10-05 --days 30   # record (CARLA) -> labels -> ingest
python scripts\run_towns.py                              # all three, unattended
```

- The first `source.render_days` days are filmed in CARLA (on the workspace's `carla_gpu`) and read by SAM 3.1, like a real town's recordings.
  The rest come straight from SUMO as the same `observations` (source `sumo`).
- `netstudy whatif-labels` makes paired SUMO runs, used only to score the graph model's road changes.
- `netstudy score` compares SAM 3.1 with SUMO's labels.
- CARLA 0.9.16 is installed by `scripts\setup_tools.ps1`.

## Limits and open points

- **Not yet run on a real city.** The real-town path is tested on a made-up town with a stand-in for SAM 3.1, and SAM 3.1
  has only read rendered video.
- **Traffic Assistant models:** Muse Glimmer for local testing, Muse Spark (Meta Model API) for live use.
- **Accuracy so far is on sample data:** flow error 25–37% per 5 minutes, speed error 5–7 km/h. The table is in the
  [status section](docs/Global%20Increasing%20Traffic%20-%20FlowCast%20-%20Build%20for%20Your%20World.md#proof-of-concept-status).
- **Signal plans** of a real town are not imported (OpenStreetMap has none), so the forecast uses each approach's observed
  capacity. Entering a real town's timing sheet (as the sample towns have under `network.town.signals`) is not
  supported yet.
- **People's labels** (counts checked by hand) are not built yet. A real town's ingest is checked through the
  calibration view and its observations, not scored.
- **Clock:** recordings carry local time and there is no time-zone handling. Keep the recorders on the town's
  clock (NTP).
- **Overpass:** `--fetch` is for town-sized areas. For larger ones, download an extract and set `network.osm.file`.
