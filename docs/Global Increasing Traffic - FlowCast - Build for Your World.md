# Global Increasing Traffic — FlowCast — Build for Your World

*A camera-first traffic intelligence system, designed to be automated from recording to recommendation: measure the
roads cameras can see, forecast the day ahead, and support earlier decisions.*

---

## Summary

Urban traffic is a large global problem under growing pressure from urbanisation and vehicle growth. Traditional
practice struggles at this scale: people watching CCTV walls, signal plans retimed every few years, and traffic studies
commissioned site by site.

**FlowCast** is designed to replace much of that manual work with an automated, AI-driven system. It reads recordings
from a city's cameras, measures traffic on the roads those cameras cover, learns how each road usually behaves, and
forecasts a chosen day and time, including the effect of a road closure before it happens. Planned later phases would
turn those forecasts into signal timing, traffic control, and evidence for infrastructure investment.

It is built on Meta's models:
- **SAM 3.1** finds vehicles in video. In production it would run on cloud GPU pools.
- **Muse Spark**, Meta's hosted model on the Meta Model API, is the planned production model for asking for forecasts in
  plain language, with no language GPUs for FlowCast to operate.

A working proof of concept runs today on a development machine. It has been exercised on sample data, and not yet on a
real city's recordings ([status](#proof-of-concept-status)).

---

## I. The problem

### 1. A global issue

Congestion is not a problem of a few megacities. INRIX's 2024 Global Traffic Scorecard ranks nearly 950 cities in 37
countries:

| Indicator | Figure | Source |
|---|---|---|
| Time lost by the most-delayed city's drivers | **105 hours per driver per year** (Istanbul); New York and Chicago 102 | INRIX 2024 |
| United States, average driver | 43 hours and about $771 a year | INRIX 2024 |
| United Kingdom, typical driver | 62 hours and about £581 a year | INRIX 2024 |
| Road traffic deaths worldwide | **1.19 million per year** | WHO, Global status report on road safety 2023 |

Road deaths are a road-safety figure, not a measure of congestion. They are included because they are part of the case
for managing traffic well.

### 2. An increasing rate

The pressure is projected to keep rising:

- **Urbanisation:** 68% of the world's population is projected to live in urban areas by 2050, up from 55% in 2018.
  That adds about 2.5 billion urban residents, nearly 90% of them in Asia and Africa (UN DESA, World Urbanization
  Prospects 2018).
- **Vehicles:** the global light-duty fleet was about 1.31 billion in 2020 and is projected to reach about **2.21
  billion by 2050** (US EIA, International Energy Outlook 2021).
- **Delay compounds when nothing changes:** a signal timing plan that is not updated loses ground as demand shifts.
  Delay is reported to grow by 3–5% a year at outdated signals (Minnesota DOT, 2014).

### 3. The scale of the problem

Large cities have thousands of signalised junctions: New York City DOT oversees more than 13,250 signalised
intersections (NY State Comptroller, 2024), and Greater London had 2,532 signalised road junctions in 2009 (London
Assembly report). Each junction has several approaches, and each approach behaves differently by hour and day type.

```
   New York City              13,250 signalised intersections  (sourced)
   × approaches per junction  × 4         = 53,000 approaches  (our assumption: a four-way junction)
   × hours per day            × 24        = about 1.27 million approach-hours of traffic every day
   × day types                × working day, weekend, holiday, event day …
   × what-ifs                 × every closure, roadworks, event, incident …
```

The United States alone has more than **330,000 traffic signals** (FHWA, Every Day Counts). The question is never
just "what is happening now" but also "what will happen on Thursday at 08:30 if this road is closed".

### 4. Traditional teams and signal systems cannot keep up

| Traditional practice | Why it does not scale |
|---|---|
| **Control-room operators watching CCTV** | Attention is finite. Lab studies report operators detecting fewer events as the number of feeds grows, with a drop between 9 and 16 feeds (Tickner and Poulton, reported in later literature). A study of 14 control rooms found at least 20 cameras per operator. Even at 16 feeds each, watching 53,000 approaches would take over 3,300 operators at a time. Watching also produces no measurements to learn from. |
| **Manual traffic counts and studies** | They give samples: particular sites on particular days, which age as the network and demand change. |
| **Fixed-time signal plans** | Agencies typically retime on a 3–5 year cycle, at about $4,500 per intersection (FHWA); only a minority retime that often (FHWA Arterial Management Program). Plans therefore drift away from actual demand. |
| **Reactive incident and roadworks planning** | The network-wide effect of a closure is hard to judge from experience alone, and an unplanned incident is usually dealt with after the queue has formed. |
| **Infrastructure decisions** | Without continuous measurement, business cases depend on sampled counts rather than the long-run behaviour of each road. |

The problem calls for:
- **continuous measurement** of the road network
- **learning** each road's usual behaviour
- **forecasting** days ahead and the effect of changes
- **acting** on those forecasts

That is impractical by hand at this scale. It calls for an automated system that people supervise.

---

## II. The solution: a fully automated, AI-driven traffic system

This is the design goal. Today, Phase 1 runs from recording to forecast without manual steps once a town is onboarded;
onboarding itself (map, camera placement, calibration points) is manual. The Act and Invest stages and the feedback loop
are not built yet. People set goals, approve actions and handle the exceptions.

```
   ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  OBSERVE   │──►│  UNDERSTAND  │──►│   FORECAST   │──►│     ACT      │──►│    INVEST    │
   │ roads, from│   │ how each road│   │ a chosen day,│   │ signals,     │   │ where the    │
   │ CCTV       │   │ behaves      │   │ road changes │   │ diversions   │   │ network needs│
   └────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
         ▲          Phase 1 (proof of concept)     │       Phase 2 and 3      Phase 4
         └───── measured results feed back (envisioned) ┘
```

### Phase 1 (proof of concept): study traffic behaviour and forecast a given day

- **Observe:** each calibrated camera becomes a counting and speed station for the one road direction it watches.
  Every 5 minutes, it reports:
  - the vehicles on its approach, by type
  - the flow and speed
  - the occupancy and congestion level

  Video comes in as recordings pushed by the city's existing systems, so no new roadside hardware is needed. Those
  systems do need setting up to export the recordings. Measured accuracy so far is in the
  [status section](#proof-of-concept-status).
- **Understand:** the measurements form the state of the whole road network:
  - each road direction with a camera (a road without one has no measurement; the forecast estimates it from its
    neighbours)
  - the network KPIs: vehicles in town, share congested, speed index, delay in vehicle-hours

  From that state, FlowCast learns the *usual* day for each road, time slot and day type, how traffic splits at each
  junction, and each road's capacity.
- **Forecast:**
  - **A given day:** pick a date (within the calendar's range) and a time, and FlowCast shows the expected state of the
    roads, the expected hotspots and the recommended actions. A day type with no recordings falls back on a similar one,
    and says so.
  - **What-if:** ask "close this road on Thursday at 09:00 for an hour" and FlowCast shows the extra delay at each
    affected junction and the diversions. A change sets up to three roads to busy, slow, jammed or closed, for 30 to 120
    minutes. It gives a verdict (*Recommended / With caution / Not recommended*) and an
    action list: detour signing, signal changes, officers on site.
- **Study:** the studies find the most critical roads, the best time of day for roadworks and the slowest hour of the
  week. The closure study is run by a script today, not on a schedule.

In the FlowCast dashboard:
- **City overview → Observed:** what the cameras measured.
- **City overview → Forecast:** the expected traffic and the what-if cases.
- **Technical view → Onboarding & ingest:** brings a new town in.

### Next phases: from forecast to control to infrastructure

| Phase | Capability | What changes for the city |
|---|---|---|
| **2. Forecast-based traffic signalling** (envisioned) | Timing plans computed from the forecast for each junction and period, and checked against the measured result. Starts as recommendations for engineers to approve, then automated within safety limits. | Goal: signal plans updated from measured demand, far more often than a 3–5 year retiming cycle. |
| **3. Traffic control** (envisioned) | Incidents found from the camera data. Diversion plans, message signs and temporary signal plans proposed and, where approved, applied. Planned events and roadworks scheduled for the least harm. | Goal: faster response to incidents, and planned closures chosen on evidence. |
| **4. Traffic infrastructure recommendation** (envisioned) | Long-run evidence on which junctions limit the network, which roads have no alternative, and where capacity, turn lanes, bus priority or new signals pay back. Changes are tested in the network model before anything is built. | Goal: capital budgets informed by measured need. |

### Why automation works here

- **Scale becomes mainly a compute cost:** more cameras need more GPUs, not proportionally more people watching.
- **Forecasts can be measured against what then happens:** this is how accuracy is scored on the sample data today. Doing
  it continuously on a real city's data is a production goal.
- **Numbers come from models, not opinion:** the language model only interprets questions and words the answer.
  Every figure it states is checked against the network model's output.
- **People stay in charge:** in the design, operators supervise, approve actions and override. The system does the
  counting, modelling and drafting.

---

## III. Meta's models in FlowCast

FlowCast is built on Meta's models, each in a defined role:

| Meta model | Role in FlowCast | How it is used |
|---|---|---|
| **SAM 3.1** (Segment Anything Model 3.1) | **CCTV traffic observation** | Text-prompted detection ("car", "motorcycle", "bus", "truck") on video frames. FlowCast does no training per camera or city; accuracy on real city CCTV is still to be measured. Each detection is placed on the road through the camera's calibration. Vehicles in the camera's measured zone are counted, and pairs of frames give speeds. The output is the per-camera, 5-minute observation. |
| **Muse Spark** (Meta Model API) | **Analysis and forecasting assistant** (planned production model) | Meta's hosted model. It reads an operator's question ("what if High Street is closed tomorrow at 9 for an hour?") into a structured plan of road changes, using structured output. FlowCast's network model computes the forecast, and Muse Spark words the key insights. FlowCast operates no GPUs for it, and its capacity depends on the API's limits. Image input, listed by third-party guides, could serve Phase 3 incident checks. Muse Glimmer is used for local testing, Muse Spark for live use. |
| **Muse Glimmer** (open weights) | The same assistant, **development option** | A low-cost stand-in for development and testing. It runs on a developer's local GPU, with no API cost and no network needed. It uses the same plan schema and the same checks as Muse Spark. A different model can read a question differently, so the question set must be re-scored on Muse Spark. |
| **DINOv3** (envisioned) | Camera health and scene understanding | Visual features to detect camera faults (moved, obstructed, blurred) and changes in the scene. |

### How the models deliver the solution

```
  City CCTV recordings
         │
         ▼
  ┌──────────────┐  vehicles, types,   ┌────────────────────┐   usual day, capacity,  ┌──────────────────┐
  │   SAM 3.1    │  positions, speeds  │ FlowCast network   │   turn shares, road-    │ Forecast         │
  │ (observation)│────────────────────►│ state + graph model│   change propagation    │ (chosen day,     │
  └──────────────┘                     └────────────────────┘────────────────────────►│  road changes)   │
                                               ▲                                       └────────┬─────────┘
  Operator: "Close Mill Lane northbound        │ structured plan                                │ figures
   on Friday 08:30 for 2 hours?"               │ (road, level, time, duration)                  ▼
         │                              ┌──────┴───────────────────┐   insight in plain words ┌──────────────┐
         └─────────────────────────────►│ Muse Spark  (production)  │◄─────────────────────────│ verdict,     │
                                        │ (understands, explains)   │  numbers checked against │ actions      │
                                        └───────────────────────────┘  the model's output      └──────────────┘
  Development: Muse Glimmer (open weights, local) takes Muse Spark's place at low cost; same plan, same checks.
```

| Solution need (II) | Meta model (III) | Result |
|---|---|---|
| Measure roads without staff watching | SAM 3.1 | Each calibrated camera becomes a counting and speed station |
| Avoid per-site model training | SAM 3.1's text prompts | A new town needs a map and calibrated cameras, with no model training; real-world accuracy to be measured |
| Make forecasts usable by non-specialists | Muse Spark | Questions in plain language; answers as verdicts, insights and actions |
| Serve operators without running language GPUs | Muse Spark (Meta Model API) | Hosted by Meta, within the API's limits |
| Scale observation to many cameras | SAM 3.1 on cloud GPU pools (design) | GPU count grows with the camera count |
| Develop and test at low cost | Muse Glimmer and SAM 3.1 on one local GPU (development option) | Phase 1 runs on a developer's machine |
| Keep the system reliable as cameras age | DINOv3 (envisioned) | Camera faults found automatically |

### Proof of concept status

Phase 1 runs end to end on a development machine, with these limits:

- **No real city yet.** The real-town path (OpenStreetMap map, calibrated cameras, pushed recordings) has been tested on
  a made-up town with a stand-in for SAM 3.1. SAM 3.1 itself has only read rendered video, not real CCTV.
- **Traffic Assistant models:** Muse Glimmer for local testing, Muse Spark (Meta Model API) for live use.
- **Results on the three sample towns** (Ashford, Brookhaven, Clearwater: simulated traffic, rendered video):

| Measure | Ashford | Brookhaven | Clearwater |
|---|---|---|---|
| Camera flow error per 5 minutes (WAPE, on the rendered day) | 37% | 30% | 25% |
| Camera speed error (km/h) | 6.0 | 6.8 | 5.2 |
| Real traffic inside the forecast's usual range | 74% | 72% | 74% |
| Roads really affected by a road change, and found (6 test cases) | 20 of 31 | 33 of 60 | 69 of 109 |
| False alarms in those cases | 0 | 17 | 39 |
| Answer time per question, median (Muse Glimmer) | 6.8 s | 7.0 s | 11.7 s |

These numbers are from sample data, and real-city accuracy may differ.

The production design (cloud GPU pools for SAM 3.1, Muse Spark through the Meta Model API) and the later phases
are in **Architecture.md**.

---

### Sources

- INRIX, *2024 Global Traffic Scorecard*: [inrix.com/scorecard-2024](https://inrix.com/scorecard-2024/);
  [UK press release](https://inrix.com/press-releases/2024-global-traffic-scorecard-uk/)
- New York State Comptroller, *Oversight of Selected Aspects of Traffic Controls*, 2024:
  [osc.ny.gov](https://www.osc.ny.gov/state-agencies/audits/2024/04/11/oversight-selected-aspects-traffic-controls-follow)
- London Assembly report, signalised junctions in Greater London (2009 figure):
  [london.gov.uk](https://www.london.gov.uk/media/37399/download)
- CCTV operator load, feeds and control-room survey: *Frontiers in Human Neuroscience*, 2013:
  [frontiersin.org](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2013.00615/pdf).
  The Tickner and Poulton figures are second-hand from later literature.
- World Health Organization, *Global status report on road safety 2023*:
  [who.int](https://www.who.int/publications/i/item/9789240086517)
- UN DESA, *World Urbanization Prospects 2018*:
  [key facts](https://population.un.org/wup/publications/Files/WUP2018-KeyFacts.pdf)
- US EIA, *International Energy Outlook 2021*, light-duty vehicle fleet:
  [eia.gov](https://www.eia.gov/todayinenergy/detail.php?id=50096)
- FHWA, *Arterial Management Program*, signal retiming practice:
  [ops.fhwa.dot.gov](https://ops.fhwa.dot.gov/publications/fhwahop09006/app_c.htm); FHWA Every Day Counts
  (EDC-4), signal count and retiming cost
- Minnesota DOT, report 2014-38, delay growth at outdated signal timing:
  [conservancy.umn.edu](https://conservancy.umn.edu/bitstream/handle/11299/169502/MnDOT2014-38.pdf?sequence=1)
