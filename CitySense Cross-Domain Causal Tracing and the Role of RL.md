# CitySense: Cross-Domain Causal Tracing and the Role of RL

Positioning: an Urban Intelligence and Civic Coordination Platform

Status: design document for the hackathon build. All scenario details are illustrative. No performance figures are claimed. Anything labelled simulated is simulated.

## 1. Summary

CitySense links symptoms that appear in one part of the city to causes that sit in another department's data, then helps a human decide what to do about it and records what actually happened afterwards.

The core idea in one sentence: a complaint on one street is usually a symptom, and the cause often lives in a different system, such as a water main, a utility permit or a transit change.

Reinforcement learning (RL) has one precise job in this design. It is the adaptive controller inside the digital twin, so that every counterfactual the city rehearses reflects how the city would actually respond, and so that response options come from learned policies rather than fixed rules. It is not the part that finds the cause. Section 7 states this exactly.

## 2. The problem

- Departments hold separate data: roads, water, sanitation, transit, power, complaints.
- A single underlying fault produces symptoms in several of them. Each department sees only its own symptom and files its own ticket.
- Repairs get aimed at the symptom. A road crew patches potholes that a leaking main keeps destroying.
- Decisions are approved on a map view, with no test of who they affect, and nobody checks afterwards whether the decision worked.

## 3. The solution: a decision lifecycle

1. **Observe.** Citizen reports, sensor readings, permits, work orders, schedules and traffic data become observations with a type, location, time and source.
2. **Correlate.** Deterministic clustering groups observations into a situation, with the evidence listed.
3. **Trace.** The system walks dependency layers (pipes, drains, road graph, bus routes, permits) to find candidate causes in other departments.
4. **Propose.** Specialised agents with different objectives propose options.
5. **Rehearse.** A digital twin tests each option and each causal hypothesis by counterfactual simulation.
6. **Decide.** A human approves, edits or rejects. Dissent is recorded.
7. **Track.** The outcome tracker compares predicted and observed results, using later complaints and counts as ground truth.
8. **Calibrate.** Confirmed or rejected hypotheses adjust future scoring.

## 4. Worked example A: the street that keeps cracking

What each department sees separately:

- Roads: nine pothole reports on Lane 4 over three weeks.
- Sanitation: five reports of damp walls and smell on the same lane.
- Water board: a pressure dip in one zone, logged as noise.

What CitySense does:

1. Every report becomes an observation.
2. The correlator clusters the road and sanitation symptoms by place and time into one situation on Lane 4.
3. It then walks the water-pipe layer rather than the road layer, and finds that a main runs beneath Lane 4, downhill from a junction about a kilometre away.
4. Along that pipe it finds the pressure dip and two earlier low-pressure complaints.
5. It scores the hypothesis "leaking main near the junction": the cause precedes the symptoms, the damage follows the pipe rather than the road, and a known mechanism exists (leaks undermine road surface).
6. Output: one situation, a ranked hypothesis, the evidence trail and a proposed action to inspect the main, owned by the water board. Road patching is flagged as likely wasted effort until the main is checked.

RL is not involved in this example. The tracing is graph traversal and rule-based scoring. Saying otherwise would be wrong.

## 5. Worked example B: the jam with no cause on its own street

Market Street is jammed every evening since Monday. Traffic data shows congestion and nothing on that street explains it.

1. CitySense searches the road graph for recent changes on routes that feed Market Street.
2. It finds a utility-trench permit narrowing a lane on a parallel arterial about two kilometres away, starting the same Monday.
3. The twin reruns the evening peak with that lane closed. If the simulated jam on Market Street matches the observed one, the hypothesis gains weight. If it does not, the hypothesis drops in rank.
4. Options to relieve it are generated and rehearsed: retime signals, shift the permit's working hours, or divert a bus route.

This is where RL matters, explained next.

## 6. Architecture and module status

The existing stack is preserved. No API contract or schema change is made without explicit approval.

- **Keep:** governance and approval workflow, audit log, PostgreSQL with PostGIS and pgvector, the multilingual RAG assistant, complaint tracking and escalation.
- **Evolve:** NetworkX routing becomes the generator of diversion and corridor options. The SUMO and DQN work is retargeted from optimising signals in general to evaluating and proposing policies under disruption.
- **Deprioritise:** YOLOv8 becomes one optional observation source with a recorded-clip fallback. Flood and accident prediction becomes an input to situations rather than the product.
- **New:** a Situation entity linking observations, evidence, confidence and status (proposed as a migration, not applied silently). The dependency-layer tracer. The options and disagreement view. The outcome tracker. An asynchronous rehearsal API between the platform and the simulation service.
- **Separation of concerns:** deterministic logic decides situations, scores and approvals. LLMs only parse typed requests into schema-validated JSON and explain results in plain language. They never produce numbers or decide a cause.

## 7. The role of RL, exactly

### 7.1 What RL does

**Role 1: adaptive control inside the twin.** When a lane closes, a real city does not keep fixed signal timings. Operators and adaptive systems respond. If the twin used fixed timings, every counterfactual would overstate the damage and the hypothesis tests would be biased. A trained multi-agent policy controls the signals in the simulation so that the rehearsed future reflects an adaptive response.

**Role 2: generating response options.** Policies trained with different reward weightings produce different candidate plans for the same situation. Examples are an emergency-first policy, an equity-first policy, an emissions-first policy and a balanced policy. Each plan becomes a point on a trade-off chart, and the operator chooses. This is how agents with conflicting goals produce visibly different proposals without any language-model negotiation.

**Role 3: recommending signal and diversion adjustments for recurring disruption.** For repeated conditions such as school-gate congestion or a standing lane restriction, the policy proposes timing changes that the twin evaluates before a human approves them.

### 7.2 What RL does not do

- It does not find the cause. Correlation and tracing are deterministic.
- It does not run the water, waste or lighting examples. Those use clustering, graph traversal and prioritisation rules.
- It does not act on the real city. Every action passes through human approval, and hard limits wrap the policy.
- It does not produce the explanation. That is the language model's job, working from the twin's measured output.

### 7.3 Formulation

- **Environment:** a real corridor of roughly six to ten signalised junctions imported from OpenStreetMap into SUMO, with demand fitted to whatever traffic counts are available.
- **State per junction:** queue lengths, current phase and elapsed time, disruption flag, emergency-vehicle proximity, bus presence.
- **Action:** next signal phase and duration, bounded by safety limits.
- **Reward:** negative delay, minus penalties for emissions, ambulance delay and the delay of the worst-off group (for example night-shift routes and bus riders). The weights define the policy's character.
- **Algorithm:** multi-agent PPO with shared weights across junctions, using sumo-rl or RLlib. Training is done offline and the saved policy is shipped. Nothing trains live.
- **Safety shield:** minimum green times, maximum cycle length and pedestrian phases are enforced outside the learned policy.

### 7.4 How RL connects to the rest of the system

1. The correlator proposes a situation and ranked causal hypotheses.
2. For hypotheses involving road capacity, the twin applies the suspected change, with the RL policy controlling signals, and compares the simulated congestion with what was observed.
3. Agents request plans from policies with different reward weights.
4. The twin runs each plan several times with perturbed demand and reports ranges, not single values.
5. The operator sees the trade-off chart, decides, and the dissent is logged.
6. The outcome tracker compares predictions with later observations.

### 7.5 How RL is validated

- **Baselines:** fixed-time signals and max-pressure control. Max-pressure is a strong baseline and RL often matches it closely, so a win is not assumed.
- **Where a win is plausible:** disruption conditions the baselines were not tuned for, and objectives beyond average delay, such as protecting specific groups.
- **Backtest:** replay recorded disruptions from open incident or work-permit data for the chosen area, and compare against the controllers cities actually run.
- **Reporting:** state results honestly. If the policy does not beat max-pressure, report where it does and does not, and present the framework as supporting learned policies.
- **Fallback:** if training does not converge in time, use heuristic controllers per objective. The trade-off chart and the counterfactual tests still work, and the claim changes accordingly.

## 8. Human governance

- High-impact actions require explicit approval by an authorised operator.
- Every recommendation carries evidence, reasoning, confidence, responsible department, proposed action, approval status and audit history.
- The system distinguishes recommendations, simulations, predictions and verified real-world actions, and never presents a simulated result as a verified one.
- Dissent from any agent is recorded alongside the decision.

## 9. Demo storyline

1. Complaints about Lane 4 arrive from three departments as separate tickets.
2. The correlator merges them into one situation and shows why they are linked.
3. The tracer walks the pipe layer and surfaces a ranked hypothesis in another department's data.
4. The operator approves a targeted inspection and pauses wasted patching.
5. Switch to Market Street: the trench-permit hypothesis is tested in the twin with the RL policy controlling signals.
6. Agents propose conflicting relief plans. The trade-off chart appears and the operator chooses.
7. Close with the outcome tracker comparing predicted and observed results, clearly labelled as simulated.

## 10. Risks and limitations

- Correlation is not causation, so outputs are ranked hypotheses, never a declared cause.
- Real dependency data is hard to obtain and join. The demo uses hand-built layers for a few kilometres, labelled simulated.
- Mechanism tables are hand-authored and kept small and visible.
- RL training may not converge in the available time, and RL may not beat max-pressure.
- The platform is wide. Polish the correlator, the twin and the governance flow first, and cut peripheral modules.
- Outcome tracking in the demo uses simulated outcomes.

## 11. Open decisions

- Which city or corridor, and what real data is available: traffic counts, incident or permit records, drainage or pipe maps.
- Days remaining, which sets how far the backtest goes.
- Who owns the simulation and training work.
- Whether the water-pipe layer, the permit layer, or both are built for the demo.

## 12. One-line pitch

Complaints are symptoms. CitySense finds the cause in another department's data, rehearses the response on a twin, lets a human decide on the record, and checks afterwards whether it worked.


# CitySense briefing: decision lifecycle with a CTM twin and a learned signal policy

## Scope
A platform that links symptoms in one department's data to causes in another's, rehearses responses on a digital twin, lets a human decide on the record, and checks the outcome afterwards. RL has one job: adaptive signal control inside the twin and option generation.

## Pipeline
1. **Observe:** reports, speeds, permits and weather become typed observations (type, location, time, source).
2. **Correlate:** cluster them into a situation, with an evidence list.
3. **Trace:** find candidate causes in other layers.
4. **Propose:** policies with different goals generate options.
5. **Rehearse:** run each option on the twin, several times with perturbed demand.
6. **Decide:** a human approves, and dissent is logged.
7. **Track:** compare predicted and observed outcomes.

## Components and tools

| Component | Technique | Tools |
|---|---|---|
| Platform, approvals, audit | Existing governance module | Spring Boot, PostgreSQL with PostGIS and pgvector |
| AI service | Typed endpoints, async rehearsal jobs | FastAPI |
| Correlator | Spatio-temporal clustering (DBSCAN-style), evidence scoring | scikit-learn, PostGIS |
| Cause tracing | Walk dependency graphs (roads, pipes, permits). On roads, back-trace the jam's tail with `w = (q2 − q1) / (k2 − k1)` | NetworkX, NumPy |
| Twin | Cell transmission model, triangular fundamental diagram, PCU units, signals as capacity gates | Python and NumPy, OSM corridor |
| Optional validation | Higher-fidelity check of key scenarios | SUMO |
| RL | Double DQN, one shared network across junctions, neighbour queues in state | PyTorch (Stable-Baselines3 for a PPO comparison) |
| Language layer | LLM only parses requests into schema-validated JSON and explains results | Existing RAG assistant |
| Citizen side | Complaints, multilingual assistant | Existing Android app |

## The RL spec
- **State per junction:** normalised queues per approach, current phase, time in phase, disruption flag, neighbour queues.
- **Action:** keep or switch phase, with minimum green and maximum cycle enforced outside the network.
- **Reward:** negative total queue, plus an extra penalty on the worst approach (the equity term).
- **Training:** offline, with randomised demand and random capacity drops. The saved policy is shipped, and nothing trains live.
- **Baselines:** fixed-time and longest-queue. I haven't run any training, so there's no performance claim.

## Data requirements
- **Tier 0 (to start):** OSM corridor of 4 to 6 junctions, assumed link parameters drawn from published Indian ranges, synthetic peak demand, random disruptions. Labelled as assumptions.
- **Tier 1:** one real speed-time series to sanity-check the jam shape. Candidates found earlier: DataMeet's Bangalore traffic data, a Zenodo Delhi dataset, the Aalto city-level speeds. I haven't verified coverage, so check each.
- **Tier 2:** real permits or incident records for backtesting.
- **Known gaps:** speed doesn't give capacity, signal timings aren't public, and per-link history is scarce.

## Compute and team
- CPU is enough for the CTM and a small DQN. A free GPU is optional.
- Docker for deployment.
- Roles: twin and RL owner, backend and governance owner, correlator and tracer owner, frontend and agents owner.

## Validation
- Compare against both baselines on the same scenarios.
- Hold out a day for validation, run a sensitivity analysis on assumed parameters, and backtest on real disruptions if data allows.
- Report where the policy ties or loses.

## Rules we hold to
- Deterministic logic decides situations, scores and approvals. LLMs never produce numbers or decide a cause.
- High-impact actions need human approval.
- Simulated results are always labelled simulated.
- No API or schema change without explicit approval.

## Build order
1. CTM environment and baselines.
2. Double DQN, then optional PPO.
3. Correlator and tracer.
4. Platform integration.
5. Outcome tracker and dashboard.
6. Demo rehearsal.

## Open decisions
Which city and corridor, which real data is obtainable, days remaining, and who owns the twin.

I can put this into a shareable doc for your team, or write the minimal CTM environment and training script. Which first?

These are scenario patterns I built from how Indian roads typically behave. They aren't documented incidents, so label them as scenarios. Each has the same shape: a symptom on one street, a cause in another system, and a way to test it. None involves an accident or a natural disaster.

**1. The Friday jam (weekly market)**
- **Symptom:** an arterial crawls every Friday, 5 to 8 pm.
- **Hidden cause:** a weekly market and parked vehicles narrow the carriageway.
- **Link found by:** the correlator spots the weekly repeat, then matches it to a market and event calendar layer.
- **Twin test:** fit a lower capacity to the observed speeds and rerun Friday evening in the CTM.
- **RL's role:** retime signals around the reduced capacity.
- **Decision:** shift market hours or relocate a stall block. Compare options in the twin.

**2. The jam after the school-timing change**
- **Symptom:** a new peak appears 40 minutes earlier on a feeder road.
- **Hidden cause:** a school group moved its start time, shifting when parents' trips happen.
- **Link found by:** policy-change log lined up with the demand shift.
- **Twin test:** move the demand peak in the CTM and check that it reproduces the jam.
- **RL's role:** retime for the new peak.
- **Decision:** stagger start times across nearby schools.

**3. The slow garbage truck (waste to traffic)**
- **Symptom:** a recurring 8 am queue at one narrow stretch.
- **Hidden cause:** a collection truck crawls and stops along that road at peak hour.
- **Link found by:** time-stamped truck GPS against the speed dip.
- **Twin test:** insert a slow, partially blocking vehicle into the CTM.
- **RL's role:** none. This is a scheduling fix.
- **Decision:** move collection to off-peak and rehearse the effect.

**4. The relocated bus stop**
- **Symptom:** a junction blocks daily, with no obvious cause.
- **Hidden cause:** a stop moved 100 m upstream, so buses queue back into the junction.
- **Link found by:** transit stop-change record against the date the queue began.
- **Twin test:** CTM spillback from the stop cell blocking the junction.
- **RL's role:** minor. The fix is physical.
- **Decision:** move the stop back or add a bay.

**5. The slowly shrinking road (construction creep)**
- **Symptom:** complaints grow week after week on a corridor.
- **Hidden cause:** work-zone barricades creep outward past the permitted footprint.
- **Link found by:** change-point detection on fitted capacity per week, matched against the permit footprint.
- **Twin test:** re-run with the fitted capacity.
- **RL's role:** none for detection, retiming once capacity is known.
- **Decision:** enforce the permit boundary.

**6. The dark stretch that slows traffic**
- **Symptom:** speeds fall after dusk on one stretch.
- **Hidden cause:** a streetlight outage cuts effective capacity at night.
- **Link found by:** lighting-fault layer against the dusk speed drop.
- **Twin test:** apply the night capacity reduction.
- **Decision:** rank repairs by **vehicle-hours lost**, not by complaint count. This is a clean cross-department prioritisation story.

**7. The dark-store curb jam**
- **Symptom:** a lane is blocked every evening near a block of shops.
- **Hidden cause:** delivery riders cluster at a quick-commerce dark store.
- **Link found by:** business-location layer plus order-hour peaks, if you can obtain them.
- **Twin test:** capacity loss from curb blocking.
- **Decision:** a designated pickup bay and staggered dispatch.

**8. The spillback that blames the wrong junction**
- **Symptom:** a junction looks "badly timed".
- **Hidden cause:** a downstream bottleneck is spilling back into it.
- **Link found by:** **shockwave back-tracing**, with the jam tail's upstream speed pointing to the true origin.
- **RL's role:** a policy trained on the junction alone would learn the wrong lesson. Retiming it won't help, and the twin shows that.
- **Decision:** fix the downstream cause.

**Where RL actually appears**

| Example | Tracing | CTM | RL |
|---|---|---|---|
| 1 Friday market | Yes | Yes | Yes |
| 2 School timing | Yes | Yes | Yes |
| 3 Garbage truck | Yes | Yes | No |
| 4 Bus stop | Yes | Yes | Minor |
| 5 Construction creep | Yes | Yes | After detection |
| 6 Streetlight | Yes | Yes | No |
| 7 Dark store | Yes | Yes | No |
| 8 Spillback | Yes | Yes | Shows its limit |

RL is central in only two of eight. That's accurate and it strengthens the pitch: **the platform is the tracer plus the twin, and RL is one tool inside it.** Example 8 is also your best demo moment. It shows the system refusing to blame the wrong junction.

**Data check:** examples 1 to 3 need calendars or truck GPS, which rarely exist as open data. Examples 5, 6 and 8 work from speed series alone, so they're the most feasible with what you can obtain.

Want a storyboard for 1 and 8 as your two demo cases?