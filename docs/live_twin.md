# Live digital twin service (module 15)

`scripts/serve_twin.py` streams the CTM corridor in real time for the CitySense operations dashboard.
Config: `config/live.yaml` (validated by `citysense.live.config.LiveConfig`). Tests: `tests/module_15`.

```bash
uv pip install --python .venv/bin/python websockets   # uvicorn needs a WebSocket library
.venv/bin/python scripts/serve_twin.py                  # http://127.0.0.1:8003/api/twin/health
```

## Network (config/network.yaml)

10 four-way signalised junctions on 2.64 km of arterial (176 cells x 15 m) with two 180 m side-road legs each
(`cross_street_legs: 2`, `cross_cells_per_leg: 12`): 416 cells, 21 trip entries. Drawn on 2.7 km of Rajpur-Mussoorie
Road, Dehradun (`scripts/build_street_geometry.py`): 14 of 20 legs follow real side roads (some via a staggered
junction within 120 m); 6 have no mapped road on that side and are drawn as faint straight stubs. The CTM engine
models chains only (no merges), so side roads end at their junction rather than feeding the arterial.
The engine's signal mask is vectorised (~200 simulated s per wall s with 4 policy runs). Scenario bottlenecks
moved for the longer corridor: recurring at cell 47 (J3 stop line), spillback at 78 (before J5), creeping at 20
(it must be reached by steady traffic before its 200 s onset). `scripts/pretrain_live_dqn.py` pretrains the live DQN.

## What runs

* **One engine per policy, in lockstep.** DoubleDQN plus every `rl.baseline_algorithms` entry each get their
  own `NativeCTMEngine` + `SignalSafetyShield`. All of them see the same background demand
  (`demand.yaml` × `background_demand_factor`), the same planted scenario and the same agent trips. A
  per-episode comparison is therefore like-for-like. The dashboard map shows one run (`policy`).
* **Signals are wired into the engine** (`signal_controlled_cells`, `lanes`, `cell_length` from the network).
  See *Findings* below for why this matters.
* **Online learning, visible exploration.** The DQN acts ε-greedily with
  ε_e = max(ε_end, ε_start · `epsilon_episode_decay`^e) for episode e. It pushes one transition per junction per
  step (reward ÷ `queue_norm_factor`) and runs one Double DQN update per step. The checkpoint and episode history are
  saved after every episode (`outputs/live/`). `POST /api/twin/control {"learning": false}` switches to greedy-only.
* **Explainable decisions.** A decision record is logged whenever a phase switches, the shield overrides the
  proposal, or exploration changed the outcome. Each record holds the mode (explore / exploit / rule), ε,
  Q(keep), Q(switch), the shield reason, the 7 named observation features, gradient × input attributions on
  Q(switch) − Q(keep), and a deterministic one-line explanation (`citysense/live/explain.py`).
* **Car clusters.** `POST /api/twin/trips` adds a trip to every run's entrance buffer at its departure time. A
  Lagrangian tracer per run then waits FIFO in the buffer and moves at the local CTM cell speed. It stops at
  the stop line while its approach is red and records, for each junction, its wait and the id of the decision
  that released it. Exit gives travel time and delay against free flow.
* **Congestion events** use `diagnosis.congestion_speed_factor` with hysteresis (`jam_min_cells`,
  `jam_confirm_sec`, `jam_clear_sec`), so red-light stop-line queues do not flicker as jams.
* The existing rehearsal / root-cause API is mounted at `/api/twin/diagnosis` (`/rehearsal`, `/jobs/{id}`,
  `/situations/{id}`).

## Real-street geometry

`scripts/build_street_geometry.py` (run once, needs internet) routes the arterial on OpenStreetMap through the
public OSRM router, picks one real intersection per junction (closest to even spacing), routes each cross
street along the most perpendicular real side road, and caches everything in `config/street_geometry.json`.
`config/live.yaml: street_geometry_path` points the twin at it (null/missing file = schematic layout). Physics
are untouched: each CTM link is drawn along the real segment between its two intersections
(`link_scale` = drawn / simulated length, currently 0.79–1.86 on Rajpur-Mussoorie Road, Dehradun).

## Endpoints (paths from `config/live.yaml`)

`GET topology | state | events?after= | decisions?policy=&after= | episodes | platoons | health`,
`POST control {action: play|pause|step|reset, policy, scenario, steps_per_tick, learning}`,
`POST trips {trips: [{cluster_id, cluster_name, intent, plan_id, source_id, vehicles, depart_in_sec}]}`,
`WS stream` (a `hello` snapshot, then one `tick` per `tick_interval_sec` with state, new events, decisions and finished car clusters).

## Root-cause tracing of live congestion (`POST /api/twin/diagnose {cell}`, `citysense/live/diagnose.py`)

Each run keeps 15 min of noisy sensor observations (the repo's `ObservationGenerator`), engine snapshots every
`snapshot_every_steps` and its signal-phase history. For a clicked cell the repo's own pipeline runs on that data:
`JamDetector` → `ShockwaveTracer` → `CapacityEstimator` → a live counterfactual ladder → `build_situation`.
The ladder replays copies of the live engine from a snapshot taken `diagnosis_lead_sec` before the jam, with the
same background demand (perturbed per replicate) and the agent trips, and **with real signal controllers in every
rung**: baseline = the running policy; restore_capacity; demand_peak_shift; and every alternative signal policy
(DQN greedy, fixed-time, longest-queue, max-pressure), the best of which is the `rl_signal_control` rung. The RL
controller's behaviour at the junctions in and around the jam (green shares, switches, explore/exploit/shield,
Q-margins) is attached as evidence. The planted ground truth is returned only under `validation`.
Guards added after reviewing early traces:
* **Maturity gate**: a jam must have been confirmed by the twin for `diagnosis_min_jam_age_sec` (180 s) before it is
  traced; younger requests return `immature` with `retry_after_sec` (the dashboard shows those ripples in orange).
* **Forward rehearsal**: rungs replay from a snapshot before the jam, through now, and `diagnosis_forward_sec`
  (600 s) into the future, so effects are large enough to compare.
* **Verdict**: `conclusive` (top rung above `diagnosis_min_effect_vhr` and its 95% interval clear of the next),
  `tied` (intervals overlap) or `inconclusive` (largest effect below the minimum). The LLM must respect it.
* `origin.wave_speed_is_fallback` marks when the "estimated" wave speed is just the configured value.
* **Origin candidates** (chained bottlenecks): the tracer's front, internal discharge fronts (a congested cell whose
  observed downstream neighbours flow) and junction stop lines inside the jam, up to `diagnosis_max_origin_candidates`,
  each get their own capacity-restore replay; another candidate replaces the traced front only if its recovery is
  clear of the front's 95% interval. Note: restore acts on the twin's own capacity state; a field twin would use
  calibrated capacities, so localisation here is an upper bound.
* **Demand rung = best of** retiming the background peak and **thinning a trip surge** (entries whose trip rate in the
  replay window is at least `diagnosis_surge_ratio` x their earlier rate are scaled back to that earlier rate).
* **Detector flow noise** (`sensor_flow_noise_rel`, 10%) is added in the live twin, since the repo's generator reports
  exact flows.
Typical runtime ~5 s per trace (5 replicates, horizon up to 1500 s). Benchmark: `scripts/benchmark_live_diagnosis.py`
→ `reports/live_diagnosis_benchmark.md`.

## Findings from wiring this up (worth fixing in the research pipeline)

1. **FIXED: `report/evaluation.py` built the policy engine without `signal_controlled_cells`**, so all four
   controllers scored exactly 43.58. It now wires signals, lanes and cell lengths; regenerated
   `final_results.md`: longest-queue 45.15, max-pressure 61.45, DoubleDQN 67.23, fixed-time 76.31 mean queue.
   (Intervals are still zero-width: evaluation episodes are deterministic, the seed does not vary demand.)
2. **FIXED: the DQN was evaluated untrained**; it is now trained for `rl.total_training_steps` (200 steps, so still
   barely trained; raise that budget, which changes the config hash, for a meaningful comparison).
3. **The batch ladder's `rl_signal_control` rung is not the RL policy**: `counterfactual/ladder.py` uses a hard-coded
   all-arterial-green controller, and its other rungs run with `set_signal_controller(None)` (phases frozen). The live
   ladder runs the actual controllers instead.
4. **`ShockwaveTracer.trace_origin` uses each cell's earliest observation** (any speed) as its onset, so given a long
   observation buffer it reports onsets from before the jam. The live pipeline passes only the jam's time window.
5. **restore_capacity searches only downstream of the origin**; the discharge-front origin is often one cell past
   the true bottleneck, which made the rung recover 0. The live ladder searches ±`bottleneck_search_radius_cells`.
6. **FIXED: the demand rung produced negative demand.** `ladder.py` evaluated `base_demand.get_rate(t - peak_shift_sec)`,
   which also shifts the warm-up and returns negative rates for t < 420 s (e.g. −0.44 veh/s at t = 150 s), deleting
   vehicles from the replay and inflating "demand peak shift" recovery. Fixed in both ladders: the retimed profile is
   `CorridorDemand(config, peak_shift_sec=...)` (moves only the peak window), and `get_rate` clamps negative time.
7. **`demand_peak_shift` scenario is a no-op**: `DemandPeakShiftScenario.apply_to_engine` does nothing, so in the engine
   it is identical to `control`; the suite cannot test demand-caused jams with it.
8. **Sensor flows are noiseless** (repo generator unchanged; the live twin adds `sensor_flow_noise_rel` noise): `ObservationGenerator` multiplies flow noise by 0.0 and reports the true flow, which
   makes the capacity fit easier than with real detectors (its CI collapses to a point).
9. **Cross-street approaches are sinks** (`downstream = -1`). The engine only applies the signal mask to cells
   with a downstream cell, so a cross-street red never holds traffic, and any time spent in the cross phase
   only costs the arterial. The learned policy can exploit this by favouring arterial green. The live twin
   mirrors the engine, so the bias is present here too. Give cross streets a downstream exit cell, or mask sink
   exits for signalised cells.

## Benchmark (`scripts/benchmark_live_diagnosis.py`, report in `reports/live_diagnosis_benchmark.md`)

54 runs (6 cases x 3 running policies x 3 seeds), pinned DQN checkpoint, plus a legacy-demand comparison and an
ablation of the candidate search / surge thinning. Headline (mature jams): spillback 100% correct (origin error 1.0
cells), creeping 100% (2.15), recurring at a stop line 82% (0.00; capacity a material cause in 100%), no false
capacity conclusions on negative controls. Still weak: a trip surge is named as demand in only 40% of surge
diagnoses (signals are credited in 60%), and the surge detector also fires on ordinary agent variation.
