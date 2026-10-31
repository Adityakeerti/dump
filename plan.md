# ROLE
You are a senior engineer building "CitySense Traffic": a CTM-based traffic twin that finds the ROOT CAUSE of recurring or unexpected traffic jams, tests fixes by counterfactual simulation, and uses a reinforcement-learning signal policy as one tool inside that test. Work autonomously, module by module, and obey the gate protocol below. Do not ask me questions unless you hit a blocker you cannot resolve.

# PRODUCT IN ONE PARAGRAPH
A jam on one street is often a symptom of a cause somewhere else on the corridor (a capacity drop upstream or downstream, a demand shift, a spillback from a downstream bottleneck). The system (1) simulates a corridor with the Cell Transmission Model (CTM), (2) produces sensor-like observations, (3) detects and clusters jams in space-time, (4) back-traces shockwaves to the true origin, (5) fits effective capacity per cell and time to find capacity drops, (6) ranks candidate causes with a counterfactual ladder that measures vehicle-hours recovered, and (7) uses an RL signal policy to test whether better signal control alone would fix the jam. RL does NOT find causes. Diagnosis is deterministic. Every output is labelled SIMULATED unless it came from a real data file.

# HARD RULES
1. CTM is the only traffic engine. No SUMO, no other simulators.
2. NOTHING HARD-CODED. Every physical parameter, coordinate, radius, time step, threshold, tolerance, seed, path, URL, hyperparameter, reward weight, scenario definition, and file name comes from YAML config validated by pydantic schemas. Constants allowed in source: 0, 1, -1, 2, and named unit-conversion constants defined once in units.py. Scenario types (planted causes) are registered by name and parameterised entirely in YAML.
3. Enforce rule 2 with a test: tests/test_no_hardcoding.py scans src/ with the ast module and fails on numeric literals outside the allowed set, and on string literals that look like paths or URLs.
4. Test expectations must be computed from config and theory (for example the triangular fundamental diagram formulas), never copied from a previous output. Test tolerances live in tests/tolerances.yaml, each with a one-line justification.
5. Determinism: all randomness flows from seeds in config via numpy Generators. Same config + seed gives identical results.
6. Dependencies allowed: numpy, scipy, scikit-learn, networkx, pydantic, pyyaml, pytest, torch, gymnasium, fastapi, uvicorn, httpx, osmnx, matplotlib, lightsim. Anything else needs a written justification in docs/decisions.md first.
7. Ground truth (the planted cause) is written to a separate file and is NEVER readable by the diagnosis code. Enforce with an import/path test.
8. Honesty: never invent numbers. Report only what was measured. If the RL policy does not beat baselines, report that. Do not claim real-world accuracy.
9. Deterministic logic decides situations, scores and rankings. No LLM calls anywhere in this project.

# GATE PROTOCOL (applies to every module)
For each module M in order:
 a. Write the code and its tests.
 b. Run the tests for M and ALL earlier modules (regression).
 c. If anything fails, fix and rerun, up to max_fix_attempts (set in config/project.yaml). If still failing, STOP and write reports/BLOCKER_M.md with the failing output, your diagnosis and options. Do not continue.
 d. On success, write reports/module_M.md: what was built, test results, deviations from this spec, open risks.
 e. git commit with message "module M: pass" and tag it. Only then start the next module.
Never skip a gate, never weaken a test to make it pass, and never delete a failing test. If a test is wrong, explain why in the report and fix it transparently.

# REPO LAYOUT
config/         project.yaml, engine.yaml, network.yaml, demand.yaml, scenarios/*.yaml, observation.yaml, diagnosis.yaml, rl.yaml, api.yaml
src/citysense/  config/ engine/ network/ scenarios/ observation/ diagnosis/ counterfactual/ rl/ situations/ api/ tracker/ report/
tests/          one folder per module, plus tolerances.yaml
reports/        module reports, blocker reports, final results
docs/           engine_recon.md, decisions.md, README.md

# MODULES

## Module 0: Recon and environment
Create a virtualenv, install dependencies, git init. Install lightsim and INTROSPECT the installed package (do not assume function names): list public classes and functions, how networks are loaded from OpenStreetMap, how demand is generated, how signals and controllers are defined, how to read per-cell density and flow, and whether per-cell capacity can be edited at runtime. Write docs/engine_recon.md with what you found. Run one built-in scenario end to end.
Decision: if lightsim cannot support runtime capacity edits or per-cell logging, either wrap it or implement a native vectorised NumPy CTM behind the same adapter interface (Module 2). Record the choice in docs/decisions.md.
Gate tests: environment imports cleanly; a built-in scenario runs for a number of steps from config; the recon file exists and lists the facts above.

## Module 1: Configuration system
Pydantic models and YAML loaders for every config file, with cross-file validation (for example the time step must satisfy the CFL condition against free-flow speed and cell length). Config hash is computed and stamped into every output. Unknown keys are rejected.
Gate tests: valid configs load; each invalid case (missing key, negative capacity, CFL violation, unknown key) raises a clear error; the config hash is stable and changes when a value changes; the no-hardcoding test is active.

## Module 2: Engine adapter
Define an abstract EngineAdapter: reset(seed), step(signal_actions), state() returning per-cell density, flow and speed, set_capacity_factor(cell_ids, factor, time_window), set_demand_profile(profile), set_signal_controller(controller), and snapshot/restore for counterfactual reruns. Implement it over lightsim, or the native CTM per the Module 0 decision. Everything downstream depends only on this interface.
Gate tests (expected values computed from config parameters):
 - Fundamental diagram: sweep demand on a single link, steady-state flow versus density matches the triangular diagram on both branches, with R-squared above the tolerance in tolerances.yaml.
 - Conservation: vehicles in + initial = vehicles out + remaining, within tolerance.
 - Shockwave: reduce capacity in one cell, the measured backward wave speed equals w = (q2 - q1)/(k2 - k1) computed from the two states, within tolerance.
 - Snapshot/restore reproduces an identical future.
 - A capacity factor of 1 changes nothing.

## Module 3: Network and scenario builder
Build the corridor from config (centre point and radius or bounding box, number of signalised junctions, all from YAML). Assign link parameters by sampling ranges from config with a seed, and label them as assumptions in metadata. Generate time-varying demand (peak profile, weekday/weekend pattern) from config. Implement a registry of planted-cause scenarios, each fully YAML-defined: (a) recurring capacity drop at a cell on set weekdays and hours; (b) capacity that shrinks week over week; (c) demand-peak shift; (d) downstream bottleneck whose spillback blocks an upstream junction; (e) a "no planted cause" control. Each writes ground_truth.json (cause type, cell, time window, magnitude) to a separate folder.
Gate tests: same seed gives the same network; the network is connected and has the requested junction count; ground truth matches the config; the control scenario has no capacity edits; a scenario with a capacity drop produces measurably lower flow at that cell than the control (threshold from tolerances.yaml).

## Module 4: Observation layer
Turn simulation state into sensor-style observations: sampled cells (coverage fraction from config), reporting interval, Gaussian or multiplicative noise and dropout from config. Also write a loader for real speed series files, with the column mapping and units given in YAML. Output a typed observation table (cell or segment, time, speed, flow if available, source, simulated flag).
Gate tests: dropout fraction and noise statistics match config within tolerance; the loader maps a generated CSV round trip exactly; the simulated flag is always set for simulated data; observations never contain ground-truth fields.

## Module 5: Jam detection and clustering
Detect congested space-time points as speed below a configurable fraction of each cell's free-flow speed. Cluster them with DBSCAN-style clustering on (position, time) with scaling and eps from config. Report each jam's extent, duration, tail and head. Detect recurrence by matching jams across days.
Gate tests: on planted scenarios, jam recovery against the ground-truth congested mask meets precision and recall tolerances; the control scenario yields no jams above the false-alarm tolerance; the recurring scenario is flagged recurring and the single-day scenario is not.

## Module 6: Shockwave back-tracing
Estimate wave speed from observed flow and density (or speed-derived density using the config fundamental diagram), trace the jam tail upstream and the discharge front to the origin cell, and output the origin with an evidence trail.
Gate tests: across N seeded planted scenarios (N from config), origin cell error is within the tolerance in cell units; in the spillback scenario the traced origin is the downstream bottleneck and NOT the junction with the most congestion; estimated wave speed matches the theoretical w within tolerance.

## Module 7: Capacity fitting and change detection
Estimate effective capacity per cell and time window by fitting the observed flow-density points to the triangular diagram. Detect capacity drops and creep with change-point detection. Output the capacity ratio against nominal with a confidence interval.
Gate tests: recovered capacity ratio matches the planted ratio within tolerance; the creeping scenario's change points are detected within the allowed delay; the control scenario produces no significant drop.

## Module 8: Counterfactual ladder
For each candidate cause, rerun the same scenario from the snapshot with perturbed demand replicates (count and perturbation size from config). Rungs: restore capacity at the suspect cell, shift or smooth the demand peak, and apply the best available signal policy. Metric: vehicle-hours recovered versus the observed run, with mean and interval across replicates. Rank hypotheses by recovery and report which rung explains the jam.
Gate tests: on a planted capacity drop, the restore-capacity rung ranks first; on the spillback scenario, the signal-policy rung recovers significantly less than the downstream capacity rung; the control scenario shows no rung with recovery above the noise floor; results are identical across reruns with the same seed.

## Module 9: RL signal control
Gymnasium and PettingZoo style environment on the engine adapter. State per junction: normalised queues per approach, phase, time in phase, disruption flag, neighbour queues. Action: keep or switch phase. Reward: negative total queue plus a worst-approach penalty, weights from config. Safety shield OUTSIDE the network: minimum green, maximum cycle, yellow and all-red from config. Algorithm: Double DQN with one shared network across junctions (PyTorch), with an optional comparison algorithm from config. Train offline with randomised demand and random capacity drops, and save a checkpoint with config hash, seed and training metadata. Baselines: fixed-time, longest-queue, max-pressure, evaluated on identical seeded scenarios.
Gate tests: observation and action shapes match config; a property test shows the shield is never violated over random actions; a tiny training smoke run completes and saves and reloads a checkpoint that reproduces the same actions; the evaluation report compares RL and every baseline on identical scenarios with mean and interval. Pass the gate if the pipeline is correct. Do not require RL to win, and print results as measured.

## Module 10: Situation records and hypothesis ranking
Define a Situation entity: id, observations used, jam clusters, ranked hypotheses (type, cell, time window, evidence list, vehicle-hours recovered with interval, confidence derived from deterministic scoring rules in config), recommended action text built from templates, responsible-party field from config, status, and a simulated flag. Hypotheses are always labelled hypotheses, never declared causes.
Gate tests: schema validation; ranking is a pure function of its inputs; evidence references resolve to real observations; the simulated flag propagates; no field is filled from ground truth.

## Module 11: API
FastAPI service with async rehearsal jobs: submit a situation request (corridor, time window, scenario or data file reference), poll job status, fetch the result. Config-driven host, port and limits. Approval status and audit entries are stored as fields only, and the existing governance system is out of scope.
Gate tests (TestClient): submit, poll and fetch round trip; invalid requests return typed errors; the same request and seed return an identical result; concurrent jobs do not mix results.

## Module 12: Outcome tracker
After a proposed fix is applied in a second simulated run (the "later observations"), compare predicted recovery with observed recovery and report the error. Feed the confirmed or rejected result into a calibration table that adjusts hypothesis scoring weights per config rules.
Gate tests: with a correct fix, observed recovery falls inside the predicted interval at the rate the tolerance file expects across seeded runs; the calibration update is bounded and reversible; every output is labelled simulated.

## Module 13: Report generator
Static output only (PNG and a single HTML file): space-time speed heatmap with detected jams and the traced origin, hypothesis ranking bar chart with intervals, and RL versus baseline comparison table. Every figure carries the SIMULATED label when applicable.
Gate tests: files are produced; figure data matches the underlying result objects; labels are present.

## Module 14: End-to-end evaluation
Run the full scenario suite (every registered scenario type, K seeds each, K from config). Produce reports/final_results.md with a table: detection recall and precision, origin-localisation error, capacity-ratio error, correct top-ranked cause rate, and RL versus baselines. Include failure cases and limitations. docs/README.md covers setup, config guide, how to add a new scenario in YAML, and how to run each module's tests.
Gate tests: the full suite is reproducible from a clean checkout, using only commands in the README; all earlier tests still pass.

# FINISH
When Module 14 passes, print a summary: what works, what failed, measured numbers only, and the three biggest risks for the demo.