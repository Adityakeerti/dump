# Engine Reconnaissance: LightSim

Date: 2026-10-31
Package: lightsim v0.2.0

## 1. Overview and Public Interface
LightSim is a macroscopic traffic simulator implementing the Cell Transmission Model (CTM) in pure Python / NumPy with Gymnasium and PettingZoo environment interfaces.

### Public Classes
- `lightsim.core.network.Network`: Builder class for road networks consisting of nodes, links (discretized into cells), movements, and signal phases.
- `lightsim.core.network.CompiledNetwork`: Immutable dataclass containing vectorized arrays (`n_cells`, `lanes`, `length`, `vf`, `w`, `kj`, `Q`, `mov_from_cell`, `mov_to_cell`, etc.) compiled from a `Network` for simulation execution.
- `lightsim.core.engine.SimulationEngine`: The core stepping loop holding simulation state, signals, demand, and flow model.
- `lightsim.core.engine.SimState`: Dataclass tracking simulation state (`density`, `time`, `step_count`, `total_entered`, `total_exited`).
- `lightsim.core.flow_model.CTMFlowModel`: Vectorized triangular fundamental diagram flow computer (sending flow, receiving flow, merge, diverge).
- `lightsim.core.demand.DemandProfile`: Specifies traffic demand injection at origins over time intervals.
- `lightsim.envs.single_agent.LightSimEnv`: Gymnasium environment wrapper.
- Signal Controllers:
  - `FixedTimeController`
  - `MaxPressureController`
  - `WebsterController`
  - `GreenWaveController`
  - `SOTLController`
  - `RLController`

### Public Functions
- `lightsim.make(scenario_name, ...)`: Creates Gymnasium environment.
- `lightsim.parallel_env(scenario_name, ...)`: Creates PettingZoo multi-agent environment.
- `lightsim.networks.osm.from_osm(query=..., point=..., ...)`: Loads OpenStreetMap graph via `osmnx` and converts into a `Network`.
- `lightsim.benchmarks.scenarios.list_scenarios()`: Lists built-in scenario names.
- `lightsim.benchmarks.scenarios.get_scenario(name)`: Returns a factory returning `(Network, list[DemandProfile])`.

---

## 2. OpenStreetMap Loading
OpenStreetMap graphs are imported using `lightsim.networks.osm.from_osm()`:
- Input: geocoding query string or bounding box `(north, south, east, west)` or `point=(lat, lon)` with `dist` radius.
- `osmnx.graph_from_point` / `graph_from_bbox` fetches the road network.
- Coordinates are projected to UTM meters.
- Boundary nodes with degree 1 or entering/exiting links are identified as origins (`NodeType.ORIGIN`) and destinations (`NodeType.DESTINATION`).
- Interior nodes with degree >= `signal_degree_threshold` (default 4) are marked `NodeType.SIGNALIZED` with multi-phase signal plans generated.
- Links are discretized into cells subject to the CFL condition: `cell_length >= vf * dt`.

---

## 3. Demand Generation
Demand in LightSim is represented by `DemandProfile(node_id, intervals=[(start_time, end_time, flow_veh_per_sec)])`.
- `DemandManager` aggregates active profiles at simulation time `t`.
- Vehicles are injected into origin cells at each step: `injection = demand_rate * dt`.
- Boundary demand can also be generated across origins using `lightsim.networks.osm.generate_demand(net, total_vehicles, peak_time, duration)`.

---

## 4. Signal and Controller Definitions
- Signals are managed per signalized node by `SignalManager`.
- Each phase references allowed `MovementID`s.
- `SignalController` implementations provide an `act(step, time, net, states, density)` method returning next phase choices per node.
- `SignalManager` enforces transition lost time (yellow + all-red intervals) and computes a binary movement mask `signal_mask` (0 = red, 1 = green).
- A green-ramp factor is computed during start-up lost time for movement flows.

---

## 5. Per-Cell Density and Flow Reading
- **Density**: Readily available at `engine.state.density`, which is a 1D NumPy float array of shape `(n_cells,)` holding vehicle density (veh/m/lane) for each cell in `CompiledNetwork`.
- **Flow**: In LightSim's `SimulationEngine.step()`, `intra_flow` and `movement_flow` are computed locally inside `step()` and applied to update density, but are **not** persisted or exposed on `SimState` or `SimulationEngine`. To access per-cell flow, the flow must either be intercepted during the step or computed post-hoc.

---

## 6. Runtime Capacity Editing
- In LightSim, cell capacity $Q$ (veh/s/lane) is stored statically in `net.Q` inside `CompiledNetwork`.
- `SignalManager.get_capacity_factor()` only applies a temporary startup lost-time factor to *movements* at signalized intersections; it does **not** support runtime capacity editing of specific road cells (e.g. lane closures, work zones, or bottlenecks over time windows).
- `SimulationEngine` lacks methods to schedule or dynamically apply capacity factors per cell.

---

## 7. Built-in Scenario Execution
A built-in scenario (e.g. `single-intersection-v0`, `arterial-5-v0`, `grid-4x4-v0`) can be obtained via `factory = get_scenario('single-intersection-v0')`, yielding `(net, demand_profiles)`. Stepping `SimulationEngine(net, dt, demand_profiles=dem)` advances vehicle counts cleanly and stably.
