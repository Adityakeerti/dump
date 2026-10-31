# Architectural Decisions Record

## Decision 1: CTM Simulation Engine Architecture
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**:
  The CitySense platform requires:
  1. Runtime capacity reduction on specific cells over specified time windows (e.g. simulating lane closures, accidents, creeping capacity drops).
  2. Full cell-level logging of density, flow, and speed at every simulation time step.
  3. Snapshot and restore capabilities to run counterfactual ladder evaluations and branched simulations from identical checkpoints.
  4. Strict adherence to triangular fundamental diagram physics, conservation of vehicles, and shockwave propagation speed.

  Introspection of `lightsim` (v0.2.0) revealed:
  - `SimulationEngine` computes flow internally in `step()` and discards `intra_flow` without exposing it to callers.
  - `lightsim` has no mechanism for dynamic per-cell capacity degradation over time windows (only intersection movement start-up lost-time ramp).
  - `SimulationEngine` lacks deep snapshot/restore support for counterfactual branching.

- **Decision**:
  Implement a native, fully vectorized NumPy CTM engine adhering strictly to the Daganzo (1994, 1995) Cell Transmission Model equations behind the `EngineAdapter` abstract interface (Module 2).
  The native engine directly tracks:
  - Per-cell density $k$, flow $q$, and speed $v = q / \max(k, \epsilon)$.
  - Dynamic cell capacity schedules: $Q_{\text{eff}}(c, t) = Q_{\text{nom}}(c) \cdot \alpha(c, t)$.
  - Snapshot and restore of complete engine state (densities, step counts, signal phases, timers, scheduled perturbations, random generator states).
  - Direct integration with corridor networks built from OSM via `osmnx` or synthetic corridor generators.

## Decision 2: Dependency Scope and Hardcoding Compliance
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**:
  Rule 6 restricts dependencies to: `numpy, scipy, scikit-learn, networkx, pydantic, pyyaml, pytest, torch, gymnasium, fastapi, uvicorn, httpx, osmnx, matplotlib, lightsim`.
  Rule 2 forbids any hardcoding outside YAML config and unit definitions in `units.py`.
- **Decision**:
  Only the listed dependencies are permitted. All physical constants, thresholds, time steps, network parameters, and tolerances are loaded from validated YAML configurations via Pydantic schemas.
