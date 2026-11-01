"""Root-cause tracing for a live congestion point, using the repo's deterministic diagnosis pipeline.

Pipeline on the displayed run's own recent sensor data:
  1. JamDetector: DBSCAN over sub-threshold speed observations -> the jam cluster containing the clicked cell
  2. ShockwaveTracer: discharge-front back-tracing -> origin (bottleneck) cell and evidence trail
  3. CapacityEstimator: effective capacity and change points at the origin
  4. Live counterfactual ladder: copies of the live engine, restored from a snapshot taken before the jam,
     replay the same background demand and agent trips under
       - baseline: the policy that was actually running,
       - restore_capacity: capacity restored at the origin (and bottleneck_search_radius_cells downstream),
       - demand_peak_shift: background peak retimed by demand.peak_shift_sec,
       - rl_signal_control: every alternative signal policy (incl. the greedy DQN); the best one is the rung.
     Signals keep being driven by real controllers in every rung (unlike a frozen-phase replay).
  5. build_situation: deterministic hypothesis ranking (vehicle-hours recovered, confidence)
RL evidence (decisions, phase shares at the junctions inside the jam) is attached for explanation. The scenario's
planted ground truth is returned separately under "validation" and is never an input to steps 1-5.
"""

import copy
import dataclasses
import time
from typing import Any

import numpy as np

from citysense.config.units import SECONDS_PER_HOUR
from citysense.counterfactual.types import CounterfactualLadderResult, CounterfactualRungResult
from citysense.diagnosis.capacity import CapacityEstimator
from citysense.diagnosis.clustering import JamDetector
from citysense.diagnosis.shockwave import ShockwaveTracer
from citysense.engine.ctm import NativeCTMEngine
from citysense.rl.baselines import FixedTimeController, LongestQueueController, MaxPressureController
from citysense.rl.env import CorridorSignalEnv
from citysense.situations.builder import build_situation


def _rollout(
    twin: Any,
    snap: dict[str, Any],
    policy: str,
    qnet: Any,
    demand_fn: Any,
    trips: list[Any],
    restore_cells: set[int],
    steps: int,
) -> float:
    """Replays the corridor from a snapshot and returns arterial congestion delay in vehicle-hours."""
    cfg = twin.cfg
    net = twin.net
    engine = NativeCTMEngine(
        config=cfg.engine,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
        lanes=net.lanes,
        cell_length=net.cell_length,
        signal_controlled_cells=net.signal_controlled_cells,
        source_demand_weights=twin.source_weights,
    )
    env = CorridorSignalEnv(engine, net, twin.background, cfg.rl, cfg.engine)
    engine.restore(copy.deepcopy(snap["engine"]))
    if restore_cells:
        engine.capacity_windows = [w for w in engine.capacity_windows if not set(w.cell_ids) & restore_cells]
    engine.set_demand_profile(demand_fn)
    env.shield.states = copy.deepcopy(snap["shield"])
    fixed = FixedTimeController(cfg.network, cfg.rl)
    fixed.timers, fixed.phases = dict(snap["fixed"][0]), dict(snap["fixed"][1])
    longest = LongestQueueController(cfg.rl)
    pressure = MaxPressureController(cfg.rl)
    from citysense.live.twin import policy_actions  # local import: twin imports this module

    kc = cfg.engine.capacity / cfg.engine.free_flow_speed
    art = twin.arterial
    dt_h = cfg.engine.dt / SECONDS_PER_HOUR
    pending = sorted(trips, key=lambda tr: tr.depart_sec)
    k = 0
    total = 0.0
    obs = env._get_obs()
    for _ in range(steps):
        t = engine.time_sec
        while k < len(pending) and pending[k].depart_sec <= t:
            engine.entrance_buffer[twin.sources[pending[k].source_id]["path"][0]] += pending[k].vehicles
            k += 1
        actions, _ = policy_actions(policy, twin, env, obs, fixed, longest, pressure, qnet, 0.0, twin.rng)
        obs, _, _, _ = env.step(actions)
        s = engine.state()
        excess = np.maximum(0.0, s.density[art] - kc) * net.cell_length[art] * net.lanes[art]
        total += float(np.sum(excess)) * dt_h
    return total


def _aggregate(name: str, recs: list[float]) -> CounterfactualRungResult:
    mean = float(np.mean(recs))
    std = float(np.std(recs)) if len(recs) > 1 else 0.0
    half = 2.0 * std / np.sqrt(len(recs)) if recs else 0.0
    return CounterfactualRungResult(name, mean, float(max(0.0, mean - half)), float(mean + half), [float(x) for x in recs], 0)


def diagnose(twin: Any, cell: int, prepared: dict[str, Any]) -> dict[str, Any]:
    """Runs the pipeline for the jam containing `cell` (call from a worker thread with `prepared` copies)."""
    t0 = time.perf_counter()
    cfg = twin.cfg
    diag = cfg.diagnosis
    live = twin.live
    run_name = prepared["policy"]
    now = prepared["now"]
    r = twin.r

    arterial = set(int(c) for c in twin.arterial)
    if cell in arterial:
        lane_cells = arterial
    else:  # the side-road leg containing the cell
        lane_cells = next(set(src["path"]) for src in twin.sources.values() if cell in src["path"])
    obs = [o for o in prepared["observations"] if o.cell_id in lane_cells]
    detector = JamDetector(diag)
    ff = {i: float(twin.net.vf[i]) for i in range(twin.net.n_cells)}
    clusters = detector.cluster_jams(detector.identify_congested_points(obs, ff), cfg.network.min_cell_length)
    if not clusters:
        return {"found": False, "reason": "No congested sensor cluster around this point in the diagnosis window.",
                "cell": cell, "policy": run_name, "sim_time": now}

    def dist(c: Any) -> float:
        inside = min(c.cell_ids) <= cell <= max(c.cell_ids)
        return (0.0 if inside else min(abs(cell - x) for x in c.cell_ids)) - c.end_time_sec / SECONDS_PER_HOUR

    cluster = min(clusters, key=dist)

    # ---- maturity gate: only diagnose jams the twin has confirmed for diagnosis_min_jam_age_sec ----------
    cpl = cfg.network.num_cells_per_link
    links = {c // cpl for c in cluster.cell_ids if c in arterial}
    hot = [prepared["link_hot_since"][k] for k in links if k in prepared["jammed_links"] and k in prepared["link_hot_since"]]
    confirmed_since = min(hot) if hot else None
    age = now - confirmed_since if confirmed_since is not None else 0.0
    if age < live.diagnosis_min_jam_age_sec:
        return {
            "found": False,
            "immature": True,
            "cell": cell,
            "policy": run_name,
            "sim_time": r(now),
            "age_sec": r(age),
            "min_age_sec": live.diagnosis_min_jam_age_sec,
            "retry_after_sec": r(live.diagnosis_min_jam_age_sec - age),
            "reason": (
                f"This jam has been confirmed for {age:.0f} s; root causes are traced once a jam has persisted "
                f"{live.diagnosis_min_jam_age_sec:.0f} s, so its effects are large enough to compare."
                if confirmed_since is not None else "Sensors see slow traffic here, but the twin has not confirmed a jam yet."
            ),
        }
    # The repo's tracer takes each cell's earliest observation as its onset; restrict it to the jam's own time
    # window so onsets are times the cell was part of this jam, not the start of the observation buffer.
    jam_obs = [o for o in obs if cluster.start_time_sec <= o.timestamp_sec <= cluster.end_time_sec]
    traced = ShockwaveTracer(cfg.engine, diag).trace_origin(cluster, jam_obs, cfg.network.min_cell_length)

    # ---- replay setup ----------------------------------------------------------------------------
    start_t = max(prepared["snapshots"][0]["t"], cluster.start_time_sec - live.diagnosis_lead_sec)
    snap = max((s for s in prepared["snapshots"] if s["t"] <= start_t), key=lambda s: s["t"], default=prepared["snapshots"][0])
    # Replay from before the jam through now, then rehearse diagnosis_forward_sec into the future.
    steps = int(min(now - snap["t"] + live.diagnosis_forward_sec, live.diagnosis_horizon_max_sec) / cfg.engine.dt)
    trips = [tr for tr in prepared["trips"] if tr.depart_sec >= snap["t"]]
    radius = diag.bottleneck_search_radius_cells

    # ---- origin candidates (chained bottlenecks) ---------------------------------------------------
    # The tracer always names the jam's downstream-most cell. When a bottleneck's queue merges with another
    # queue downstream (e.g. the next signal's), that cell is the wrong one. Candidates: the traced front, every
    # internal discharge front (a congested cell whose observed downstream neighbour is flowing), and every
    # junction stop line inside the jam. The counterfactual decides: the candidate whose capacity restore
    # removes most delay becomes the origin.
    factor = diag.congestion_speed_factor
    speeds: dict[int, list[float]] = {}
    for o in jam_obs:
        speeds.setdefault(o.cell_id, []).append(o.speed_mps)
    mean_v = {c: float(np.mean(v)) for c, v in speeds.items()}
    slow = {c for c, v in mean_v.items() if v < factor * float(twin.net.vf[c])}
    # Rank fronts by speed contrast (mean speed just downstream minus the cell's): a capacity cut leaves a sharp,
    # persistent front; a red light's front washes out over its green phases. Stop lines are not special-cased.
    fronts: list[tuple[float, int]] = []
    for c in slow:
        nxt = [d for d in mean_v if c < d <= c + radius and d in lane_cells]
        if nxt and all(d not in slow for d in nxt):
            fronts.append((float(np.mean([mean_v[d] for d in nxt])) - mean_v[c], c))
    cands: list[tuple[int, str]] = [(traced.origin_cell, "discharge front (tracer)")]
    cands += [(c, f"discharge front, speed contrast {contrast:.1f} m/s") for contrast, c in sorted(fronts, reverse=True)]
    picked: list[tuple[int, str]] = []
    for c, kind in cands:
        if all(abs(c - p) > radius for p, _ in picked):
            picked.append((c, kind))
    picked = picked[: live.diagnosis_max_origin_candidates]

    def restore_set(c: int) -> set[int]:
        # Symmetric search: a discharge front can sit a cell past the true bottleneck.
        return {x for x in range(c - radius, c + radius + 1) if x in lane_cells}

    # ---- demand interventions -------------------------------------------------------------------------
    retimed = twin.background.retimed()
    span = max(now - snap["t"], cfg.engine.dt)
    before = [tr for tr in prepared["trips"] if tr.depart_sec < snap["t"]]
    thin_scale: dict[int, float] = {}
    for src in {tr.source_id for tr in trips if tr.depart_sec <= now}:
        win = sum(tr.vehicles for tr in trips if tr.source_id == src and tr.depart_sec <= now) / span
        ref_span = max(snap["t"], cfg.engine.dt)
        ref = sum(tr.vehicles for tr in before if tr.source_id == src) / ref_span
        if before and ref > 0.0 and win >= live.diagnosis_surge_ratio * ref:
            thin_scale[src] = ref / win
    thinned = [dataclasses.replace(tr, vehicles=tr.vehicles * thin_scale.get(tr.source_id, 1.0)) if tr.depart_sec <= now else tr
               for tr in trips]

    # ---- live counterfactual ladder ---------------------------------------------------------------
    rng = np.random.default_rng(twin.seed + int(now))
    alt_policies = [p for p in twin.policies if p != run_name]
    rec_restore: dict[int, list[float]] = {c: [] for c, _ in picked}
    rec_retime: list[float] = []
    rec_thin: list[float] = []
    rec_alt: dict[str, list[float]] = {p: [] for p in alt_policies}
    base_vh: list[float] = []
    qnet = prepared["qnet"]
    for _ in range(live.diagnosis_replicates):
        mult = max(0.0, 1.0 + float(rng.normal(0.0, diag.counterfactual_perturbation_std)))
        base_fn = (lambda t, m=mult: twin.background.get_rate(t) * m)
        shift_fn = (lambda t, m=mult, d=retimed: d.get_rate(t) * m)
        vb = _rollout(twin, snap, run_name, qnet, base_fn, trips, set(), steps)
        base_vh.append(vb)
        for c, _ in picked:
            rec_restore[c].append(max(0.0, vb - _rollout(twin, snap, run_name, qnet, base_fn, trips, restore_set(c), steps)))
        rec_retime.append(max(0.0, vb - _rollout(twin, snap, run_name, qnet, shift_fn, trips, set(), steps)))
        rec_thin.append(max(0.0, vb - _rollout(twin, snap, run_name, qnet, base_fn, thinned, set(), steps)) if thin_scale else 0.0)
        for p in alt_policies:
            rec_alt[p].append(max(0.0, vb - _rollout(twin, snap, p, qnet, base_fn, trips, set(), steps)))

    restore_results = {c: _aggregate(diag.top_ranked_name_capacity, v) for c, v in rec_restore.items()}
    best_c = max(restore_results, key=lambda c: restore_results[c].mean_recovery_vhr)
    # Keep the tracer's front unless another candidate is clearly better (beyond its 95% interval).
    if restore_results[best_c].ci_lower_vhr <= restore_results[traced.origin_cell].ci_upper_vhr:
        best_c = traced.origin_cell
    origin = dataclasses.replace(traced, origin_cell=int(best_c))
    capacity = CapacityEstimator(cfg.engine, diag).fit_capacity(origin.origin_cell, obs, (cluster.start_time_sec, now))
    restore = restore_set(best_c)

    demand_results = {"retime_background_peak": _aggregate(diag.top_ranked_name_demand, rec_retime),
                      "thin_trip_surge": _aggregate(diag.top_ranked_name_demand, rec_thin)}
    best_dem_name = max(demand_results, key=lambda k: demand_results[k].mean_recovery_vhr)
    alt_results = {p: _aggregate(p, v) for p, v in rec_alt.items()}
    best_alt = max(alt_results.values(), key=lambda x: x.mean_recovery_vhr) if alt_results else None
    rungs = [restore_results[best_c], demand_results[best_dem_name]]
    if best_alt is not None:
        rungs.append(CounterfactualRungResult(diag.top_ranked_name_signal, best_alt.mean_recovery_vhr,
                                              best_alt.ci_lower_vhr, best_alt.ci_upper_vhr, best_alt.replicate_recoveries, 0))
    rungs.sort(key=lambda x: x.mean_recovery_vhr, reverse=True)
    for i, rung in enumerate(rungs):
        rung.rank = i + 1
    explains = rungs[0].mean_recovery_vhr > diag.counterfactual_noise_floor_vhr
    top_rung = rungs[0]
    second = rungs[1] if len(rungs) > 1 else None
    if top_rung.mean_recovery_vhr < live.diagnosis_min_effect_vhr:
        verdict = {"status": "inconclusive",
                   "reason": f"largest recovery {top_rung.mean_recovery_vhr:.2f} veh-h is below the minimum effect "
                             f"{live.diagnosis_min_effect_vhr:.2f} veh-h"}
    elif second is not None and top_rung.ci_lower_vhr <= second.ci_upper_vhr:
        verdict = {"status": "tied",
                   "reason": f"95% intervals of {top_rung.rung_name} and {second.rung_name} overlap; the ladder cannot separate them"}
    else:
        verdict = {"status": "conclusive", "reason": f"{top_rung.rung_name} is separated from the next rung and above the minimum effect"}
    ladder = CounterfactualLadderResult(
        scenario_name=prepared["scenario"],
        rungs=rungs,
        top_ranked_rung=rungs[0].rung_name if explains else diag.top_ranked_name_control,
        baseline_total_vhr=float(np.mean(base_vh)),
        noise_floor_vhr=diag.counterfactual_noise_floor_vhr,
        explains_jam=explains,
    )
    situation = build_situation(
        situation_id=f"LIVE-{prepared['episode']}-{int(now)}-{origin.origin_cell}",
        corridor_name=cfg.network.corridor_name,
        observations=obs,
        jam_clusters=[cluster],
        shockwave_origin=origin,
        capacity_fit=capacity,
        ladder_result=ladder,
        diag_cfg=diag,
        config_hash=twin.cfg_hash,
    ).to_dict()

    # ---- RL / signal evidence at junctions inside or bounding the jam -----------------------------
    lo, hi = min(cluster.cell_ids), max(cluster.cell_ids)
    w0, w1 = cluster.start_time_sec, now
    jn_evidence = []
    for j, jc in sorted(twin.net.junction_cells.items()):
        a = jc["arterial_approach"]
        if not (lo - twin.cfg.network.num_cells_per_link <= a <= hi + twin.cfg.network.num_cells_per_link):
            continue
        hist = [ph[j] for t, ph in prepared["phase_history"] if w0 <= t <= w1]
        decs = [d for d in prepared["decisions"] if d["junction"] == j and w0 <= d["sim_time"] <= w1]
        modes: dict[str, int] = {}
        for d in decs:
            modes[d["mode"]] = modes.get(d["mode"], 0) + 1
        margins = [d["q"][cfg.rl.action_switch] - d["q"][cfg.rl.action_keep] for d in decs if d.get("q")]
        jn_evidence.append({
            "junction": f"J{j + 1}",
            "arterial_approach_cell": int(a),
            "position": "inside jam" if lo <= a <= hi else ("downstream of jam" if a > hi else "upstream of jam"),
            "arterial_green_share": r(sum(1 for p in hist if p == 0) / len(hist)) if hist else None,
            "decisions_logged": len(decs),
            "switches": sum(1 for d in decs if d["switched"]),
            "mode_counts": modes,
            "shield_overrides": sum(1 for d in decs if d["shield"]),
            "mean_q_margin_switch_minus_keep": r(float(np.mean(margins))) if margins else None,
            "recent_decisions": [{"id": d["id"], "t": r(d["sim_time"]), "text": d["text"]} for d in decs[-live.diagnosis_evidence_limit // 2:]],
        })

    # ---- planted ground truth (validation only; not an input above) -------------------------------
    gt_windows = [{"cells": w.cell_ids, "factor": w.factor, "window": [w.start_time_sec, w.end_time_sec]}
                  for w in prepared["capacity_windows"]]
    gt_cells = [c for w in gt_windows for c in w["cells"]]
    sc = cfg.scenarios[prepared["scenario"]]
    capacity_types = {diag.cause_name_recurring, diag.cause_name_creeping, diag.cause_name_spillback}
    expected = (diag.cause_name_recurring if sc.scenario_type in capacity_types
                else diag.cause_name_demand_shift if sc.scenario_type == diag.cause_name_demand_shift
                else diag.top_ranked_name_control)
    top = situation["top_hypothesis"] or {}

    return {
        "found": True,
        "simulated": True,
        "cell": cell,
        "policy": run_name,
        "episode": prepared["episode"],
        "sim_time": r(now),
        "jam": {**cluster.to_dict(), "cells": sorted(set(int(c) for c in cluster.cell_ids)),
                "observation_count": len(cluster.cell_ids), "age_sec": r(age), "confirmed_since_sec": r(confirmed_since),
                "mean_speed_mps": r(cluster.mean_speed_mps), "extent_m": r(cluster.extent_m)},
        "origin": {
            "cell": origin.origin_cell,
            "earliest_onset_sec": r(origin.earliest_time_sec),
            "wave_speed_mps": r(origin.estimated_wave_speed_mps),
            # The repo's estimator silently returns the configured w when its estimate is out of tolerance.
            "wave_speed_is_fallback": origin.estimated_wave_speed_mps == cfg.engine.wave_speed,
            "theoretical_wave_speed_mps": cfg.engine.wave_speed,
            "evidence_trail": origin.evidence_trail[: live.diagnosis_evidence_limit],
            "method": ("downstream-most cell of the jam's discharge front (Rankine-Hugoniot back-tracing)"
                       if origin.origin_cell == traced.origin_cell else
                       "candidate bottleneck whose counterfactual capacity restore removed the most delay "
                       f"(the traced front, cell {traced.origin_cell}, removed less)"),
            "traced_front_cell": traced.origin_cell,
            "candidates": [
                {"cell": int(c), "kind": kind, **restore_results[c].to_dict()} for c, kind in picked
            ],
        },
        "capacity_at_origin": capacity.to_dict(),
        "counterfactual": {
            **ladder.to_dict(),
            "verdict": verdict,
            "min_effect_vhr": live.diagnosis_min_effect_vhr,
            "snapshot_sec": r(snap["t"]),
            "horizon_sec": r(steps * cfg.engine.dt),
            "replicates": live.diagnosis_replicates,
            "running_policy": run_name,
            "restored_cells": sorted(restore),
            "demand_alternatives": [
                {"intervention": k, **v.to_dict()} for k, v in sorted(demand_results.items(), key=lambda kv: kv[1].mean_recovery_vhr, reverse=True)
            ],
            "best_demand_intervention": best_dem_name,
            "surge_sources_thinned": {str(k): r(v) for k, v in thin_scale.items()},
            "signal_alternatives": [{"policy": p, **res.to_dict()} for p, res in
                                    sorted(alt_results.items(), key=lambda kv: kv[1].mean_recovery_vhr, reverse=True)],
            "best_signal_alternative": best_alt.rung_name if best_alt else None,
            "assumptions": [
                "background demand x N(1, perturbation_std) per replicate; agent trips replayed at their departure times",
                "demand rung = best of: retime the background peak, or thin trip departures at entries whose rate in the replay window exceeded diagnosis_surge_ratio x their earlier rate",
                "capacity restore and the candidate search act on the twin's own capacity state; a field twin would use calibrated (estimated) capacities, so localisation here is an upper bound",
                "the DQN rung is evaluated greedily (no exploration) with a frozen copy of the online network",
                "restore_capacity clears capacity reductions within bottleneck_search_radius_cells on both sides of the origin",
                "rungs are different-sized interventions; vehicle-hours recovered ranks relief, not blame",
                f"replay starts before the jam and rehearses {live.diagnosis_forward_sec:.0f} s past now; future agent trips are unknown",
                "95% intervals reflect demand-perturbation replicates only, not sensor or model uncertainty",
            ],
        },
        "situation": situation,
        "rl_evidence": jn_evidence,
        "validation": {
            "note": "Planted scenario ground truth. Hidden from the diagnosis pipeline and from the LLM; shown for evaluation.",
            "scenario": prepared["scenario"],
            "scenario_type": sc.scenario_type,
            "planted_windows": gt_windows,
            "origin_error_cells": min((abs(origin.origin_cell - c) for c in gt_cells), default=None),
            "expected_top_hypothesis": expected,
            "top_hypothesis": top.get("hypothesis_type"),
            "top_matches": top.get("hypothesis_type") == expected,
        },
        "runtime_sec": r(time.perf_counter() - t0),
    }
