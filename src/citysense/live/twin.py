"""Live digital twin: one CTM run per signal policy in lockstep, online DQN learning, car-cluster tracers.

Every policy run sees the same background demand, the same planted scenario and the same agent trips,
so the per-episode comparison between the learning DQN and the classical controllers is like-for-like.
Car clusters (platoons) are Lagrangian tracers moving through each run's CTM speed field; they wait at
red signals and record which decision released them, which links individual trips to policy decisions.
"""

import copy
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from citysense.config.schemas import CitySenseConfig
from citysense.config.units import HALF_FACTOR, SECONDS_PER_MINUTE
from citysense.engine.ctm import NativeCTMEngine
from citysense.live.config import LiveConfig
from citysense.live.explain import FEATURE_NAMES, PHASE_LABELS, attribute, describe, q_values
from citysense.live.geometry import build_geometry, source_paths
from citysense.network.corridor import build_corridor_network, source_demand_weights
from citysense.network.demand import CorridorDemand
from citysense.observation.generator import ObservationGenerator
from citysense.observation.types import ObservationRecord
from citysense.rl.baselines import FixedTimeController, LongestQueueController, MaxPressureController
from citysense.rl.dqn import DoubleDQNAgent
from citysense.rl.env import CorridorSignalEnv
from citysense.scenarios.registry import create_scenario

POLICY_FIXED = "fixed-time"
POLICY_LONGEST = "longest-queue"
POLICY_PRESSURE = "max-pressure"

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"


class ScaledDemand:
    """Background demand: the configured profile scaled by a constant factor."""

    def __init__(self, demand: CorridorDemand, factor: float) -> None:
        self.demand = demand
        self.factor = factor
        self.config = demand.config

    def get_rate(self, time_sec: float) -> float:
        return self.demand.get_rate(time_sec) * self.factor

    def as_callable(self) -> Callable[[float], float]:
        return self.get_rate

    def retimed(self) -> "ScaledDemand":
        """Same profile with the peak window moved later by demand.peak_shift_sec (warm-up unchanged)."""
        d = self.demand
        shifted = CorridorDemand(d.config, is_weekend=d.is_weekend, peak_shift_sec=d.peak_shift_sec + d.config.peak_shift_sec)
        return ScaledDemand(shifted, self.factor)


@dataclass
class Trip:
    """A car cluster requested by a people-cluster agent (shared by all policy runs)."""

    trip_id: int
    cluster_id: str
    cluster_name: str
    intent: str
    plan_id: str
    source_id: int
    vehicles: float
    depart_sec: float


@dataclass
class Platoon:
    """Per-run tracer for one trip."""

    trip: Trip
    path: list[int]
    remaining_ahead: float
    status: str = "waiting"  # waiting -> moving -> done
    idx: int = 0
    offset_m: float = 0.0
    enter_sec: float | None = None
    exit_sec: float | None = None
    hold_since: float | None = None
    free_flow_sec: float = 0.0
    passes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EpisodeAcc:
    steps: int = 0
    system_veh_sum: float = 0.0
    system_veh_max: float = 0.0
    jam_cell_sec: float = 0.0
    switches: int = 0
    reward_sum: float = 0.0
    loss_sum: float = 0.0
    loss_n: int = 0
    explore_n: int = 0
    explore_effective_n: int = 0
    action_n: int = 0
    shield_n: int = 0
    platoons_done: int = 0
    delay_sum: float = 0.0
    wait_sum: float = 0.0


def policy_actions(
    name: str,
    twin: "LiveTwin",
    env: CorridorSignalEnv,
    obs: np.ndarray,
    fixed: FixedTimeController,
    longest: LongestQueueController,
    pressure: MaxPressureController,
    qnet: Any,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Proposed action per junction for one policy (shared by live runs and counterfactual rollouts)."""
    rl = twin.cfg.rl
    s = env.engine.state()
    num_j = env.num_junctions
    is_dqn = name == rl.algorithm
    qv = q_values(qnet, obs) if is_dqn else None
    actions: dict[int, int] = {}
    meta: list[dict[str, Any]] = []
    for j in range(num_j):
        jc = twin.net.junction_cells[j]
        art_q = float(s.queue[jc["arterial_approach"]])
        cross_q = float(sum(s.queue[c] for c in jc["cross_approaches"]))
        phase = env.shield.states[j].current_phase
        m: dict[str, Any] = {"art_q": art_q, "cross_q": cross_q, "mode": "rule", "greedy": None}
        if is_dqn:
            greedy = int(np.argmax(qv[j]))
            if epsilon > 0.0 and rng.uniform(0.0, 1.0) < epsilon:
                act = int(rng.choice(rl.action_space_size))
                m["mode"] = "explore"
            else:
                act = greedy
                m["mode"] = "exploit"
            m.update(greedy=greedy, q=[float(v) for v in qv[j]], epsilon=epsilon)
        elif name == POLICY_FIXED:
            act = fixed.act(j, twin.cfg.engine.dt)
            m["rule"] = f"fixed half-cycle of {fixed.half_cycle:g}s"
        elif name == POLICY_LONGEST:
            act = longest.act(j, phase, art_q, cross_q)
            m["rule"] = f"longest queue: arterial {art_q:.1f} vs cross {cross_q:.1f} veh"
        else:
            ds_q = 0.0
            if j < num_j - 1:
                ds_q = float(s.queue[twin.net.junction_cells[j + 1]["arterial_approach"]])
            act = pressure.act(j, phase, art_q, ds_q, cross_q)
            m["rule"] = f"pressure: arterial {art_q - ds_q:.1f} vs cross {cross_q:.1f}"
        actions[j] = act
        meta.append(m)
    return actions, meta


class PolicyRun:
    """One CTM engine + signal shield driven by a single policy."""

    def __init__(self, name: str, twin: "LiveTwin") -> None:
        self.name = name
        self.twin = twin
        cfg = twin.cfg
        net = twin.net
        self.engine = NativeCTMEngine(
            config=cfg.engine,
            n_cells=net.n_cells,
            upstream_cells=net.upstream_cells,
            downstream_cells=net.downstream_cells,
            lanes=net.lanes,
            cell_length=net.cell_length,
            signal_controlled_cells=net.signal_controlled_cells,
            source_demand_weights=twin.source_weights,
        )
        self.env = CorridorSignalEnv(self.engine, net, twin.background, cfg.rl, cfg.engine)
        self.fixed = FixedTimeController(cfg.network, cfg.rl)
        self.longest = LongestQueueController(cfg.rl)
        self.pressure = MaxPressureController(cfg.rl)
        self.decisions: deque[dict[str, Any]] = deque(maxlen=twin.live.decision_log_size)
        self.history: deque[dict[str, Any]] = deque(maxlen=twin.live.history_size)
        self.platoons: list[Platoon] = []
        self.finished: deque[dict[str, Any]] = deque(maxlen=twin.live.decision_log_size)
        self.last_decision: dict[int, int | None] = {}
        self.recent_explore: deque[int] = deque(maxlen=twin.live.recent_window_steps)
        self.last_loss = 0.0
        self.acc = EpisodeAcc()
        self.obs = np.zeros((0, 0), dtype=np.float32)
        # Diagnosis inputs: noisy sensor observations, periodic engine snapshots, signal-phase history.
        window_steps = int(twin.live.diagnosis_window_sec / cfg.engine.dt)
        report_steps = max(1, int(round(cfg.observation.reporting_interval_sec / cfg.engine.dt)))
        self.sensors = ObservationGenerator(cfg.observation, net.n_cells, twin.seed)
        self.observations: deque[ObservationRecord] = deque(maxlen=net.n_cells * (window_steps // report_steps + 1))
        self.snapshots: deque[dict[str, Any]] = deque(
            maxlen=window_steps // twin.live.snapshot_every_steps + 1
        )
        self.phase_history: deque[tuple[float, tuple[int, ...]]] = deque(maxlen=window_steps)

    def reset(self, seed: int) -> None:
        self.obs = self.env.reset(seed=seed)
        self.twin.scenario.apply_to_engine(self.engine, self.twin.net)
        self.fixed.reset(self.env.num_junctions)
        self.history.clear()
        self.platoons.clear()
        self.last_decision = {j: None for j in range(self.env.num_junctions)}
        self.acc = EpisodeAcc()
        self.observations.clear()
        self.snapshots.clear()
        self.phase_history.clear()
        self._take_snapshot()

    def _take_snapshot(self) -> None:
        self.snapshots.append(
            {
                "t": float(self.engine.time_sec),
                "engine": self.engine.snapshot(),
                "shield": copy.deepcopy(self.env.shield.states),
                "fixed": (dict(self.fixed.timers), dict(self.fixed.phases)),
            }
        )

    # ---- decisions -----------------------------------------------------------------------------
    def choose(self) -> tuple[dict[int, int], list[dict[str, Any]]]:
        twin = self.twin
        eps = twin.epsilon if twin.learning else 0.0
        return policy_actions(
            self.name, twin, self.env, self.obs, self.fixed, self.longest, self.pressure,
            twin.agent.online_net, eps, twin.rng,
        )

    def step(self, record: bool) -> None:
        twin = self.twin
        rl = twin.cfg.rl
        dt = twin.cfg.engine.dt
        num_j = self.env.num_junctions
        t_prev = self.engine.time_sec
        buf_prev = self.engine.entrance_buffer.copy()
        rate = self.engine.demand_fn(t_prev) if self.engine.demand_fn is not None else 0.0
        prev_phase = {j: self.env.shield.states[j].current_phase for j in range(num_j)}
        actions, meta = self.choose()
        shielded = {j: self.env.shield.filter_action(j, actions[j]) for j in range(num_j)}
        tip = {j: self.env.shield.states[j].time_in_phase for j in range(num_j)}
        prev_obs = self.obs

        next_obs, reward, _, _ = self.env.step(actions)
        state = self.engine.state()
        self.obs = next_obs
        acc = self.acc
        acc.steps += 1
        acc.reward_sum += reward

        if self.name == rl.algorithm:
            explored = 0
            for j in range(num_j):
                acc.action_n += 1
                if meta[j]["mode"] == "explore":
                    acc.explore_n += 1
                    explored += 1
                if twin.learning:
                    twin.agent.buffer.push(
                        prev_obs[j], actions[j], reward / rl.queue_norm_factor, next_obs[j], False
                    )
            self.recent_explore.append(explored)
            if twin.learning:
                loss = twin.agent.update()
                if len(twin.agent.buffer) >= rl.batch_size:
                    self.last_loss = loss
                    acc.loss_sum += loss
                    acc.loss_n += 1

        if self.name == rl.algorithm:
            # Effective exploration: a random action that differed from the greedy one and survived the shield.
            for j in range(num_j):
                m = meta[j]
                if m["mode"] == "explore" and actions[j] != m["greedy"] and shielded[j] == actions[j]:
                    acc.explore_effective_n += 1

        for j in range(num_j):
            phase = self.env.shield.states[j].current_phase
            switched = phase != prev_phase[j]
            shield = None
            if shielded[j] != actions[j]:
                shield = "min_green" if tip[j] < rl.shield_min_green_sec else "max_cycle"
                acc.shield_n += 1
            if switched:
                acc.switches += 1
            m = meta[j]
            changed_by_explore = m["mode"] == "explore" and actions[j] != m["greedy"]
            if record and (switched or shield is not None or changed_by_explore):
                self._record_decision(j, actions[j], prev_phase[j], phase, switched, shield, m, prev_obs[j])

        # system-level accounting
        system_veh = float(np.sum(state.queue)) + float(state.queued_demand)
        acc.system_veh_sum += system_veh
        acc.system_veh_max = max(acc.system_veh_max, system_veh)
        speed_ratio = state.speed / self.engine.vf
        congested = speed_ratio < twin.cfg.diagnosis.congestion_speed_factor
        acc.jam_cell_sec += float(np.sum(congested[twin.arterial])) * dt

        admitted = buf_prev + rate * dt - self.engine.entrance_buffer
        self._move_platoons(state, admitted)

        self.observations.extend(self.sensors.observe_step(state, dt))  # flow noise: observation.flow_noise_rel
        self.sensors.records.clear()  # the generator keeps every record otherwise (unbounded in a live twin)
        self.phase_history.append(
            (float(state.time_sec), tuple(self.env.shield.states[j].current_phase for j in range(num_j)))
        )
        if self.engine.step_count % twin.live.snapshot_every_steps == 0:
            self._take_snapshot()

        if self.engine.step_count % twin.live.history_sample_every_steps == 0:
            self.history.append(
                {"t": float(state.time_sec), "v": [twin.r(x) for x in speed_ratio[twin.arterial]]}
            )

    def _record_decision(
        self,
        j: int,
        action: int,
        prev_phase: int,
        phase: int,
        switched: bool,
        shield: str | None,
        m: dict[str, Any],
        obs_row: np.ndarray,
    ) -> None:
        twin = self.twin
        rec: dict[str, Any] = {
            "id": twin.next_decision_id(),
            "policy": self.name,
            "episode": twin.episode,
            "sim_time": float(self.engine.time_sec),
            "junction": j,
            "proposed": int(action),
            "greedy": m.get("greedy"),
            "prev_phase": int(prev_phase),
            "executed_phase": int(phase),
            "switched": bool(switched),
            "shield": shield,
            "mode": m["mode"],
            "epsilon": m.get("epsilon"),
            "q": m.get("q"),
            "rule": m.get("rule"),
            "art_queue": twin.r(m["art_q"]),
            "cross_queue": twin.r(m["cross_q"]),
            "features": {FEATURE_NAMES[i]: twin.r(float(obs_row[i])) for i in range(len(FEATURE_NAMES))},
            "attribution": None,
            "ts": time.time(),
        }
        if self.name == twin.cfg.rl.algorithm:
            rec["attribution"] = [
                {k: (twin.r(v) if isinstance(v, float) else v) for k, v in a.items()}
                for a in attribute(twin.agent.online_net, obs_row, twin.cfg.rl)
            ]
        rec["text"] = describe(rec, twin.cfg.rl)
        self.decisions.append(rec)
        if switched:
            self.last_decision[j] = rec["id"]

    # ---- car clusters ---------------------------------------------------------------------------
    def add_trip(self, trip: Trip) -> None:
        twin = self.twin
        path = twin.sources[trip.source_id]["path"]
        src = path[0]
        ahead = float(self.engine.entrance_buffer[src])
        self.engine.entrance_buffer[src] += trip.vehicles
        if len(self.platoons) >= twin.live.max_active_platoons:
            return
        free_flow = float(sum(twin.net.cell_length[c] / twin.net.vf[c] for c in path))
        self.platoons.append(
            Platoon(
                trip=trip,
                path=path,
                remaining_ahead=ahead + trip.vehicles * HALF_FACTOR,
                free_flow_sec=free_flow,
            )
        )

    def _move_platoons(self, state: Any, admitted: np.ndarray) -> None:
        twin = self.twin
        dt = twin.cfg.engine.dt
        t = float(state.time_sec)
        still: list[Platoon] = []
        for p in self.platoons:
            if p.status == "waiting":
                p.remaining_ahead -= float(admitted[p.path[0]])
                if p.remaining_ahead <= 0.0:
                    p.status = "moving"
                    p.enter_sec = t
                still.append(p)
                continue
            cell = p.path[p.idx]
            move = float(state.speed[cell]) * dt
            length = float(twin.net.cell_length[cell])
            if p.offset_m + move < length:
                p.offset_m += move
                still.append(p)
                continue
            jid = twin.signal_junction(cell)
            if jid is not None and self.env.shield.states[jid].current_phase != twin.net.signal_controlled_cells[cell][1]:
                p.offset_m = length
                if p.hold_since is None:
                    p.hold_since = t
                still.append(p)
                continue
            if jid is not None:
                wait = t - p.hold_since if p.hold_since is not None else 0.0
                p.passes.append(
                    {
                        "junction": jid,
                        "sim_time": t,
                        "wait_sec": twin.r(wait),
                        "released_by": self.last_decision.get(jid),
                        "policy": self.name,
                    }
                )
                self.acc.wait_sum += wait
                p.hold_since = None
            p.offset_m = p.offset_m + move - length
            p.idx += 1
            if p.idx >= len(p.path):
                p.status = "done"
                p.exit_sec = t
                travel = t - p.trip.depart_sec
                delay = max(0.0, travel - p.free_flow_sec)
                self.acc.platoons_done += 1
                self.acc.delay_sum += delay
                self.finished.append({**twin.platoon_view(p, done=True), "seq": twin.next_finish_seq()})
                continue
            p.offset_m = min(p.offset_m, float(twin.net.cell_length[p.path[p.idx]]))
            still.append(p)
        self.platoons = still

    def kpis(self) -> dict[str, Any]:
        r = self.twin.r
        acc = self.acc
        s = self.engine.state()
        steps = max(acc.steps, 1)
        return {
            "system_vehicles": r(float(np.sum(s.queue)) + float(s.queued_demand)),
            "in_network": r(float(np.sum(s.queue))),
            "backlog": r(float(s.queued_demand)),
            "throughput": r(float(s.total_exited)),
            "mean_system_vehicles": r(acc.system_veh_sum / steps),
            "max_system_vehicles": r(acc.system_veh_max),
            "jam_cell_min": r(acc.jam_cell_sec / SECONDS_PER_MINUTE),
            "switches": acc.switches,
            "shield_overrides": acc.shield_n,
            "platoons_done": acc.platoons_done,
            "mean_delay_sec": r(acc.delay_sum / acc.platoons_done) if acc.platoons_done else None,
            "mean_reward": r(acc.reward_sum / steps),
            "explore_frac": r(acc.explore_n / acc.action_n) if acc.action_n else None,
            "explore_effective_frac": r(acc.explore_effective_n / acc.action_n) if acc.action_n else None,
            "mean_loss": r(acc.loss_sum / acc.loss_n) if acc.loss_n else None,
        }


class LiveTwin:
    """Owns the network, the policy runs, the learning agent, the event log and episode history."""

    def __init__(self, cfg: CitySenseConfig, cfg_hash: str, live: LiveConfig) -> None:
        self.cfg = cfg
        self.cfg_hash = cfg_hash
        self.live = live
        self.seed = cfg.project.seed
        self.rng = np.random.default_rng(self.seed)
        self.net = build_corridor_network(cfg.network, seed=self.seed)
        self.background = ScaledDemand(CorridorDemand(cfg.demand), live.background_demand_factor)
        self.source_weights = source_demand_weights(self.net, cfg.demand.cross_street_flow_factor)
        self.arterial = np.array(self.net.arterial_cells, dtype=np.int64)
        self.sources = source_paths(self.net, cfg.network.num_cells_per_link)
        self.geometry = build_geometry(self.net, cfg.network, live)
        self._signal_junction = {c: jid for c, (jid, _) in self.net.signal_controlled_cells.items()}

        if live.default_scenario not in cfg.scenarios:
            raise ValueError(f"Unknown default scenario {live.default_scenario}")
        self.scenario_name = live.default_scenario
        self.scenario = create_scenario(cfg.scenarios[self.scenario_name])

        self.policies = [cfg.rl.algorithm] + list(cfg.rl.baseline_algorithms)
        if live.default_policy not in self.policies:
            raise ValueError(f"Unknown default policy {live.default_policy}")
        self.active_policy = live.default_policy

        num_j = cfg.network.num_signalized_junctions
        probe = CorridorSignalEnv(
            NativeCTMEngine(cfg.engine, self.net.n_cells, self.net.upstream_cells, self.net.downstream_cells),
            self.net,
            self.background,
            cfg.rl,
            cfg.engine,
        )
        self.agent = DoubleDQNAgent(cfg.rl, probe.obs_dim_per_junction)
        self.num_junctions = num_j
        self.learning = live.learn_online_default
        self.episodes_done = 0
        self.episodes: deque[dict[str, Any]] = deque(maxlen=live.episode_history_size)
        self._load_progress()

        self.status = STATUS_RUNNING if live.autoplay else STATUS_PAUSED
        self.steps_per_tick = live.default_steps_per_tick
        self.events: deque[dict[str, Any]] = deque(maxlen=live.event_log_size)
        self._event_id = 0
        self._decision_id = 0
        self._finish_seq = 0
        self._trip_id = 0
        self.pending: list[Trip] = []
        self.trips_this_episode = 0
        self.episode_started_wall = time.time()
        self.runs = {name: PolicyRun(name, self) for name in self.policies}
        self._jammed_links: set[int] = set()
        self._link_hot_since: dict[int, float] = {}
        self._link_quiet_since: dict[int, float] = {}
        self._active_windows: set[int] = set()
        self._reset_runs()
        self.emit("system", "twin_started", f"Twin online: {self.scenario_name}, policy {self.active_policy}")

    # ---- helpers --------------------------------------------------------------------------------
    def r(self, x: float) -> float:
        return round(float(x), self.live.precision_digits)

    @property
    def episode(self) -> int:
        return self.episodes_done

    @property
    def epsilon(self) -> float:
        rl = self.cfg.rl
        return max(rl.epsilon_end, rl.epsilon_start * self.live.epsilon_episode_decay**self.episodes_done)

    def signal_junction(self, cell: int) -> int | None:
        """Junction whose signal gates leaving this cell (the engine gates transfers and side-road exits)."""
        return self._signal_junction.get(cell)

    def next_finish_seq(self) -> int:
        self._finish_seq += 1
        return self._finish_seq

    def next_decision_id(self) -> int:
        self._decision_id += 1
        return self._decision_id

    def emit(self, category: str, kind: str, text: str, **data: Any) -> dict[str, Any]:
        self._event_id += 1
        engine = self.runs[self.active_policy].engine if hasattr(self, "runs") else None
        ev = {
            "id": self._event_id,
            "ts": time.time(),
            "sim_time": float(engine.time_sec) if engine is not None else 0.0,
            "episode": self.episodes_done,
            "category": category,
            "kind": kind,
            "text": text,
            **data,
        }
        self.events.append(ev)
        return ev

    # ---- persistence ----------------------------------------------------------------------------
    def _load_progress(self) -> None:
        ckpt = Path(self.live.checkpoint_path)
        if ckpt.is_file():
            meta = self.agent.load_checkpoint(ckpt)
            if meta.get("config_hash") == self.cfg_hash:
                self.episodes_done = int(meta.get("metadata", {}).get("episodes_done", 0))
            else:
                self.agent = DoubleDQNAgent(self.cfg.rl, self.agent.input_dim)
        ep_path = Path(self.live.episodes_path)
        if ep_path.is_file() and self.episodes_done:
            with open(ep_path, "r", encoding="utf-8") as f:
                self.episodes.extend(json.load(f))

    def _save_progress(self) -> None:
        self.agent.save_checkpoint(
            self.live.checkpoint_path,
            self.cfg_hash,
            self.seed,
            metadata={"episodes_done": self.episodes_done},
        )
        ep_path = Path(self.live.episodes_path)
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump(list(self.episodes), f)

    # ---- lifecycle ------------------------------------------------------------------------------
    def _reset_runs(self) -> None:
        seed = self.seed + self.episodes_done * self.live.episode_seed_increment
        sc_cfg = self.cfg.scenarios[self.scenario_name]
        self.scenario = create_scenario(sc_cfg)
        # The repo applies demand_peak_shift scenarios through the demand profile (peak_shift_sec), not the engine.
        self.background = ScaledDemand(
            CorridorDemand(self.cfg.demand, peak_shift_sec=sc_cfg.demand_shift_sec), self.live.background_demand_factor
        )
        for run in self.runs.values():
            run.env.demand = self.background
            run.reset(seed)
        self.pending.clear()
        self.trip_log: list[Trip] = []
        self.trips_this_episode = 0
        self._jammed_links.clear()
        self._link_hot_since.clear()
        self._link_quiet_since.clear()
        self._active_windows.clear()
        self.episode_started_wall = time.time()

    def _finish_episode(self) -> None:
        summary = {
            "episode": self.episodes_done,
            "scenario": self.scenario_name,
            "epsilon": self.r(self.epsilon),
            "learning": self.learning,
            "agent_trips": self.trips_this_episode,
            "wall_sec": self.r(time.time() - self.episode_started_wall),
            "policies": {name: run.kpis() for name, run in self.runs.items()},
        }
        self.episodes.append(summary)
        dqn = summary["policies"][self.cfg.rl.algorithm]["mean_system_vehicles"]
        best_name, best = min(
            ((n, k["mean_system_vehicles"]) for n, k in summary["policies"].items() if n != self.cfg.rl.algorithm),
            key=lambda x: x[1],
        )
        self.emit(
            "rl",
            "episode_complete",
            f"Episode {self.episodes_done + 1} done: DQN mean {dqn:.1f} veh in system vs best baseline "
            f"{best_name} {best:.1f} (epsilon {self.epsilon:.2f})",
            summary=summary,
        )
        self.episodes_done += 1
        self._save_progress()
        self._reset_runs()
        self.emit("rl", "episode_start", f"Episode {self.episodes_done + 1} started (epsilon {self.epsilon:.2f})")

    def advance(self, steps: int) -> None:
        duration = self.cfg.demand.simulation_duration_sec
        for _ in range(steps):
            t = self.runs[self.active_policy].engine.time_sec
            due = [tr for tr in self.pending if tr.depart_sec <= t]
            if due:
                self.pending = [tr for tr in self.pending if tr.depart_sec > t]
                for tr in due:
                    for run in self.runs.values():
                        run.add_trip(tr)
            for name, run in self.runs.items():
                run.step(record=True)
            self._detect_events()
            if self.runs[self.active_policy].engine.time_sec >= duration:
                self._finish_episode()

    def _detect_events(self) -> None:
        run = self.runs[self.active_policy]
        s = run.engine.state()
        cpl = self.cfg.network.num_cells_per_link
        ratio = s.speed / run.engine.vf
        factor = self.cfg.diagnosis.congestion_speed_factor
        t = float(s.time_sec)
        for link in range(len(self.arterial) // cpl):
            cells = self.arterial[link * cpl : (link + 1) * cpl]
            n = int(np.sum(ratio[cells] < factor))
            hot = n >= self.live.jam_min_cells
            if hot:
                self._link_quiet_since.pop(link, None)
                since = self._link_hot_since.setdefault(link, t)
                if link not in self._jammed_links and t - since >= self.live.jam_confirm_sec:
                    self._jammed_links.add(link)
                    self.emit(
                        "congestion",
                        "jam_onset",
                        f"Congestion on {self.link_name(link)}: {n} of {cpl} cells below "
                        f"{factor:g} x free-flow speed for {t - since:.0f}s",
                        link=link,
                        cells_congested=n,
                        min_speed_ratio=self.r(float(np.min(ratio[cells]))),
                    )
            else:
                self._link_hot_since.pop(link, None)
                quiet = self._link_quiet_since.setdefault(link, t)
                if link in self._jammed_links and t - quiet >= self.live.jam_clear_sec:
                    self._jammed_links.discard(link)
                    self.emit("congestion", "jam_cleared", f"Congestion cleared on {self.link_name(link)}", link=link)

        active = {
            i for i, w in enumerate(run.engine.capacity_windows) if w.start_time_sec <= s.time_sec < w.end_time_sec
        }
        for i in sorted(active - self._active_windows):
            w = run.engine.capacity_windows[i]
            self.emit(
                "disruption",
                "disruption_start",
                f"{self.scenario_name}: capacity on cell(s) {w.cell_ids} cut to {w.factor:.0%}",
                cells=w.cell_ids,
                factor=w.factor,
            )
        for i in sorted(self._active_windows - active):
            w = run.engine.capacity_windows[i]
            self.emit("disruption", "disruption_end", f"Capacity restored on cell(s) {w.cell_ids}", cells=w.cell_ids)
        self._active_windows = active

    def link_name(self, link: int) -> str:
        last = self.num_junctions
        start = "Gateway" if link == 0 else f"J{link}"
        end = "Exit" if link == last else f"J{link + 1}"
        return f"link {start}→{end}"

    # ---- external inputs --------------------------------------------------------------------------
    def control(
        self,
        action: str | None,
        policy: str | None,
        scenario: str | None,
        steps_per_tick: int | None,
        learning: bool | None,
    ) -> None:
        if policy is not None and policy != self.active_policy:
            if policy not in self.policies:
                raise ValueError(f"Unknown policy {policy}")
            self.active_policy = policy
            self._jammed_links.clear()
            self._link_hot_since.clear()
            self._link_quiet_since.clear()
            self.emit("control", "policy_changed", f"Map now shows the {policy} run")
        if scenario is not None and scenario != self.scenario_name:
            if scenario not in self.cfg.scenarios:
                raise ValueError(f"Unknown scenario {scenario}")
            self.scenario_name = scenario
            self._reset_runs()
            self.emit("control", "scenario_changed", f"Scenario set to {scenario}; episode restarted")
        if steps_per_tick is not None:
            if steps_per_tick not in self.live.speed_options:
                raise ValueError(f"steps_per_tick must be one of {self.live.speed_options}")
            self.steps_per_tick = steps_per_tick
        if learning is not None and learning != self.learning:
            self.learning = learning
            self.emit(
                "rl",
                "learning_toggled",
                "Online learning on (epsilon-greedy exploration)" if learning else "Learning off: greedy policy only",
            )
        if action == "play":
            self.status = STATUS_RUNNING
        elif action == "pause":
            self.status = STATUS_PAUSED
        elif action == "step":
            self.advance(self.steps_per_tick)
        elif action == "reset":
            self._reset_runs()
            self.emit("control", "episode_reset", "Episode restarted by operator")

    def add_trips(self, trips: list[dict[str, Any]]) -> list[int]:
        if len(trips) > self.live.max_trips_per_request:
            raise ValueError(f"At most {self.live.max_trips_per_request} trips per request")
        now = self.runs[self.active_policy].engine.time_sec
        ids: list[int] = []
        by_cluster: dict[str, list[Trip]] = {}
        for t in trips:
            if t["source_id"] not in self.sources:
                raise ValueError(f"Unknown source {t['source_id']}")
            if not 0.0 < t["vehicles"] <= self.live.max_vehicles_per_trip:
                raise ValueError(f"vehicles must be in (0, {self.live.max_vehicles_per_trip}]")
            if not 0.0 <= t["depart_in_sec"] <= self.live.max_trip_delay_sec:
                raise ValueError(f"depart_in_sec must be in [0, {self.live.max_trip_delay_sec}]")
            self._trip_id += 1
            trip = Trip(
                trip_id=self._trip_id,
                cluster_id=t["cluster_id"],
                cluster_name=t["cluster_name"],
                intent=t["intent"],
                plan_id=t["plan_id"],
                source_id=t["source_id"],
                vehicles=float(t["vehicles"]),
                depart_sec=now + float(t["depart_in_sec"]),
            )
            self.pending.append(trip)
            self.trip_log.append(trip)
            by_cluster.setdefault(trip.cluster_name, []).append(trip)
            ids.append(trip.trip_id)
        self.trips_this_episode += len(ids)
        for name, group in by_cluster.items():
            veh = sum(tr.vehicles for tr in group)
            src = self.sources[group[0].source_id]["name"]
            self.emit(
                "population",
                "trips_planned",
                f"{name}: {len(group)} car clusters ({veh:.0f} veh, {group[0].intent}) from {src}",
                cluster_id=group[0].cluster_id,
                plan_id=group[0].plan_id,
            )
        return ids

    # ---- views ----------------------------------------------------------------------------------
    def platoon_view(self, p: Platoon, done: bool = False) -> dict[str, Any]:
        cell = p.path[min(p.idx, len(p.path) - 1)]
        frac = 1.0 if done else p.offset_m / float(self.net.cell_length[cell])
        travel = (p.exit_sec if p.exit_sec is not None else None)
        return {
            "id": p.trip.trip_id,
            "cluster_id": p.trip.cluster_id,
            "cluster": p.trip.cluster_name,
            "plan_id": p.trip.plan_id,
            "intent": p.trip.intent,
            "vehicles": self.r(p.trip.vehicles),
            "source_id": p.trip.source_id,
            "status": p.status,
            "held": p.hold_since is not None,
            "cell": int(cell),
            "frac": self.r(frac),
            "depart": self.r(p.trip.depart_sec),
            "travel_sec": self.r(travel - p.trip.depart_sec) if travel is not None else None,
            "delay_sec": self.r(max(0.0, travel - p.trip.depart_sec - p.free_flow_sec)) if travel is not None else None,
            "passes": p.passes,
        }

    def topology(self) -> dict[str, Any]:
        sc = self.cfg.scenarios
        return {
            "corridor_name": self.cfg.network.corridor_name,
            "config_hash": self.cfg_hash,
            "simulated": True,
            "dt": self.cfg.engine.dt,
            "episode_duration_sec": self.cfg.demand.simulation_duration_sec,
            "congestion_speed_factor": self.cfg.diagnosis.congestion_speed_factor,
            "num_junctions": self.num_junctions,
            "cells_per_link": self.cfg.network.num_cells_per_link,
            "arterial_cells": [int(c) for c in self.arterial],
            "policies": self.policies,
            "learning_policy": self.cfg.rl.algorithm,
            "speed_options": self.live.speed_options,
            "diagnosis_min_jam_age_sec": self.live.diagnosis_min_jam_age_sec,
            "tick_interval_sec": self.live.tick_interval_sec,
            "feature_names": FEATURE_NAMES,
            "phase_labels": {str(k): v for k, v in PHASE_LABELS.items()},
            "shield": {
                "min_green_sec": self.cfg.rl.shield_min_green_sec,
                "max_phase_sec": self.cfg.rl.shield_max_cycle_sec,
            },
            "scenarios": [
                {
                    "name": n,
                    "type": s.scenario_type,
                    "description": s.description,
                    "window": [s.start_time_sec, s.end_time_sec],
                    "target_cell": s.target_cell_id,
                    "capacity_factor": s.capacity_factor,
                }
                for n, s in sc.items()
            ],
            "geometry": self.geometry,
        }

    def learning_view(self) -> dict[str, Any]:
        run = self.runs[self.cfg.rl.algorithm]
        window = max(len(run.recent_explore), 1) * self.num_junctions
        return {
            "enabled": self.learning,
            "epsilon": self.r(self.epsilon if self.learning else 0.0),
            "epsilon_schedule": [
                self.r(max(self.cfg.rl.epsilon_end, self.cfg.rl.epsilon_start * self.live.epsilon_episode_decay**e))
                for e in range(self.episodes_done, self.episodes_done + self.live.epsilon_preview_episodes)
            ],
            "recent_explore_rate": self.r(sum(run.recent_explore) / window),
            "updates": self.agent.total_steps,
            "replay_size": len(self.agent.buffer),
            "last_loss": self.r(run.last_loss),
            "episodes_done": self.episodes_done,
        }

    def state(self) -> dict[str, Any]:
        run = self.runs[self.active_policy]
        s = run.engine.state()
        eng = run.engine
        cpl = self.cfg.network.num_cells_per_link
        ratio = s.speed / eng.vf
        factor = self.cfg.diagnosis.congestion_speed_factor
        links = []
        for link in range(len(self.arterial) // cpl):
            cells = self.arterial[link * cpl : (link + 1) * cpl]
            links.append(
                {
                    "id": link,
                    "name": self.link_name(link),
                    "mean_speed_ratio": self.r(float(np.mean(ratio[cells]))),
                    "jammed": link in self._jammed_links,
                    "cells_congested": int(np.sum(ratio[cells] < factor)),
                    "vehicles": self.r(float(np.sum(s.queue[cells]))),
                }
            )
        junctions = []
        for j in range(self.num_junctions):
            jc = self.net.junction_cells[j]
            sh = run.env.shield.states[j]
            junctions.append(
                {
                    "id": j,
                    "phase": sh.current_phase,
                    "time_in_phase": self.r(sh.time_in_phase),
                    "art_queue": self.r(float(s.queue[jc["arterial_approach"]])),
                    "cross_queue": self.r(float(sum(s.queue[c] for c in jc["cross_approaches"]))),
                    "cross_leg_queues": [self.r(float(s.queue[c])) for c in jc["cross_approaches"]],
                    "last_decision": run.last_decision.get(j),
                }
            )
        disruptions = [
            {"cells": w.cell_ids, "factor": w.factor, "window": [w.start_time_sec, w.end_time_sec]}
            for w in eng.capacity_windows
            if w.start_time_sec <= s.time_sec < w.end_time_sec
        ]
        return {
            "status": self.status,
            "sim_time": self.r(s.time_sec),
            "episode": self.episodes_done,
            "episode_duration_sec": self.cfg.demand.simulation_duration_sec,
            "scenario": self.scenario_name,
            "policy": self.active_policy,
            "steps_per_tick": self.steps_per_tick,
            "background_rate": self.r(self.background.get_rate(s.time_sec)),
            "cells": {
                "speed_ratio": [self.r(x) for x in ratio],
                "density_ratio": [self.r(x) for x in s.density / eng.kj],
                "vehicles": [self.r(x) for x in s.queue],
            },
            "links": links,
            "junctions": junctions,
            "disruptions": disruptions,
            "congestion_points": self.congestion_points(ratio, float(s.time_sec)),
            "source_backlog": {str(k): self.r(float(eng.entrance_buffer[v["path"][0]])) for k, v in self.sources.items()},
            "platoons": [self.platoon_view(p) for p in run.platoons],
            "pending_trips": len(self.pending),
            "kpis": {name: r.kpis() for name, r in self.runs.items()},
            "learning": self.learning_view(),
        }

    def prepare_diagnosis(self) -> dict[str, Any]:
        """Copies everything a root-cause trace needs, so it can run in a worker thread."""
        run = self.runs[self.active_policy]
        return {
            "policy": self.active_policy,
            "now": float(run.engine.time_sec),
            "episode": self.episodes_done,
            "scenario": self.scenario_name,
            "observations": list(run.observations),
            "snapshots": list(run.snapshots),
            "phase_history": list(run.phase_history),
            "decisions": list(run.decisions),
            "trips": list(self.trip_log),
            "qnet": copy.deepcopy(self.agent.online_net),
            "capacity_windows": copy.deepcopy(run.engine.capacity_windows),
            "link_hot_since": dict(self._link_hot_since),
            "jammed_links": set(self._jammed_links),
        }

    def congestion_points(self, ratio: np.ndarray, t: float) -> list[dict[str, Any]]:
        """Confirmed jammed links with their slowest stretch: the clickable congestion points."""
        cpl = self.cfg.network.num_cells_per_link
        factor = self.cfg.diagnosis.congestion_speed_factor
        points = []
        for link in sorted(self._jammed_links):
            cells = [int(c) for c in self.arterial[link * cpl : (link + 1) * cpl] if ratio[c] < factor]
            if not cells:
                continue
            points.append(
                {
                    "link": link,
                    "name": self.link_name(link),
                    "cells": cells,
                    "head_cell": max(cells),
                    "min_speed_ratio": self.r(float(min(ratio[c] for c in cells))),
                    "since_sec": self.r(self._link_hot_since.get(link, t)),
                }
            )
        return points

    def decisions(self, policy: str | None, after: int, limit: int) -> list[dict[str, Any]]:
        run = self.runs[policy or self.active_policy]
        return [d for d in run.decisions if d["id"] > after][-limit:]

    def history(self) -> list[dict[str, Any]]:
        return list(self.runs[self.active_policy].history)

    def finished_platoons(self, limit: int) -> list[dict[str, Any]]:
        return list(self.runs[self.active_policy].finished)[-limit:]
