"""Comparative evaluation pipeline benchmarking RL against classical baseline signal controllers."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from citysense.config.schemas import EngineConfig, NetworkConfig, RLConfig
from citysense.engine.adapter import EngineAdapter
from citysense.network.corridor import CorridorNetwork
from citysense.network.demand import CorridorDemand
from citysense.rl.baselines import (
    FixedTimeController,
    LongestQueueController,
    MaxPressureController,
)
from citysense.rl.dqn import DoubleDQNAgent
from citysense.rl.env import CorridorSignalEnv


@dataclass
class PolicyEvaluationResult:
    policy_name: str
    mean_queue: float
    ci_lower: float
    ci_upper: float
    episodes: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "mean_queue": round(self.mean_queue, 2),
            "ci_lower": round(self.ci_lower, 2),
            "ci_upper": round(self.ci_upper, 2),
        }


def evaluate_policy_suite(
    env: CorridorSignalEnv,
    rl_agent: DoubleDQNAgent,
    net_config: NetworkConfig,
    rl_config: RLConfig,
    eval_episodes: int,
    eval_steps: int,
    seed: int,
) -> dict[str, PolicyEvaluationResult]:
    """Evaluates RL policy and classical baselines on identical seeded episodes."""
    results: dict[str, PolicyEvaluationResult] = {}
    policies = ["DoubleDQN", "fixed-time", "longest-queue", "max-pressure"]
    num_j = env.num_junctions

    fixed_ctrl = FixedTimeController(net_config, rl_config)
    longest_ctrl = LongestQueueController(rl_config)
    pressure_ctrl = MaxPressureController(rl_config)

    for pol in policies:
        episode_queues: list[float] = []

        for ep in range(eval_episodes):
            ep_seed = seed + ep
            obs = env.reset(seed=ep_seed)
            fixed_ctrl.reset(num_j)
            total_ep_queue = 0.0

            for _ in range(eval_steps):
                actions: dict[int, int] = {}
                s = env.engine.state()

                for j in range(num_j):
                    j_cells = env.network.junction_cells[j]
                    art_q = float(s.queue[j_cells["arterial_approach"]])
                    cross_q = float(sum(s.queue[c] for c in j_cells.get("cross_approaches", [j_cells["cross_approach"]])))
                    curr_phase = env.shield.states[j].current_phase

                    if pol == "DoubleDQN":
                        actions[j] = rl_agent.select_action(obs[j], epsilon=0.0)
                    elif pol == "fixed-time":
                        actions[j] = fixed_ctrl.act(j, env.eng_cfg.dt)
                    elif pol == "longest-queue":
                        actions[j] = longest_ctrl.act(j, curr_phase, art_q, cross_q)
                    elif pol == "max-pressure":
                        ds_q = 0.0
                        if j < num_j - 1:
                            next_c = env.network.junction_cells[j + 1]["arterial_approach"]
                            ds_q = float(s.queue[next_c])
                        actions[j] = pressure_ctrl.act(j, curr_phase, art_q, ds_q, cross_q)

                obs, reward, done, info = env.step(actions)
                total_ep_queue += float(np.sum(s.queue))

            mean_step_queue = total_ep_queue / max(1, eval_steps)
            episode_queues.append(mean_step_queue)

        mean_val = float(np.mean(episode_queues))
        std_val = float(np.std(episode_queues)) if len(episode_queues) > 1 else 0.0
        ci_half = 2.0 * (std_val / np.sqrt(len(episode_queues)))

        results[pol] = PolicyEvaluationResult(
            policy_name=pol,
            mean_queue=mean_val,
            ci_lower=max(0.0, mean_val - ci_half),
            ci_upper=mean_val + ci_half,
            episodes=episode_queues,
        )

    return results
