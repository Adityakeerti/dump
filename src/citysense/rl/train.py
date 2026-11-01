"""Minimal Double DQN training loop on the corridor signal environment (shared weights across junctions)."""

from citysense.rl.dqn import DoubleDQNAgent
from citysense.rl.env import CorridorSignalEnv


def train_dqn(env: CorridorSignalEnv, agent: DoubleDQNAgent, steps: int, seed: int) -> list[float]:
    """Runs `steps` environment steps with epsilon-greedy actions; returns the per-update TD losses."""
    obs = env.reset(seed=seed)
    losses: list[float] = []
    for _ in range(steps):
        actions = {j: agent.select_action(obs[j]) for j in range(env.num_junctions)}
        nxt, reward, _, _ = env.step(actions)
        scaled = reward / env.rl_cfg.queue_norm_factor
        for j in range(env.num_junctions):
            agent.buffer.push(obs[j], actions[j], scaled, nxt[j], False)
        losses.append(agent.update())
        obs = nxt
    return losses
