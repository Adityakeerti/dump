"""Double DQN agent with shared weights across corridor junctions."""

import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from citysense.config.schemas import RLConfig


class QNetwork(nn.Module):
    """Shared MLP Q-network taking per-junction observations."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for off-policy Double DQN updates."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DoubleDQNAgent:
    """Double DQN controller with online and target networks."""

    def __init__(self, config: RLConfig, input_dim: int) -> None:
        self.config = config
        self.input_dim = input_dim
        self.action_dim = config.action_space_size

        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        self.online_net = QNetwork(input_dim, config.hidden_dim, self.action_dim)
        self.target_net = QNetwork(input_dim, config.hidden_dim, self.action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(config.buffer_capacity)

        self.total_steps = 0
        self.epsilon = config.epsilon_start

    def select_action(self, state: np.ndarray, epsilon: float | None = None) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        if self.rng.uniform(0.0, 1.0) < eps:
            return int(self.rng.choice(self.action_dim))

        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.online_net(s_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> float:
        if len(self.buffer) < self.config.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        # Online Q(s, a)
        q_eval = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: action selected by online net, evaluated by target net
            next_actions = torch.argmax(self.online_net(next_states), dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + (1.0 - dones) * self.config.gamma * q_next

        loss = self.loss_fn(q_eval, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        return float(loss.item())

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        config_hash: str,
        seed: int,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        out_p = Path(checkpoint_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state": self.online_net.state_dict(),
            "config_hash": config_hash,
            "seed": seed,
            "total_steps": self.total_steps,
            "metadata": metadata or {},
        }
        torch.save(ckpt, out_p)
        return out_p

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        ckpt = torch.load(checkpoint_path, weights_only=False)
        self.online_net.load_state_dict(ckpt["model_state"])
        self.target_net.load_state_dict(ckpt["model_state"])
        self.total_steps = ckpt.get("total_steps", 0)
        return ckpt
