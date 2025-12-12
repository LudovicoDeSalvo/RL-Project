import math
import random
from collections import namedtuple
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done", "distance_to_end"],
)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[Optional[Transition]] = [None] * capacity
        self.distances = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.missing_distance_seen = False
        self.warned_missing_distance = False

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        distance_to_end: Optional[int] = None,
    ) -> None:
        if distance_to_end is None:
            self.missing_distance_seen = True
        dist_val = float(distance_to_end) if distance_to_end is not None else 100.0
        self.memory[self.position] = Transition(
            state, action, reward, next_state, done, distance_to_end
        )
        self.distances[self.position] = dist_val
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, transitions: Iterable[Tuple[np.ndarray, int, float, np.ndarray, bool]], use_distance: bool) -> None:
        # Compute distance_to_end per step so ReMERT can prioritize tail transitions.
        episode: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = list(transitions)
        total = len(episode)
        for idx, (state, action, reward, next_state, done) in enumerate(episode):
            distance = total - idx if use_distance else None
            self.push(state, action, reward, next_state, done, distance)

    def sample(self, batch_size: int, weighted: bool, device: torch.device) -> Tuple[torch.Tensor, ...]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        actual_batch = min(batch_size, self.size)
        replace = self.size < batch_size

        if weighted:
            # Weighted sampling (ReMERT) prioritizes transitions near episode end.
            if self.missing_distance_seen and not self.warned_missing_distance:
                warnings.warn(
                    "ReMERT weighted sampling requested but some transitions are missing "
                    "distance_to_end; falling back to weight 1.0 for those samples."
                )
                self.warned_missing_distance = True

            valid_distances = self.distances[: self.size]
            weights = 1.0 / (valid_distances + 1.0)
            probabilities = weights / weights.sum()
            indices = np.random.choice(self.size, size=actual_batch, replace=replace, p=probabilities)
        else:
            indices = np.random.choice(self.size, size=actual_batch, replace=replace)

        sampled = [self.memory[idx] for idx in indices]
        batch = Transition(*zip(*sampled))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,
        target_update_freq: int = 200,
        hidden_sizes: Sequence[int] = (128, 128),
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.policy_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0
        self.update_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )
        self.steps_done += 1

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_tensor).argmax(dim=1).item())

    def update(self, buffer: ReplayBuffer, batch_size: int, weighted: bool = False) -> float:
        if len(buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = buffer.sample(batch_size, weighted, self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.sync_target_net()

        return float(loss.item())

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
