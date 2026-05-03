import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random


class DQN(nn.Module):
    """
    Standard 3-layer CNN DQN (similar to the original Atari DQN paper).
    Input: (batch, C*n_stack, H, W) uint8 — normalized to [0,1] inside forward()
    Output: (batch, n_actions) Q-values
    """

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float() / 255.0)


class ReplayBuffer:
    """
    Simple circular replay buffer storing (obs, action, reward, next_obs, done).
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.uint8),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.uint8),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)