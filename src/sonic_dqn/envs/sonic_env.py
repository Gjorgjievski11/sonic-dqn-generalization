import numpy as np
import stable_retro
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import cv2


class SonicEnv(gym.Env):
    """
    Wraps stable-retro's Sonic environment with:
    - Grayscale + resize to (obs_height, obs_width)
    - Frame stacking
    - Discrete action space (only meaningful button combos)
    - Reward clipping to [-1, 1]
    """

    # Only the actions that actually matter in Sonic
    ACTIONS = [
        [],           # no-op
        ["LEFT"],
        ["RIGHT"],
        ["RIGHT", "B"],   # run right
        ["RIGHT", "A"],   # jump right
        ["B"],            # jump in place
        ["A"],            # also jump
        ["DOWN"],         # curl/roll
    ]

    def __init__(self, cfg: dict):
        super().__init__()
        self.game = cfg["game"]
        self.state = cfg["state"]
        self.width = cfg["obs_width"]
        self.height = cfg["obs_height"]
        self.grayscale = cfg["grayscale"]
        self.n_stack = cfg["frame_stack"]

        self._env = stable_retro.make(
            game=self.game,
            state=self.state,
            render_mode=None,
        )

        # Build button-index action array
        self._buttons = self._env.unwrapped.buttons
        self._action_array = self._build_actions()

        self.action_space = spaces.Discrete(len(self._action_array))

        channels = 1 if self.grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.n_stack * channels, self.height, self.width),
            dtype=np.uint8,
        )

        self._frames: deque = deque(maxlen=self.n_stack)

    def _build_actions(self) -> list:
        actions = []
        for combo in self.ACTIONS:
            arr = [False] * len(self._buttons)
            for btn in combo:
                if btn in self._buttons:
                    arr[self._buttons.index(btn)] = True
            actions.append(arr)
        return actions

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = frame[np.newaxis, ...]  # (1, H, W)
        else:
            frame = np.transpose(frame, (2, 0, 1))  # (3, H, W)
        return frame

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)  # (C*n_stack, H, W)

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset()
        frame = self._process_frame(obs)
        for _ in range(self.n_stack):
            self._frames.append(frame)
        return self._get_obs(), info

    def step(self, action: int):
        act = self._action_array[action]
        obs, reward, terminated, truncated, info = self._env.step(act)
        frame = self._process_frame(obs)
        self._frames.append(frame)
        reward = float(np.clip(reward, -1.0, 1.0))
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self._env.close()