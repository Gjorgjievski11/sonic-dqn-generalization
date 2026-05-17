import numpy as np
import stable_retro
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from collections import deque
import cv2


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeats the chosen action for 'skip' frames.
    Sums the reward and returns the final observation.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        
        for _ in range(self.skip):
            step_result = self.env.step(action)
            
            # Handle both older Gym (4 returns) and newer Gymnasium (5 returns)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
            else:
                obs, reward, terminated, truncated, info = step_result
                
            total_reward += reward
            
            if terminated or truncated:
                break
                
        # Always return 5 items to match Gymnasium standards
        return obs, total_reward, terminated, truncated, info


class StuckTimeoutWrapper(gym.Wrapper):
    """
    Ends the episode early if Sonic's x-coordinate hasn't 
    changed for a specified number of steps.
    """
    def __init__(self, env, max_stuck_steps=450):
        super().__init__(env)
        self.max_stuck_steps = max_stuck_steps
        self.current_stuck_steps = 0
        self.last_x = None

    def step(self, action):
        step_result = self.env.step(action)
        
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result
            
        current_x = info.get('x', None)

        if current_x is not None:
            if current_x == self.last_x:
                self.current_stuck_steps += 1
            else:
                self.current_stuck_steps = 0
                self.last_x = current_x

        if self.current_stuck_steps >= self.max_stuck_steps:
            truncated = True  # Force termination via truncation

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_stuck_steps = 0
        self.last_x = None
        return self.env.reset(**kwargs)


class SonicEnv(gym.Env):
    """
    Wraps stable-retro's Sonic environment with:
    - Grayscale + resize to (obs_height, obs_width)
    - Frame stacking
    - Discrete action space (only meaningful button combos)
    - Reward clipping to [-1, 1]
    - Frame Skipping & Stuck Timeout
    """

    # Only the actions that actually matter in Sonic
    ACTIONS = [
        [],               # no-op (do nothing)
        ["LEFT"],         # walk/run left
        ["RIGHT"],        # walk/run right
        ["LEFT", "B"],    # jump left
        ["RIGHT", "B"],   # jump right
        ["B"],            # jump straight up
        ["DOWN"],         # crouch (standing) / roll (running)
    ]

    def __init__(self, cfg: dict):
        super().__init__()
        self.game = cfg["game"]
        self.state = cfg["state"]
        self.width = cfg["obs_width"]
        self.height = cfg["obs_height"]
        self.grayscale = cfg["grayscale"]
        self.n_stack = cfg["frame_stack"]

        # 1. Base Environment
        self._env = stable_retro.make(
            game=self.game,
            state=self.state,
            render_mode="rgb_array",
        )

        # 2. Frame Skip (Reduces 60 FPS to 15 decisions per second)
        self._env = FrameSkipWrapper(self._env, skip=4)

        # 3. Stuck Timeout (30 seconds * 15 steps/sec = 450 steps)
        self._env = StuckTimeoutWrapper(self._env, max_stuck_steps=450)

        # 4. Record Video (Now records the much shorter, skipped episodes!)
        self._env = RecordVideo(
            self._env, 
            video_folder="/app/runs/videos",
            name_prefix="sonic-dqn",
            episode_trigger=lambda episode_id: episode_id % 3 == 0 # Records every 3th episode
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
        # We only need the first return (obs) from reset, and maybe info, 
        # handling both 1 or 2 return values from underlying envs
        reset_result = self._env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
            
        frame = self._process_frame(obs)
        for _ in range(self.n_stack):
            self._frames.append(frame)
        return self._get_obs(), info

    def step(self, action: int):
        act = self._action_array[action]
        
        # We know our wrappers now return exactly 5 items
        obs, reward, terminated, truncated, info = self._env.step(act)
        
        frame = self._process_frame(obs)
        self._frames.append(frame)
        reward = float(np.clip(reward, -1.0, 1.0))
        
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self._env.close()