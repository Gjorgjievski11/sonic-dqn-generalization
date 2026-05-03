import os
import torch
import numpy as np

from sonic_dqn.envs.sonic_env import SonicEnv
from sonic_dqn.agents.dqn import DQN


def evaluate(cfg: dict, checkpoint_path: str | None = None) -> None:
    env_cfg = cfg["env"]
    log_cfg = cfg["logging"]

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(log_cfg["checkpoint_dir"], "dqn_final.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Evaluating checkpoint: {checkpoint_path}")
    print(f"  Device : {device}")
    print(f"{'='*50}\n")

    ckpt = torch.load(checkpoint_path, map_location=device)
    trained_at_step = ckpt.get("step", "unknown")
    print(f"  Checkpoint trained at step: {trained_at_step}\n")

    env = SonicEnv(env_cfg)
    in_channels = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = DQN(in_channels, n_actions).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_episodes = cfg.get("evaluation", {}).get("n_episodes", 5)
    episode_rewards = []
    episode_lengths = []

    print(f"{'Episode':>8}  {'Reward':>10}  {'Steps':>8}  {'Max X':>8}")
    print("-" * 45)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        max_x = 0

        while True:
            with torch.no_grad():
                t = torch.tensor(obs[np.newaxis], device=device)
                action = model(t).argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # stable-retro exposes x position in info
            x_pos = info.get("x", info.get("xpos", 0))
            if x_pos > max_x:
                max_x = x_pos

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"{ep:>8}  {total_reward:>10.1f}  {steps:>8}  {max_x:>8}")

    print("-" * 45)
    print(f"  Mean reward : {np.mean(episode_rewards):.1f}")
    print(f"  Mean steps  : {np.mean(episode_lengths):.0f}")
    print(f"  Best reward : {max(episode_rewards):.1f}")
    print(f"\n{'='*50}\n")

    env.close()