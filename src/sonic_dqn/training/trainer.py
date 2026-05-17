import os
import time
import numpy as np
import wandb

import torch
import torch.nn.functional as F

from sonic_dqn.envs.sonic_env import SonicEnv
from sonic_dqn.agents.dqn import DQN, ReplayBuffer


def train(cfg: dict) -> None:
    env_cfg = cfg["env"]
    tr = cfg["training"]
    log_cfg = cfg["logging"]
    seed = cfg["experiment"]["seed"]

    os.makedirs(log_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(log_cfg["tensorboard_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Experiment : {cfg['experiment']['name']}")
    print(f"  Device     : {device}")
    print(f"  Timesteps  : {tr['total_timesteps']:,}")
    print(f"{'='*50}\n")

    env = SonicEnv(env_cfg)
    obs, _ = env.reset(seed=seed)

    in_channels = obs.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(in_channels, n_actions).to(device)
    target_net = DQN(in_channels, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=tr["learning_rate"])
    buffer = ReplayBuffer(tr["replay_buffer_size"])

    epsilon = tr["epsilon_start"]
    eps_start = tr["epsilon_start"]
    eps_end = tr["epsilon_end"]
    eps_decay = tr["epsilon_decay_steps"]

    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0
    episode_rewards = []
    start_time = time.time()
    last_log_step = 0

    print(f"{'Step':>8}  {'Episode':>8}  {'Ep Reward':>10}  {'Avg(10)':>8}  {'Epsilon':>8}  {'Steps/s':>8}")
    print("-" * 65)

    for step in range(1, tr["total_timesteps"] + 1):
        # Epsilon-greedy action
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                t = torch.tensor(obs[np.newaxis], device=device)
                action = policy_net(t).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, float(done))

        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        if done:
            episode_num += 1
            episode_rewards.append(episode_reward)
            avg10 = np.mean(episode_rewards[-10:])

            wandb.log({
                "episode/reward": episode_reward,
                "episode/length": episode_steps,
                "episode/epsilon": epsilon,
                "episode/avg_10_reward": avg10
            }, step=step)


            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        # Train
        if step >= tr["learning_starts"] and step % tr["train_freq"] == 0 and len(buffer) >= tr["batch_size"]:
            obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(tr["batch_size"])

            obs_t = torch.tensor(obs_b, device=device)
            act_t = torch.tensor(act_b, device=device)
            rew_t = torch.tensor(rew_b, device=device)
            next_obs_t = torch.tensor(next_obs_b, device=device)
            done_t = torch.tensor(done_b, device=device)

            q_vals = policy_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(next_obs_t).max(1).values
                target = rew_t + tr["gamma"] * next_q * (1 - done_t)

            loss = F.smooth_l1_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()

            wandb.log({
                "train/loss": loss.item(),
                "train/q_mean": q_vals.mean().item()
            }, step=step)

        # Update target network
        if step % tr["target_update_freq"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Human-readable log
        if step % log_cfg["log_freq"] == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step - last_log_step) / max(elapsed - (last_log_step / max(step, 1) * elapsed), 1e-6)
            steps_per_sec = step / elapsed
            avg10 = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            last_reward = episode_rewards[-1] if episode_rewards else 0.0
            print(
                f"{step:>8,}  {episode_num:>8,}  {last_reward:>10.1f}  "
                f"{avg10:>8.1f}  {epsilon:>8.3f}  {steps_per_sec:>8.1f}"
            )
            last_log_step = step

        # Checkpoint
        if step % log_cfg["checkpoint_freq"] == 0:
            ckpt_path = os.path.join(log_cfg["checkpoint_dir"], f"dqn_step{step}.pt")
            torch.save({
                "step": step,
                "model_state": policy_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epsilon": epsilon,
            }, ckpt_path)
            print(f"\n  >> Checkpoint saved: {ckpt_path}\n")

    # Final checkpoint
    ckpt_path = os.path.join(log_cfg["checkpoint_dir"], "dqn_final.pt")
    torch.save({"step": tr["total_timesteps"], "model_state": policy_net.state_dict()}, ckpt_path)
    print(f"\n{'='*50}")
    print(f"  Training complete!")
    print(f"  Final checkpoint: {ckpt_path}")
    print(f"{'='*50}\n")

    env.close()