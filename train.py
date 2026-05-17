"""
train.py — entry point for DQN training.
Run: python train.py --config configs/dqn_sonic.yaml
"""

import argparse
import wandb
from sonic_dqn.utils.config import load_config
from sonic_dqn.utils.seeding import seed_everything
from sonic_dqn.training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on Sonic the Hedgehog")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --override training.learning_rate=0.0001",
    )
    return parser.parse_args()


def main() -> None:
    config = load_config("configs/dqn_sonic.yaml")
    wandb.init(
        project=config.get("wandb", {}).get("project_name", "sonic-dqn"),
        entity=config.get("wandb", {}).get("entity", None),
        config=config,  # Tracks all your YAML hyperparameters automatically!
        sync_tensorboard=False,
        name="dqn-sonic-run", # Optional
        monitor_gym=False,  # Automatically log Gym env metrics like episode reward and length
    )
    args = parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg["experiment"]["seed"])
    train(cfg)
    wandb.finish()


if __name__ == "__main__":
    main()