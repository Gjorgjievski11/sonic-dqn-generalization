"""
evaluate.py — entry point for agent evaluation.
Run: python evaluate.py --config configs/dqn_sonic.yaml
     python evaluate.py --config configs/dqn_sonic.yaml --checkpoint runs/checkpoints/dqn_step5000.pt
"""

import argparse
from sonic_dqn.utils.config import load_config
from sonic_dqn.utils.seeding import seed_everything
from sonic_dqn.evaluation.evaluator import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN on Sonic the Hedgehog")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint file (defaults to dqn_final.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["experiment"]["seed"])
    evaluate(cfg, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()