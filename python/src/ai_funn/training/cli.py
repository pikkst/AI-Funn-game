"""Command-line interface for training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import evaluate
from .manifest import DEFAULT_RUNS_ROOT
from .train import train


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with train/evaluate subcommands."""
    parser = argparse.ArgumentParser(
        prog="ai_funn.training", description="PPO training CLI for AI-Funn-game"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PPO policy on survival-v2")
    train_parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    train_parser.add_argument(
        "--smoke", action="store_true", help="Short smoke run (5,000 timesteps)"
    )
    train_parser.add_argument(
        "--runs-root", default=str(DEFAULT_RUNS_ROOT), help="Directory for run outputs"
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a frozen PPO checkpoint")
    evaluate_parser.add_argument(
        "--run-id", required=True, help="Run identifier produced by training"
    )
    evaluate_parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Seed split to evaluate on",
    )
    evaluate_parser.add_argument(
        "--runs-root", default=str(DEFAULT_RUNS_ROOT), help="Directory for run outputs"
    )
    evaluate_parser.add_argument("--policy-seed", type=int, default=0, help="Recorded policy seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "train":
        summary = train(config_path=args.config, smoke=args.smoke, runs_root=args.runs_root)
        print(f"Training complete: run_id={summary['run_id']}")
        print(f"Run directory: {summary['run_dir']}")
        print(f"Trained timesteps: {summary['trained_timesteps']}")
        print(f"Elapsed seconds: {summary['elapsed_seconds']:.3f}")
        return 0
    if args.command == "evaluate":
        result = evaluate(
            run_id=args.run_id,
            split=args.split,
            runs_root=args.runs_root,
            policy_seed=args.policy_seed,
        )
        print(f"Evaluation complete: split={args.split}")
        print(f"Policy: {result.policy_name}  episodes: {result.num_episodes}")
        print(
            f"Mean survival: {result.mean_survival_steps:.4f}  median: {result.median_survival_steps:.4f}"
        )
        print(f"Mean food: {result.mean_food_consumed:.4f}  mean reward: {result.mean_reward:.4f}")
        print(
            f"Termination rate: {result.termination_rate:.4f}  truncation rate: {result.truncation_rate:.4f}"
        )
        print(f"Death causes: {result.death_causes}")
        print(f"Results: {Path(args.runs_root) / args.run_id / f'eval_{args.split}.json'}")
        return 0
    return 1
