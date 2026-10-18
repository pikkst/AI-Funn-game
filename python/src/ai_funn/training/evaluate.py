"""Frozen-policy evaluation of a trained PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
from stable_baselines3 import PPO

from ..agents.baselines import BasePolicy
from ..environments.survival_env import SurvivalEnv
from ..experiments.config import ExperimentConfig
from ..training.evaluation import EvaluationResult, run_evaluation, save_results
from .manifest import DEFAULT_RUNS_ROOT, load_run_config, require_run_dir, run_path

DEFAULT_POLICY_SEED: int = 0


class FrozenPPOPolicy(BasePolicy):
    """Deterministic wrapper around a frozen SB3 PPO policy for evaluation."""

    def __init__(self, model: PPO, seed: int = DEFAULT_POLICY_SEED) -> None:
        """Wrap a loaded PPO model; ``seed`` is recorded but actions are deterministic."""
        super().__init__(seed=seed)
        self.model = model

    def act(self, observation: npt.NDArray[np.float32]) -> int:
        """Return the deterministic greedy action for the observation."""
        action, _states = self.model.predict(observation, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])


def _select_seeds(config: ExperimentConfig, split: str) -> list[int]:
    """Return the seed list for a split, raising on unknown split names."""
    if split == "train":
        return list(config.seeds.train_seeds)
    if split == "validation":
        return list(config.seeds.validation_seeds)
    if split == "test":
        return list(config.seeds.test_seeds)
    raise ValueError(f"Unknown split: {split}")


def evaluate(
    *,
    run_id: str,
    split: str = "test",
    runs_root: Path | str = DEFAULT_RUNS_ROOT,
    policy_seed: int = DEFAULT_POLICY_SEED,
) -> EvaluationResult:
    """Evaluate the frozen checkpoint of a run on a disjoint seed split."""
    run_dir = require_run_dir(runs_root, run_id)
    config = load_run_config(run_dir)
    checkpoint_path = run_dir / f"{run_id}.zip"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = PPO.load(str(checkpoint_path), env=None, device="cpu")
    env = SurvivalEnv()
    try:
        policy = FrozenPPOPolicy(model, seed=policy_seed)
        seeds = _select_seeds(config, split)
        result = run_evaluation(env, policy, seeds, policy_seed=policy_seed)
    finally:
        env.close()
    save_results(result, run_dir / f"eval_{split}.json")
    return result


def _build_evaluate_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluate command."""
    parser = argparse.ArgumentParser(
        prog="ai_funn.training.evaluate",
        description="Evaluate a frozen PPO checkpoint on survival-v2",
    )
    parser.add_argument("--run-id", required=True, help="Run identifier produced by training")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Seed split to evaluate on",
    )
    parser.add_argument(
        "--runs-root", default=str(DEFAULT_RUNS_ROOT), help="Directory for run outputs"
    )
    parser.add_argument(
        "--policy-seed", type=int, default=DEFAULT_POLICY_SEED, help="Recorded policy seed"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evaluate command; return the process exit code."""
    parser = _build_evaluate_parser()
    args = parser.parse_args(argv)
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
    print(f"Results: {run_path(args.runs_root, args.run_id) / f'eval_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
