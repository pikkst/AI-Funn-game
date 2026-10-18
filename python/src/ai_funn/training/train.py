"""PPO training entry point for survival-v2."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ..environments.survival_env import SurvivalEnv
from ..experiments.config import ExperimentConfig
from .manifest import (
    DEFAULT_RUNS_ROOT,
    git_commit,
    make_run_id,
    run_path,
    write_checkpoint_companion,
    write_run_manifest,
)

DEFAULT_N_STEPS: int = 2048
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_N_EPOCHS: int = 10
DEFAULT_GAMMA: float = 0.99
SMOKE_TOTAL_TIMESTEPS: int = 5_000


def _load_config(config_path: Path | str) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load the YAML config file and return the parsed config plus raw training section."""
    with open(Path(config_path), encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping")
    config = ExperimentConfig.from_dict(raw)
    training_raw = raw.get("training", {})
    if not isinstance(training_raw, dict):
        raise ValueError("Config training section must be a mapping")
    return config, training_raw


def _make_vec_env(n_envs: int, seed: int) -> DummyVecEnv:
    """Create ``n_envs`` independent SurvivalEnv instances and seed them."""
    env_fns: list[Callable[[], SurvivalEnv]] = [lambda: SurvivalEnv() for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)  # type: ignore[arg-type]
    vec_env.seed(seed)
    return vec_env


def train(
    *,
    config_path: Path | str,
    smoke: bool = False,
    runs_root: Path | str = DEFAULT_RUNS_ROOT,
    total_timesteps_override: int | None = None,
) -> dict[str, Any]:
    """Train a PPO policy, save the checkpoint and manifests; return a run summary."""
    config, training_raw = _load_config(config_path)
    run_id = make_run_id(config.experiment_id, config.training.seed)
    run_dir = run_path(runs_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tb"

    torch.manual_seed(config.training.seed)

    total_timesteps = SMOKE_TOTAL_TIMESTEPS if smoke else config.training.total_timesteps
    if total_timesteps_override is not None:
        total_timesteps = int(total_timesteps_override)

    n_steps = int(training_raw.get("n_steps", DEFAULT_N_STEPS))
    batch_size = int(training_raw.get("batch_size", DEFAULT_BATCH_SIZE))
    n_epochs = int(training_raw.get("n_epochs", DEFAULT_N_EPOCHS))
    gamma = float(training_raw.get("gamma", DEFAULT_GAMMA))

    vec_env = _make_vec_env(config.training.n_envs, config.training.seed)

    hidden_layers = list(config.training.hidden_layers)
    policy_kwargs = dict(net_arch=[dict(pi=list(hidden_layers), vf=list(hidden_layers))])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.training.learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        seed=config.training.seed,
        device=config.training.device,
        verbose=1,
    )

    start_time = time.perf_counter()
    model.learn(total_timesteps=int(total_timesteps), tb_log_name=run_id)
    elapsed_seconds = time.perf_counter() - start_time

    trained_timesteps = int(model.num_timesteps)
    checkpoint_name = f"{run_id}.zip"
    model.save(str(run_dir / checkpoint_name))

    extra_hyperparameters = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
    }
    write_run_manifest(
        run_dir,
        config,
        git_revision=git_commit(),
        elapsed_seconds=elapsed_seconds,
        trained_timesteps=trained_timesteps,
        extra_hyperparameters=extra_hyperparameters,
    )
    write_checkpoint_companion(run_dir, checkpoint_name, trained_timesteps)

    vec_env.close()
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "trained_timesteps": trained_timesteps,
        "elapsed_seconds": elapsed_seconds,
    }


def _build_train_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the train command."""
    parser = argparse.ArgumentParser(
        prog="ai_funn.training.train", description="Train PPO on survival-v2"
    )
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument("--smoke", action="store_true", help="Short smoke run (5,000 timesteps)")
    parser.add_argument(
        "--runs-root", default=str(DEFAULT_RUNS_ROOT), help="Directory for run outputs"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the train command; return the process exit code."""
    parser = _build_train_parser()
    args = parser.parse_args(argv)
    summary = train(config_path=args.config, smoke=args.smoke, runs_root=args.runs_root)
    print(f"Training complete: run_id={summary['run_id']}")
    print(f"Run directory: {summary['run_dir']}")
    print(f"Trained timesteps: {summary['trained_timesteps']}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
