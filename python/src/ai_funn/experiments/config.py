"""Versioned experiment configuration and reproducibility manifests."""

from __future__ import annotations

import datetime
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy


class ConfigurationError(ValueError):
    def __init__(self, message: str, field_path: str | None = None) -> None:
        self.field_path = field_path
        super().__init__(message)


ENVIRONMENT_VERSION: str = "survival-v1"
OBSERVATION_VERSION: str = "obs-v1"
ACTION_VERSION: str = "act-v1"

CANONICAL_WORLD_WIDTH: int = 32
CANONICAL_WORLD_HEIGHT: int = 32
CANONICAL_OBSERVATION_DIM: int = 199
CANONICAL_ACTION_COUNT: int = 6
CANONICAL_MAX_EPISODE_STEPS: int = 512


def _require_non_empty_str(value: Any, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_path} must be a non-empty string", field_path=field_path)
    return value


def _require_non_negative_int(value: Any, field_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"{field_path} must be a non-negative integer", field_path=field_path
        )
    if value < 0:
        raise ConfigurationError(
            f"{field_path} must be a non-negative integer", field_path=field_path
        )
    return int(value)


@dataclass(frozen=True)
class EnvironmentConfig:
    environment_version: str = ENVIRONMENT_VERSION
    world_width: int = CANONICAL_WORLD_WIDTH
    world_height: int = CANONICAL_WORLD_HEIGHT
    max_episode_steps: int = CANONICAL_MAX_EPISODE_STEPS

    def __post_init__(self) -> None:
        if self.environment_version != ENVIRONMENT_VERSION:
            raise ConfigurationError(
                "environment_version must be 'survival-v1'",
                field_path="environment.environment_version",
            )
        if self.world_width != CANONICAL_WORLD_WIDTH:
            raise ConfigurationError("world_width must be 32", field_path="environment.world_width")
        if self.world_height != CANONICAL_WORLD_HEIGHT:
            raise ConfigurationError(
                "world_height must be 32", field_path="environment.world_height"
            )
        if self.max_episode_steps != CANONICAL_MAX_EPISODE_STEPS:
            raise ConfigurationError(
                "max_episode_steps must be 512", field_path="environment.max_episode_steps"
            )


@dataclass(frozen=True)
class ObservationConfig:
    observation_version: str = OBSERVATION_VERSION
    observation_dim: int = CANONICAL_OBSERVATION_DIM

    def __post_init__(self) -> None:
        if self.observation_version != OBSERVATION_VERSION:
            raise ConfigurationError(
                "observation_version must be 'obs-v1'", field_path="observation.observation_version"
            )
        if self.observation_dim != CANONICAL_OBSERVATION_DIM:
            raise ConfigurationError(
                "observation_dim must be 199", field_path="observation.observation_dim"
            )


@dataclass(frozen=True)
class ActionConfig:
    action_version: str = ACTION_VERSION
    action_count: int = CANONICAL_ACTION_COUNT

    def __post_init__(self) -> None:
        if self.action_version != ACTION_VERSION:
            raise ConfigurationError(
                "action_version must be 'act-v1'", field_path="action.action_version"
            )
        if self.action_count != CANONICAL_ACTION_COUNT:
            raise ConfigurationError("action_count must be 6", field_path="action.action_count")


def _validate_seed_partition(name: str, seeds: list[int]) -> None:
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ConfigurationError(f"{name} seeds must be non-empty list", field_path=f"seeds.{name}")
    seen: set[int] = set()
    for i, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError(
                f"{name} seeds must be integers", field_path=f"seeds.{name}[{i}]"
            )
        if seed < 0:
            raise ConfigurationError(
                f"{name} seeds must be non-negative", field_path=f"seeds.{name}[{i}]"
            )
        if seed in seen:
            raise ConfigurationError(
                f"duplicate seed in {name}: {seed}", field_path=f"seeds.{name}[{i}]"
            )
        seen.add(seed)


@dataclass(frozen=True)
class SeedSplit:
    train_seeds: list[int]
    validation_seeds: list[int]
    test_seeds: list[int]

    def __post_init__(self) -> None:
        _validate_seed_partition("train_seeds", self.train_seeds)
        _validate_seed_partition("validation_seeds", self.validation_seeds)
        _validate_seed_partition("test_seeds", self.test_seeds)
        train_set = set(self.train_seeds)
        validation_set = set(self.validation_seeds)
        test_set = set(self.test_seeds)
        overlap = train_set & validation_set | train_set & test_set | validation_set & test_set
        if overlap:
            raise ConfigurationError(
                f"Seed partitions overlap: {sorted(overlap)}", field_path="seeds"
            )


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str
    policy: str
    total_timesteps: int
    n_envs: int
    learning_rate: float
    hidden_layers: list[int]
    device: str
    seed: int = 0

    def __post_init__(self) -> None:
        _require_non_empty_str(self.algorithm, "training.algorithm")
        _require_non_empty_str(self.policy, "training.policy")
        _require_non_empty_str(self.device, "training.device")
        _require_non_negative_int(self.total_timesteps, "training.total_timesteps")
        if self.total_timesteps <= 0:
            raise ConfigurationError(
                "total_timesteps must be positive", field_path="training.total_timesteps"
            )
        _require_non_negative_int(self.n_envs, "training.n_envs")
        if self.n_envs <= 0:
            raise ConfigurationError("n_envs must be positive", field_path="training.n_envs")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise ConfigurationError(
                "learning_rate must be number", field_path="training.learning_rate"
            )
        if float(self.learning_rate) <= 0.0:
            raise ConfigurationError(
                "learning_rate must be positive", field_path="training.learning_rate"
            )
        if not isinstance(self.hidden_layers, list) or len(self.hidden_layers) == 0:
            raise ConfigurationError(
                "hidden_layers must be non-empty list", field_path="training.hidden_layers"
            )
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise ConfigurationError(
                    "hidden_layers elements must be ints", field_path=f"training.hidden_layers[{i}]"
                )
            if layer <= 0:
                raise ConfigurationError(
                    "hidden_layers elements must be positive",
                    field_path=f"training.hidden_layers[{i}]",
                )
        _require_non_negative_int(self.seed, "training.seed")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str = ""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    seeds: SeedSplit = field(
        default_factory=lambda: SeedSplit(train_seeds=[0], validation_seeds=[1], test_seeds=[2])
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=500_000,
            n_envs=8,
            learning_rate=3e-4,
            hidden_layers=[128, 128],
            device="cpu",
        )
    )

    def __post_init__(self) -> None:
        _require_non_empty_str(self.experiment_id, "experiment_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        if not isinstance(data, dict):
            raise ConfigurationError("config must be dict", field_path=None)

        def require(key: str) -> Any:
            if key not in data:
                raise ConfigurationError(f"Missing required field {key}", field_path=key)
            return data[key]

        env_data = require("environment")
        obs_data = require("observation")
        act_data = require("action")
        seeds_data = require("seeds")
        for name, value in (
            ("environment", env_data),
            ("observation", obs_data),
            ("action", act_data),
            ("seeds", seeds_data),
        ):
            if not isinstance(value, dict):
                raise ConfigurationError(f"{name} must be dict", field_path=name)
        training_data = data.get("training", {})
        if not isinstance(training_data, dict):
            raise ConfigurationError("training must be dict", field_path="training")

        def _build(field_path: str, factory: Callable[[], Any]) -> Any:
            try:
                return factory()
            except ConfigurationError:
                raise
            except (KeyError, ValueError, TypeError) as exc:
                raise ConfigurationError(str(exc), field_path=field_path) from exc

        experiment_id = _build("experiment_id", lambda: str(require("experiment_id")))
        description = _build("description", lambda: str(data.get("description", "")))
        environment = _build(
            "environment",
            lambda: EnvironmentConfig(
                environment_version=str(env_data.get("environment_version", ENVIRONMENT_VERSION)),
                world_width=int(env_data.get("world_width", CANONICAL_WORLD_WIDTH)),
                world_height=int(env_data.get("world_height", CANONICAL_WORLD_HEIGHT)),
                max_episode_steps=int(
                    env_data.get("max_episode_steps", CANONICAL_MAX_EPISODE_STEPS)
                ),
            ),
        )
        observation = _build(
            "observation",
            lambda: ObservationConfig(
                observation_version=str(obs_data.get("observation_version", OBSERVATION_VERSION)),
                observation_dim=int(obs_data.get("observation_dim", CANONICAL_OBSERVATION_DIM)),
            ),
        )
        action = _build(
            "action",
            lambda: ActionConfig(
                action_version=str(act_data.get("action_version", ACTION_VERSION)),
                action_count=int(act_data.get("action_count", CANONICAL_ACTION_COUNT)),
            ),
        )
        seeds = _build(
            "seeds",
            lambda: SeedSplit(
                train_seeds=[int(s) for s in seeds_data["train_seeds"]],
                validation_seeds=[int(s) for s in seeds_data["validation_seeds"]],
                test_seeds=[int(s) for s in seeds_data["test_seeds"]],
            ),
        )
        training = _build(
            "training",
            lambda: TrainingConfig(
                algorithm=str(training_data.get("algorithm", "PPO")),
                policy=str(training_data.get("policy", "MlpPolicy")),
                total_timesteps=int(training_data.get("total_timesteps", 500_000)),
                n_envs=int(training_data.get("n_envs", 8)),
                learning_rate=float(training_data.get("learning_rate", 3e-4)),
                hidden_layers=[int(h) for h in training_data.get("hidden_layers", [128, 128])],
                device=str(training_data.get("device", "cpu")),
                seed=int(training_data.get("seed", 0)),
            ),
        )
        return cls(
            experiment_id=experiment_id,
            description=description,
            environment=environment,
            observation=observation,
            action=action,
            seeds=seeds,
            training=training,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> ExperimentConfig:
        try:
            data = json.loads(text)
        except Exception as exc:
            raise ConfigurationError(str(exc), field_path=None)
        return cls.from_dict(data)


@dataclass(frozen=True)
class ExperimentManifest:
    config: ExperimentConfig
    git_commit: str | None
    python_version: str
    library_versions: dict[str, str]
    created_at: str
    environment_version: str
    algorithm: str
    hyperparameters: dict[str, Any]
    seeds: dict[str, list[int]]
    device: str
    elapsed_time_seconds: float | None = None
    trained_timesteps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentManifest:
        if not isinstance(data, dict):
            raise ConfigurationError("manifest must be dict", field_path=None)

        def require(key: str) -> Any:
            if key not in data:
                raise ConfigurationError(f"Missing required field {key}", field_path=key)
            return data[key]

        config_data = require("config")
        if not isinstance(config_data, dict):
            raise ConfigurationError("config must be dict", field_path="config")
        return cls(
            config=ExperimentConfig.from_dict(config_data),
            git_commit=data.get("git_commit"),
            python_version=str(require("python_version")),
            library_versions=dict(require("library_versions")),
            created_at=str(require("created_at")),
            environment_version=str(require("environment_version")),
            algorithm=str(require("algorithm")),
            hyperparameters=dict(require("hyperparameters")),
            seeds=dict(require("seeds")),
            device=str(require("device")),
            elapsed_time_seconds=data.get("elapsed_time_seconds"),
            trained_timesteps=data.get("trained_timesteps"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> ExperimentManifest:
        try:
            data = json.loads(text)
        except Exception as exc:
            raise ConfigurationError(str(exc), field_path=None)
        return cls.from_dict(data)


def create_manifest(
    config: ExperimentConfig,
    *,
    git_commit: str | None = None,
    library_versions: dict[str, str] | None = None,
    elapsed_time_seconds: float | None = None,
    trained_timesteps: int | None = None,
) -> ExperimentManifest:
    if library_versions is None:
        libs: dict[str, str] = {"numpy": str(numpy.__version__)}
    else:
        libs = dict(library_versions)
    return ExperimentManifest(
        config=config,
        git_commit=git_commit,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        library_versions=libs,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        environment_version=config.environment.environment_version,
        algorithm=config.training.algorithm,
        hyperparameters={
            "policy": config.training.policy,
            "hidden_layers": list(config.training.hidden_layers),
            "total_timesteps": config.training.total_timesteps,
            "n_envs": config.training.n_envs,
            "learning_rate": config.training.learning_rate,
            "device": config.training.device,
            "seed": config.training.seed,
        },
        seeds={
            "train_seeds": list(config.seeds.train_seeds),
            "validation_seeds": list(config.seeds.validation_seeds),
            "test_seeds": list(config.seeds.test_seeds),
        },
        device=config.training.device,
        elapsed_time_seconds=elapsed_time_seconds,
        trained_timesteps=trained_timesteps,
    )
