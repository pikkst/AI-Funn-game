from __future__ import annotations

import json

import pytest

from ai_funn.experiments.config import (
    ACTION_VERSION,
    CANONICAL_ACTION_COUNT,
    CANONICAL_MAX_EPISODE_STEPS,
    CANONICAL_OBSERVATION_DIM,
    CANONICAL_WORLD_HEIGHT,
    CANONICAL_WORLD_WIDTH,
    ENVIRONMENT_VERSION,
    OBSERVATION_VERSION,
    ConfigurationError,
    ExperimentConfig,
    ExperimentManifest,
    SeedSplit,
    TrainingConfig,
    create_manifest,
)


def test_default_config_canonical():
    c = ExperimentConfig(experiment_id="x")
    assert c.environment.environment_version == ENVIRONMENT_VERSION
    assert c.observation.observation_version == OBSERVATION_VERSION
    assert c.action.action_version == ACTION_VERSION
    assert c.environment.world_width == CANONICAL_WORLD_WIDTH
    assert c.environment.world_height == CANONICAL_WORLD_HEIGHT
    assert c.environment.max_episode_steps == CANONICAL_MAX_EPISODE_STEPS
    assert c.observation.observation_dim == CANONICAL_OBSERVATION_DIM
    assert c.action.action_count == CANONICAL_ACTION_COUNT


def test_seed_overlap_rejected():
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[1], validation_seeds=[1], test_seeds=[2])


def test_seed_overlap_train_test():
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[1], validation_seeds=[2], test_seeds=[1])


def test_seed_overlap_val_test():
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[1], validation_seeds=[2], test_seeds=[2])


def test_empty_seed_partition():
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[], validation_seeds=[1], test_seeds=[2])


def test_negative_seed():
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[-1], validation_seeds=[1], test_seeds=[2])


def test_duplicate_seed():
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[1, 1], validation_seeds=[2], test_seeds=[3])


def test_bool_seed_rejected():
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[True], validation_seeds=[1], test_seeds=[2])


def test_non_int_seed_rejected():
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[1.0], validation_seeds=[2], test_seeds=[3])  # type: ignore


def test_large_seed():
    big = 2**31 - 1
    s = SeedSplit(train_seeds=[big], validation_seeds=[big - 1], test_seeds=[big - 2])
    assert big in s.train_seeds


def test_invalid_versions():
    with pytest.raises(ConfigurationError):
        ExperimentConfig(
            experiment_id="x",
            environment=__import__(
                "ai_funn.experiments.config", fromlist=["EnvironmentConfig"]
            ).EnvironmentConfig(environment_version="nope"),
        )
    with pytest.raises(ConfigurationError):
        ExperimentConfig(
            experiment_id="x",
            observation=__import__(
                "ai_funn.experiments.config", fromlist=["ObservationConfig"]
            ).ObservationConfig(observation_version="nope"),
        )
    with pytest.raises(ConfigurationError):
        ExperimentConfig(
            experiment_id="x",
            action=__import__("ai_funn.experiments.config", fromlist=["ActionConfig"]).ActionConfig(
                action_version="nope"
            ),
        )


def test_wrong_dims():
    from ai_funn.experiments.config import ActionConfig, EnvironmentConfig, ObservationConfig

    with pytest.raises(ConfigurationError):
        EnvironmentConfig(world_width=16)
    with pytest.raises(ConfigurationError):
        EnvironmentConfig(world_height=16)
    with pytest.raises(ConfigurationError):
        EnvironmentConfig(max_episode_steps=100)
    with pytest.raises(ConfigurationError):
        ObservationConfig(observation_dim=10)
    with pytest.raises(ConfigurationError):
        ActionConfig(action_count=4)


def test_empty_experiment_id():
    with pytest.raises(ConfigurationError):
        ExperimentConfig(experiment_id="")


def test_training_validation():
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[8],
            device="cpu",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="",
            total_timesteps=1,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[8],
            device="cpu",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=0,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[8],
            device="cpu",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=0,
            learning_rate=1e-3,
            hidden_layers=[8],
            device="cpu",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=1,
            learning_rate=0,
            hidden_layers=[8],
            device="cpu",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[],
            device="cpu",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[0],
            device="cpu",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[True],
            device="cpu",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[8],
            device="",
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[8],
            device="cpu",
            seed=True,
        )
    with pytest.raises(ConfigurationError):
        TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=1,
            n_envs=1,
            learning_rate=1e-3,
            hidden_layers=[8],
            device="cpu",
            seed=-1,
        )


def test_serialization_determinism():
    c = ExperimentConfig(experiment_id="det")
    assert c.to_json() == c.to_json()


def test_json_roundtrip():
    c = ExperimentConfig(experiment_id="rt")
    c2 = ExperimentConfig.from_json(c.to_json())
    assert c == c2


def test_dict_roundtrip():
    c = ExperimentConfig(experiment_id="rt2")
    c2 = ExperimentConfig.from_dict(c.to_dict())
    assert c == c2


def test_missing_field():
    with pytest.raises(ConfigurationError):
        ExperimentConfig.from_dict({"description": "x"})


def test_manifest_roundtrip():
    c = ExperimentConfig(experiment_id="m")
    m = create_manifest(c, git_commit="abc", trained_timesteps=10)
    m2 = ExperimentManifest.from_json(m.to_json())
    assert m == m2
    assert m.hyperparameters["seed"] == c.training.seed


def test_manifest_lib_versions():
    c = ExperimentConfig(experiment_id="m2")
    m = create_manifest(c, library_versions={"numpy": "1.26.4"})
    assert m.library_versions == {"numpy": "1.26.4"}
    assert m.git_commit is None


def test_json_keys_sorted():
    c = ExperimentConfig(experiment_id="k")
    text = c.to_json()
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
