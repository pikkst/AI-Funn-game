"""Tests for the DL-014 PPO training CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_funn.environments.survival_env import SurvivalEnv
from ai_funn.experiments.config import (
    ConfigurationError,
    ExperimentConfig,
    ExperimentManifest,
    SeedSplit,
    TrainingConfig,
    create_manifest,
)
from ai_funn.training.evaluate import evaluate
from ai_funn.training.manifest import (
    git_commit,
    library_versions,
    make_run_id,
    read_json,
    run_path,
    sha256_file,
    write_json,
)
from ai_funn.training.train import train

STABLE_HEX = set("0123456789abcdef")


def _write_config(tmp_path: Path, config: ExperimentConfig) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    return config_path


def _smoke_config(tmp_path: Path, experiment_id: str = "test-smoke", seed: int = 7) -> Path:
    config = ExperimentConfig(
        experiment_id=experiment_id,
        seeds=SeedSplit(train_seeds=[11], validation_seeds=[21], test_seeds=[31]),
        training=TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=64,
            n_envs=1,
            learning_rate=3e-4,
            hidden_layers=[128, 128],
            device="cpu",
            seed=seed,
        ),
    )
    return _write_config(tmp_path, config)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="roundtrip",
        seeds=SeedSplit(train_seeds=[1], validation_seeds=[2], test_seeds=[3]),
        training=TrainingConfig(
            algorithm="PPO",
            policy="MlpPolicy",
            total_timesteps=100,
            n_envs=2,
            learning_rate=1e-3,
            hidden_layers=[128, 128],
            device="cpu",
            seed=5,
        ),
    )
    manifest = create_manifest(
        config,
        git_commit="abc123",
        library_versions={"numpy": "1.2.3"},
        elapsed_time_seconds=10.0,
        trained_timesteps=100,
    )
    write_json(tmp_path / "manifest.json", manifest.to_dict())
    loaded = ExperimentManifest.from_dict(read_json(tmp_path / "manifest.json"))
    assert loaded.config.experiment_id == "roundtrip"
    assert loaded.git_commit == "abc123"
    assert loaded.trained_timesteps == 100
    assert loaded.library_versions == {"numpy": "1.2.3"}


def test_git_commit_format() -> None:
    commit = git_commit()
    assert commit is None or (len(commit) == 40 and set(commit) <= STABLE_HEX)


def test_library_versions() -> None:
    versions = library_versions()
    assert set(versions) == {"gymnasium", "numpy", "stable-baselines3", "torch"}
    assert all(isinstance(value, str) and value for value in versions.values())


def test_make_run_id_format() -> None:
    run_id = make_run_id("exp1", 3)
    assert run_id.startswith("exp1-s3-")
    stamp = run_id.split("-", 2)[2]
    assert len(stamp) == 15
    assert stamp[8] == "T"


def test_sha256_file_deterministic(tmp_path: Path) -> None:
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert sha256_file(empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    text = tmp_path / "text.bin"
    text.write_text("hello", encoding="utf-8")
    assert sha256_file(text) == sha256_file(text)
    assert len(sha256_file(text)) == 64


def test_survival_001_config_loads() -> None:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "survival-001.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    config = ExperimentConfig.from_dict(raw)
    assert config.experiment_id == "survival-001"
    assert config.environment.environment_version == "survival-v2"
    assert config.observation.observation_dim == 199
    assert config.action.action_count == 6
    assert config.training.n_envs == 8
    assert config.training.hidden_layers == [128, 128]
    assert config.training.total_timesteps == 500000
    all_seeds = (
        list(config.seeds.train_seeds)
        + list(config.seeds.validation_seeds)
        + list(config.seeds.test_seeds)
    )
    assert len(all_seeds) == len(set(all_seeds))


def test_seed_partitions_overlap_rejected() -> None:
    with pytest.raises(ConfigurationError):
        SeedSplit(train_seeds=[1, 2], validation_seeds=[2, 3], test_seeds=[4])


def test_vector_env_reset_step_shapes() -> None:
    vec_env = DummyVecEnv([lambda: SurvivalEnv() for _ in range(2)])
    vec_env.seed(123)
    try:
        obs = vec_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2, 199)
        assert obs.dtype == np.float32
        actions = np.array([0, 1], dtype=np.int64)
        obs, rewards, dones, infos = vec_env.step(actions)
        assert obs.shape == (2, 199)
        assert obs.dtype == np.float32
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
        assert isinstance(vec_env.action_space, gym.spaces.Discrete)
        assert vec_env.action_space.n == 6
    finally:
        vec_env.close()


def _load_model(checkpoint: Path) -> PPO:
    return PPO.load(str(checkpoint), env=None, device="cpu")


def test_ppo_smoke_train_save_load_predict(tmp_path: Path) -> None:
    config_path = _smoke_config(tmp_path)
    summary = train(
        config_path=config_path,
        smoke=False,
        runs_root=tmp_path / "runs",
        total_timesteps_override=64,
    )
    run_id = summary["run_id"]
    assert run_id.startswith("test-smoke-s7-")
    run_dir = run_path(tmp_path / "runs", run_id)
    checkpoint = run_dir / f"{run_id}.zip"
    companion = run_dir / f"{run_id}.zip.manifest.json"
    assert checkpoint.is_file()
    assert companion.is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "config.json").is_file()

    companion_data = json.loads(companion.read_text(encoding="utf-8"))
    assert companion_data["sha256"] == sha256_file(checkpoint)
    assert companion_data["trained_timesteps"] == summary["trained_timesteps"]
    assert companion_data["observation_dim"] == 199
    assert companion_data["action_count"] == 6

    model_1 = _load_model(checkpoint)
    model_2 = _load_model(checkpoint)
    obs = np.zeros(199, dtype=np.float32)
    action_1 = int(np.asarray(model_1.predict(obs, deterministic=True)[0]).reshape(-1)[0])
    action_2 = int(np.asarray(model_2.predict(obs, deterministic=True)[0]).reshape(-1)[0])
    assert action_1 == action_2
    assert 0 <= action_1 < 6


def test_train_uses_torch_manual_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_funn.training.train as train_module

    calls: list[int] = []
    monkeypatch.setattr(train_module.torch, "manual_seed", lambda seed: calls.append(int(seed)))
    config_path = _smoke_config(tmp_path)
    train(
        config_path=config_path,
        smoke=False,
        runs_root=tmp_path / "runs",
        total_timesteps_override=64,
    )
    assert calls
    assert calls[0] == 7


def test_evaluate_deterministic(tmp_path: Path) -> None:
    config_path = _smoke_config(tmp_path)
    summary = train(
        config_path=config_path,
        smoke=False,
        runs_root=tmp_path / "runs",
        total_timesteps_override=64,
    )
    run_id = summary["run_id"]
    runs_root = tmp_path / "runs"

    result_a = evaluate(run_id=run_id, split="test", runs_root=runs_root, policy_seed=0)
    result_b = evaluate(run_id=run_id, split="test", runs_root=runs_root, policy_seed=0)
    assert result_a.num_episodes == 1
    assert result_b.num_episodes == 1
    assert result_a.mean_survival_steps == result_b.mean_survival_steps
    assert result_a.death_causes == result_b.death_causes
    assert (runs_root / run_id / "eval_test.json").is_file()
    assert isinstance(result_a.policy_name, str)


def test_cli_help_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "ai_funn.training", "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "train" in completed.stdout
    assert "evaluate" in completed.stdout
