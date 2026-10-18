"""Run manifests, run directories and artifact checksums for training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..experiments.config import (
    ACTION_VERSION,
    CANONICAL_ACTION_COUNT,
    CANONICAL_MAX_EPISODE_STEPS,
    CANONICAL_OBSERVATION_DIM,
    ENVIRONMENT_VERSION,
    OBSERVATION_VERSION,
    ExperimentConfig,
    ExperimentManifest,
    create_manifest,
)

DEFAULT_RUNS_ROOT: Path = Path("runs")

CHECKPOINT_ARCHITECTURE: str = "MlpPolicy"

CHECKPOINT_POLICY_KWARGS: dict[str, Any] = {"net_arch": [{"pi": [128, 128], "vf": [128, 128]}]}


def git_commit() -> str | None:
    """Return the git HEAD revision of the repository, or None when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def library_versions() -> dict[str, str]:
    """Return the library versions recorded in every training manifest."""
    import gymnasium
    import numpy
    import stable_baselines3
    import torch

    return {
        "gymnasium": gymnasium.__version__,
        "numpy": numpy.__version__,
        "stable-baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    }


def make_run_id(experiment_id: str, seed: int) -> str:
    """Build a run identifier: ``{experiment_id}-s{seed}-{utc timestamp}``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{experiment_id}-s{seed}-{stamp}"


def run_path(runs_root: Path | str, run_id: str) -> Path:
    """Directory for a run: ``{runs_root}/{run_id}``."""
    return Path(runs_root) / run_id


def require_run_dir(runs_root: Path | str, run_id: str) -> Path:
    """Return the run directory, raising ``FileNotFoundError`` when missing."""
    directory = run_path(runs_root, run_id)
    if not directory.is_dir():
        raise FileNotFoundError(f"Run directory not found: {directory}")
    return directory


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path | str, data: dict[str, Any]) -> None:
    """Write JSON with sorted keys, creating parent directories as needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), indent=2)
    file_path.write_text(text + "\n", encoding="utf-8")


def read_json(path: Path | str) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    with open(path, encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def write_run_manifest(
    run_dir: Path | str,
    config: ExperimentConfig,
    *,
    git_revision: str | None,
    elapsed_seconds: float,
    trained_timesteps: int,
    extra_hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``manifest.json`` and ``config.json`` for a run; return manifest data."""
    run_dir = Path(run_dir)
    manifest = create_manifest(
        config,
        git_commit=git_revision,
        library_versions=library_versions(),
        elapsed_time_seconds=elapsed_seconds,
        trained_timesteps=trained_timesteps,
    )
    data = manifest.to_dict()
    if extra_hyperparameters:
        data["hyperparameters"].update(extra_hyperparameters)
    write_json(run_dir / "manifest.json", data)
    write_json(run_dir / "config.json", config.to_dict())
    return data


def write_checkpoint_companion(
    run_dir: Path | str,
    checkpoint_name: str,
    trained_timesteps: int,
    policy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the companion manifest next to an SB3 checkpoint zip; return its data."""
    run_dir = Path(run_dir)
    zip_path = run_dir / checkpoint_name
    digest = sha256_file(zip_path)
    data = {
        "run_id": run_dir.name,
        "checkpoint": checkpoint_name,
        "sha256": digest,
        "trained_timesteps": int(trained_timesteps),
        "environment_version": ENVIRONMENT_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "observation_dim": CANONICAL_OBSERVATION_DIM,
        "action_count": CANONICAL_ACTION_COUNT,
        "max_episode_steps": CANONICAL_MAX_EPISODE_STEPS,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "policy_kwargs": CHECKPOINT_POLICY_KWARGS if policy_kwargs is None else policy_kwargs,
        "library_versions": library_versions(),
    }
    companion = run_dir / f"{checkpoint_name}.manifest.json"
    write_json(companion, data)
    return data


def load_run_config(run_dir: Path | str) -> ExperimentConfig:
    """Load the :class:`ExperimentConfig` recorded for a run."""
    return ExperimentConfig.from_dict(read_json(Path(run_dir) / "config.json"))


def load_run_manifest(run_dir: Path | str) -> ExperimentManifest:
    """Load the :class:`ExperimentManifest` recorded for a run."""
    return ExperimentManifest.from_dict(read_json(Path(run_dir) / "manifest.json"))
