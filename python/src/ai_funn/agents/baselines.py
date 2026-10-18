"""Baseline policy implementations for AI-Funn-game."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BasePolicy(ABC):
    """Abstract base class for all agent policies."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize policy with optional RNG seed."""
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def reset(self, seed: int | None = None) -> None:
        """Re-seed policy RNG (separate from world RNG)."""
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, observation: npt.NDArray[np.float32]) -> int:
        """Return action in [0, 6)."""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Uniform random action policy."""

    def act(self, observation: npt.NDArray[np.float32]) -> int:
        """Return a uniform random action in [0, 6). Ignores observation."""
        del observation
        return int(self.rng.integers(0, 6))


class AlwaysRestPolicy(BasePolicy):
    """Policy that always rests (action 5)."""

    def act(self, observation: npt.NDArray[np.float32]) -> int:
        """Return action 5 (rest). Ignores observation."""
        del observation
        return 5


class EatWhenAvailablePolicy(BasePolicy):
    """Eat when food is available at center tile, otherwise move randomly."""

    def act(self, observation: npt.NDArray[np.float32]) -> int:
        """Eat if food at center (obs[73] > 0), else random movement 0-3."""
        if observation[73] > 0:
            return 4  # EAT
        return int(self.rng.integers(0, 4))  # random movement actions 0-3


class SimpleWalkerPolicy(BasePolicy):
    """Reproducible movement cycling through [0, 1, 2, 3] repeating."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize with step index counter."""
        super().__init__(seed=seed)
        self._step_index = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset step index to 0 and optionally re-seed policy RNG."""
        super().reset(seed=seed)
        self._step_index = 0

    def act(self, observation: npt.NDArray[np.float32]) -> int:
        """Return next action in cycle [0, 1, 2, 3]. No RNG/observation used."""
        del observation
        action = self._step_index % 4
        self._step_index += 1
        return action
