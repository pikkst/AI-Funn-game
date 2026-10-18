"""Survival-v2 Gymnasium environment adapter.

Wraps the deterministic simulation core in the stable Gymnasium contract
documented in docs/ARCHITECTURE.md and docs/SIMULATION.md:

- action_space: Discrete(6)
- observation_space: Box(0.0, 1.0, shape=(199,), dtype=float32)
- reset(*, seed=None, options=None) -> (obs, info)
- step(action) -> (obs, reward, terminated, truncated, info)

Terminal death sets terminated=True, truncated=False.  The 512-step cap
sets terminated=False, truncated=True.  Death on the final step takes
precedence over the time-limit truncation.
"""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from ..experiments.config import (
    CANONICAL_ACTION_COUNT,
    CANONICAL_MAX_EPISODE_STEPS,
    CANONICAL_OBSERVATION_DIM,
    EnvironmentConfig,
)
from ..simulation import generate_world, step
from ..simulation.world import WorldState
from .observation import encode_observation

ENVIRONMENT_VERSION = "survival-v2"
OBSERVATION_DIM = CANONICAL_OBSERVATION_DIM
ACTION_COUNT = CANONICAL_ACTION_COUNT
MAX_EPISODE_STEPS = CANONICAL_MAX_EPISODE_STEPS

REWARD_PER_STEP = 0.01
REWARD_DEATH = -1.0


class SurvivalEnv(gym.Env[npt.NDArray[np.float32], int]):
    """Gymnasium adapter for the survival-v2 simulation."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = cast(gym.spaces.Space[int], gym.spaces.Discrete(ACTION_COUNT))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self.environment_config: EnvironmentConfig | None = None
        self.current_state: WorldState | None = None
        self.world_seed: int | None = None
        self.step_count = 0
        self.food_eaten = 0.0
        self.death_cause: str | None = None
        self.episode_end_reason: str | None = None
        self.episode_active = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset the environment, optionally seeding the world generator."""
        del options
        super().reset(seed=seed)
        env_config = EnvironmentConfig()
        if seed is None:
            world_seed = int(self.np_random.integers(0, 2**31 - 1))
        else:
            world_seed = int(seed)
        world_state = generate_world(world_seed, env_config)
        self.environment_config = env_config
        self.current_state = world_state
        self.world_seed = world_seed
        self.step_count = 0
        self.food_eaten = 0.0
        self.death_cause = None
        self.episode_end_reason = None
        self.episode_active = True
        obs = encode_observation(world_state)
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one step."""
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError(
                f"Invalid action {action!r}, must be an integer in [0, {ACTION_COUNT})"
            )
        action_int = int(action)
        if not (0 <= action_int < ACTION_COUNT):
            raise ValueError(f"Invalid action {action_int}, must be in [0, {ACTION_COUNT})")
        if not self.episode_active or self.current_state is None:
            raise ValueError("episode already terminated/truncated; call reset() before step()")

        new_state, reward, terminated, truncated, sim_info = step(self.current_state, action_int)
        events = sim_info.get("events", []) if isinstance(sim_info, dict) else []
        for event in events:
            if isinstance(event, str) and event.startswith("ate:"):
                try:
                    self.food_eaten += float(event.split(":", 1)[1])
                except (ValueError, IndexError):
                    pass

        self.current_state = new_state
        self.step_count = int(new_state.step)
        if terminated:
            self.death_cause = self._infer_death_cause(new_state, events)
        else:
            self.death_cause = None

        if terminated:
            self.episode_end_reason = "death"
        elif truncated:
            self.episode_end_reason = "time_limit"
        else:
            self.episode_end_reason = None

        if terminated or truncated:
            self.episode_active = False

        obs = encode_observation(new_state)
        info = self._build_info()
        return obs, float(reward), bool(terminated), bool(truncated), info

    @staticmethod
    def _infer_death_cause(state: WorldState, events: list[str]) -> str:
        if state.health <= 0.0:
            if "starving" in events:
                return "starvation"
            return "starvation"
        return "unknown"

    def _build_info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "food_eaten": float(self.food_eaten),
            "death_cause": self.death_cause,
            "world_seed": self.world_seed,
            "episode_end_reason": self.episode_end_reason,
        }

    def close(self) -> None:
        pass
