"""Deterministic baseline evaluation runner for AI-Funn-game."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..agents.baselines import BasePolicy
from ..environments.survival_env import (
    ACTION_COUNT,
    ENVIRONMENT_VERSION,
    MAX_EPISODE_STEPS,
    OBSERVATION_DIM,
    SurvivalEnv,
)


@dataclass(frozen=True)
class EpisodeResult:
    """Result of a single evaluation episode."""

    world_seed: int
    policy_name: str
    survival_steps: int
    food_consumed: float
    death_cause: str | None
    terminated: bool
    truncated: bool
    action_frequencies: dict[int, int]
    reward_total: float


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated evaluation results across multiple episodes."""

    environment_version: str
    observation_dim: int
    action_count: int
    policy_name: str
    eval_seeds: list[int]
    num_episodes: int
    mean_survival_steps: float
    median_survival_steps: float
    min_survival: int
    max_survival: int
    mean_food_consumed: float
    death_causes: dict[str, int]
    termination_rate: float
    truncation_rate: float
    action_frequencies: dict[int, int]
    mean_reward: float
    config_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def run_episode(
    env: SurvivalEnv, policy: BasePolicy, world_seed: int, policy_seed: int
) -> EpisodeResult:
    """Run one episode: reset env with world_seed, reset policy with policy_seed, step until done."""
    obs, info = env.reset(seed=world_seed)
    del info
    policy.reset(seed=policy_seed)
    survival_steps = 0
    food_consumed = 0.0
    death_cause: str | None = None
    terminated = False
    truncated = False
    action_frequencies: dict[int, int] = {a: 0 for a in range(ACTION_COUNT)}
    reward_total = 0.0
    while True:
        action = policy.act(obs)
        action_frequencies[action] = action_frequencies.get(action, 0) + 1
        obs, reward, terminated, truncated, info = env.step(action)
        survival_steps += 1
        reward_total += float(reward)
        food_consumed = float(info.get("food_eaten", 0.0))
        if terminated:
            death_cause = str(info.get("death_cause", "unknown"))
            break
        if truncated:
            break
    return EpisodeResult(
        world_seed=world_seed,
        policy_name=policy.__class__.__name__,
        survival_steps=survival_steps,
        food_consumed=food_consumed,
        death_cause=death_cause,
        terminated=terminated,
        truncated=truncated,
        action_frequencies=action_frequencies,
        reward_total=reward_total,
    )


def run_evaluation(
    env: SurvivalEnv,
    policy: BasePolicy,
    eval_seeds: list[int],
    policy_seed: int = 0,
) -> EvaluationResult:
    """Evaluate policy across multiple world seeds. Derive per-episode policy seed as policy_seed * 10000 + seed."""
    results = []
    for seed in eval_seeds:
        episode_policy_seed = policy_seed * 10000 + seed
        results.append(run_episode(env, policy, seed, episode_policy_seed))
    survival = [r.survival_steps for r in results]
    foods = [r.food_consumed for r in results]
    death_causes: dict[str, int] = {}
    for r in results:
        c = r.death_cause or "none"
        c_str = str(c)
        death_causes[c_str] = death_causes.get(c_str, 0) + 1
    total_af: dict[int, int] = {int(a): 0 for a in range(ACTION_COUNT)}
    for r in results:
        for a, count in r.action_frequencies.items():
            a_int = int(a)
            c_int = int(count)
            current = total_af.get(a_int, 0)
            total_af[a_int] = current + c_int
    n_term = sum(1 for r in results if r.terminated)
    n_trunc = sum(1 for r in results if r.truncated)
    n = len(results)
    return EvaluationResult(
        environment_version=ENVIRONMENT_VERSION,
        observation_dim=OBSERVATION_DIM,
        action_count=ACTION_COUNT,
        policy_name=results[0].policy_name if results else policy.__class__.__name__,
        eval_seeds=list(eval_seeds),
        num_episodes=n,
        mean_survival_steps=float(statistics.mean(survival)) if survival else 0.0,
        median_survival_steps=float(statistics.median(survival)) if survival else 0.0,
        min_survival=min(survival) if survival else 0,
        max_survival=max(survival) if survival else 0,
        mean_food_consumed=float(statistics.mean(foods)) if foods else 0.0,
        death_causes=death_causes,
        termination_rate=n_term / n if n else 0.0,
        truncation_rate=n_trunc / n if n else 0.0,
        action_frequencies=total_af,
        mean_reward=float(statistics.mean([r.reward_total for r in results])) if results else 0.0,
        config_metadata={"max_episode_steps": MAX_EPISODE_STEPS, "policy_seed": policy_seed},
    )


def save_results(result: EvaluationResult, path: str | Path) -> None:
    """Save EvaluationResult to JSON file, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json(), encoding="utf-8")
