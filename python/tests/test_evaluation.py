"""Tests for the deterministic evaluation runner in AI-Funn-game."""

from __future__ import annotations

import json

from ai_funn.agents.baselines import (
    AlwaysRestPolicy,
    EatWhenAvailablePolicy,
    RandomPolicy,
    SimpleWalkerPolicy,
)
from ai_funn.environments.survival_env import SurvivalEnv
from ai_funn.training.evaluation import (
    EpisodeResult,
    EvaluationResult,
    run_episode,
    run_evaluation,
)


def test_run_episode_returns_valid_result() -> None:
    """Run one episode; check EpisodeResult fields."""
    env = SurvivalEnv()
    policy = RandomPolicy(seed=42)
    result = run_episode(env, policy, world_seed=42, policy_seed=0)
    assert isinstance(result, EpisodeResult)
    assert result.world_seed == 42
    assert result.policy_name == "RandomPolicy"
    assert result.survival_steps > 0
    assert result.food_consumed >= 0.0
    assert isinstance(result.death_cause, (str, type(None)))
    assert isinstance(result.terminated, bool)
    assert isinstance(result.truncated, bool)
    assert isinstance(result.action_frequencies, dict)
    assert isinstance(result.reward_total, float)
    env.close()


def test_run_evaluation_structure() -> None:
    """Run 5 episodes; check EvaluationResult has all fields."""
    env = SurvivalEnv()
    policy = RandomPolicy(seed=42)
    result = run_evaluation(env, policy, eval_seeds=[1, 2, 3, 4, 5])
    assert isinstance(result, EvaluationResult)
    assert result.environment_version == "survival-v1"
    assert result.observation_dim == 199
    assert result.action_count == 6
    assert result.policy_name == "RandomPolicy"
    assert result.eval_seeds == [1, 2, 3, 4, 5]
    assert result.num_episodes == 5
    assert isinstance(result.mean_survival_steps, float)
    assert isinstance(result.median_survival_steps, float)
    assert isinstance(result.min_survival, int)
    assert isinstance(result.max_survival, int)
    assert isinstance(result.mean_food_consumed, float)
    assert isinstance(result.death_causes, dict)
    assert isinstance(result.termination_rate, float)
    assert isinstance(result.truncation_rate, float)
    assert isinstance(result.action_frequencies, dict)
    assert isinstance(result.mean_reward, float)
    assert isinstance(result.config_metadata, dict)
    env.close()


def test_evaluation_reproducibility() -> None:
    """Same seeds/policy produce identical mean_survival_steps."""
    env1 = SurvivalEnv()
    env2 = SurvivalEnv()
    policy1 = RandomPolicy(seed=42)
    policy2 = RandomPolicy(seed=42)
    seeds = [10, 20, 30]
    r1 = run_evaluation(env1, policy1, seeds)
    r2 = run_evaluation(env2, policy2, seeds)
    assert r1.mean_survival_steps == r2.mean_survival_steps
    assert r1.to_json() == r2.to_json()
    env1.close()
    env2.close()


def test_identical_seeds_across_policies() -> None:
    """Same eval_seeds list used for all 4 policies."""
    seeds = [1, 2, 3, 4, 5]
    env = SurvivalEnv()
    policies = [
        RandomPolicy(seed=42),
        AlwaysRestPolicy(seed=42),
        EatWhenAvailablePolicy(seed=42),
        SimpleWalkerPolicy(seed=42),
    ]
    results = [run_evaluation(env, p, seeds) for p in policies]
    for r in results:
        assert r.eval_seeds == seeds
    env.close()


def test_death_and_truncation_accounting() -> None:
    """terminated + truncated accounting makes sense."""
    env = SurvivalEnv()
    policy = AlwaysRestPolicy(seed=42)
    result = run_evaluation(env, policy, eval_seeds=[1, 2, 3, 4, 5])
    assert result.num_episodes > 0
    n_term = int(result.termination_rate * result.num_episodes)
    n_trunc = int(result.truncation_rate * result.num_episodes)
    assert n_term + n_trunc <= result.num_episodes
    env.close()


def test_result_json_serialization() -> None:
    """result.to_json() is valid JSON and round-trips."""
    env = SurvivalEnv()
    policy = RandomPolicy(seed=42)
    result = run_evaluation(env, policy, eval_seeds=[1, 2])
    s = result.to_json()
    parsed = json.loads(s)
    assert isinstance(parsed, dict)
    assert parsed["policy_name"] == "RandomPolicy"
    assert parsed["num_episodes"] == 2
    d = result.to_dict()
    assert d["policy_name"] == "RandomPolicy"
    env.close()


def test_no_policy_state_leakage() -> None:
    """Policy reset between episodes produces deterministic results."""
    env = SurvivalEnv()
    policy = RandomPolicy(seed=42)
    seeds = [1, 2, 3]
    r1 = run_evaluation(env, policy, seeds)
    r2 = run_evaluation(env, policy, seeds)
    assert r1.to_json() == r2.to_json()
    env.close()


def test_action_frequencies_sum() -> None:
    """Sum of action frequencies equals total steps."""
    env = SurvivalEnv()
    policy = RandomPolicy(seed=42)
    seeds = [1, 2, 3, 4, 5]
    total_steps = 0
    for s in seeds:
        ep = run_episode(env, policy, world_seed=s, policy_seed=s)
        total_steps += ep.survival_steps
    result = run_evaluation(env, policy, eval_seeds=seeds)
    total_actions = sum(result.action_frequencies.values())
    assert total_actions == total_steps
    env.close()
