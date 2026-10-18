"""Tests for baseline policies in AI-Funn-game."""

from __future__ import annotations

import numpy as np

from ai_funn.agents.baselines import (
    AlwaysRestPolicy,
    BasePolicy,
    EatWhenAvailablePolicy,
    RandomPolicy,
    SimpleWalkerPolicy,
)


def test_random_policy_action_bounds() -> None:
    """RandomPolicy(seed=42) actions are always in [0, 6)."""
    policy = RandomPolicy(seed=42)
    obs = np.zeros(199, dtype=np.float32)
    for _ in range(100):
        action = policy.act(obs)
        assert 0 <= action < 6


def test_always_rest_policy() -> None:
    """AlwaysRestPolicy always returns action 5."""
    policy = AlwaysRestPolicy(seed=42)
    obs = np.zeros(199, dtype=np.float32)
    for _ in range(100):
        assert policy.act(obs) == 5


def test_eat_policy_eats_when_food() -> None:
    """EatWhenAvailablePolicy returns 4 when obs[73] == 1.0."""
    policy = EatWhenAvailablePolicy(seed=42)
    obs = np.zeros(199, dtype=np.float32)
    obs[73] = 1.0
    assert policy.act(obs) == 4


def test_eat_policy_explores_without_food() -> None:
    """EatWhenAvailablePolicy returns action in [0, 4) when obs all zeros."""
    policy = EatWhenAvailablePolicy(seed=42)
    obs = np.zeros(199, dtype=np.float32)
    for _ in range(100):
        action = policy.act(obs)
        assert 0 <= action < 4


def test_simple_walker_only_moves() -> None:
    """SimpleWalkerPolicy actions are always in [0, 4)."""
    policy = SimpleWalkerPolicy(seed=42)
    obs = np.zeros(199, dtype=np.float32)
    for _ in range(100):
        action = policy.act(obs)
        assert 0 <= action < 4


def test_simple_walker_reproducible() -> None:
    """Two SimpleWalkerPolicy(seed=7) produce identical action sequences."""
    policy1 = SimpleWalkerPolicy(seed=7)
    policy2 = SimpleWalkerPolicy(seed=7)
    obs = np.zeros(199, dtype=np.float32)
    for _ in range(50):
        a1 = policy1.act(obs)
        a2 = policy2.act(obs)
        assert a1 == a2


def test_random_policy_deterministic() -> None:
    """Two RandomPolicy(seed=42) produce identical action sequences."""
    policy1 = RandomPolicy(seed=42)
    policy2 = RandomPolicy(seed=42)
    obs = np.zeros(199, dtype=np.float32)
    for _ in range(50):
        a1 = policy1.act(obs)
        a2 = policy2.act(obs)
        assert a1 == a2


def test_policy_reset_changes_behavior() -> None:
    """Policy reset with different seed produces different actions."""
    policy = RandomPolicy(seed=42)
    obs = np.zeros(199, dtype=np.float32)
    actions_before = [policy.act(obs) for _ in range(10)]
    policy.reset(seed=99)
    actions_after = [policy.act(obs) for _ in range(10)]
    assert actions_before != actions_after


def test_all_policies_valid_actions() -> None:
    """For each policy, 100 actions are all in [0, 6)."""
    policies: list[BasePolicy] = [
        RandomPolicy(seed=1),
        AlwaysRestPolicy(seed=1),
        EatWhenAvailablePolicy(seed=1),
        SimpleWalkerPolicy(seed=1),
    ]
    obs = np.zeros(199, dtype=np.float32)
    for policy in policies:
        for _ in range(100):
            action = policy.act(obs)
            assert 0 <= action < 6, f"{policy.__class__.__name__} produced invalid action {action}"
