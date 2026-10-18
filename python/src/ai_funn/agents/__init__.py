"""Agent policies for AI-Funn-game."""

from .baselines import (
    AlwaysRestPolicy,
    BasePolicy,
    EatWhenAvailablePolicy,
    RandomPolicy,
    SimpleWalkerPolicy,
)

__all__ = [
    "AlwaysRestPolicy",
    "BasePolicy",
    "EatWhenAvailablePolicy",
    "RandomPolicy",
    "SimpleWalkerPolicy",
]
