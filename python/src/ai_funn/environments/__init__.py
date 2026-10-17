"""Gymnasium environments for AI-Funn-game.

This subpackage wraps the simulation core in a stable Gymnasium
contract: observation float32[199], action Discrete(6), explicit
terminal/truncation handling and a fixed info schema.
"""

from .observation import encode_observation
from .survival_env import SurvivalEnv

__all__ = [
    "SurvivalEnv",
    "encode_observation",
]
