"""Training harness for AI-Funn-game."""

from __future__ import annotations

from . import manifest
from .evaluation import (
    EpisodeResult,
    EvaluationResult,
    run_episode,
    run_evaluation,
    save_results,
)

__all__ = [
    "EpisodeResult",
    "EvaluationResult",
    "manifest",
    "run_episode",
    "run_evaluation",
    "save_results",
]
