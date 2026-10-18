"""Training harness for AI-Funn-game."""

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
    "run_episode",
    "run_evaluation",
    "save_results",
]
