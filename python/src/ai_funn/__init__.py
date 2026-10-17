"""AI-Funn-game: a local-first artificial-life experiment.

This package hosts the simulation core, Gymnasium environments,
agent policies, training harnesses and experiment orchestration for
the AI-Funn-game project.  The simulation core is intentionally free of
ML, UI, API or storage imports so it can be unit-tested in isolation.
"""

__version__ = "0.1.0"

__all__ = [
    "simulation",
    "environments",
    "agents",
    "training",
    "experiments",
]
