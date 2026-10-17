"""Smoke tests for the ai_funn package bootstrap.

These tests verify that the package and its declared subpackages can be
imported on a fresh Windows installation and that the declared Python
version floor is respected.
"""

import importlib
import sys

import ai_funn


def test_package_imports():
    """The top-level package imports and reports the expected version."""
    assert ai_funn.__version__ == "0.1.0"


def test_subpackages_exist():
    """All declared subpackages are importable as modules."""
    expected = [
        "ai_funn.simulation",
        "ai_funn.environments",
        "ai_funn.agents",
        "ai_funn.training",
        "ai_funn.experiments",
    ]
    for name in expected:
        module = importlib.import_module(name)
        assert module is not None


def test_python_version():
    """The supported Python version floor is met."""
    assert sys.version_info >= (3, 11)