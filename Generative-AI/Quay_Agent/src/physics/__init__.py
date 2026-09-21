"""Deterministic physics layer.

Nothing in this package calls a language model. Every function here is a pure
computation with a checkable answer, which is what lets the agent layer above it
be judged on accuracy rather than on taste.
"""

from src.physics.model import TFIMSpec

__all__ = ["TFIMSpec"]
