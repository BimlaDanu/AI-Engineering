"""🔬 AI/ML Lab — hands-on, offline ML/DL/NLP exercises on bundled toy datasets.

Framework-agnostic (never imports Streamlit): bundled datasets, guided parameterised recipes,
an LLM code-seeder, and a restricted in-process executor for the opt-in advanced cell. The
Streamlit surface lives in ``src/ui/pages/ml_lab.py``.
"""

from __future__ import annotations

from src.lab.curriculum import (
    TRACKS,
    Lesson,
    Practice,
    Track,
    get_track,
    track_names,
    validate_curriculum,
)
from src.lab.datasets import DATASETS, LabDataset, LoadedData, datasets_for_topic
from src.lab.recipes import RECIPES, LabResult, ParamSpec, Recipe, recipes_for_topic
from src.lab.sandbox import ALLOWED_MODULES, SandboxResult, run_sandboxed
from src.lab.schemas import SeededSnippet
from src.lab.seed import seed_snippet

__all__ = [
    "ALLOWED_MODULES",
    "DATASETS",
    "RECIPES",
    "TRACKS",
    "LabDataset",
    "LabResult",
    "Lesson",
    "LoadedData",
    "ParamSpec",
    "Practice",
    "Recipe",
    "SandboxResult",
    "SeededSnippet",
    "Track",
    "datasets_for_topic",
    "get_track",
    "recipes_for_topic",
    "run_sandboxed",
    "seed_snippet",
    "track_names",
    "validate_curriculum",
]
