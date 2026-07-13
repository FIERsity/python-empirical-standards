"""Shared result metadata and reproducibility helpers."""

from empirical_standards.results.effects import EFFECT_COLUMNS, effect_data
from empirical_standards.results.metadata import (
    ModelMetadata,
    ModelSpec,
    Provenance,
    SampleInfo,
    build_metadata,
)

__all__ = [
    "EFFECT_COLUMNS",
    "ModelMetadata",
    "ModelSpec",
    "Provenance",
    "SampleInfo",
    "build_metadata",
    "effect_data",
]
