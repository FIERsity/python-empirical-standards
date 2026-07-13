"""Pre-specified subgroup analysis for panel models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from empirical_standards.panel.fixed_effects import (
    FixedEffectsResult,
    PanelCovariance,
    fit_fixed_effects,
)


@dataclass(frozen=True)
class HeterogeneityResult:
    estimates: pd.DataFrame
    models: dict[Any, FixedEffectsResult]
    group: str


def fit_fe_by_group(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str] | tuple[str, ...],
    *,
    entity: str,
    time: str,
    group: str,
    entity_effects: bool = True,
    time_effects: bool = True,
    covariance: PanelCovariance = "cluster_entity",
) -> HeterogeneityResult:
    """Repeat the same pre-specified FE model within each subgroup."""
    if group not in data:
        raise KeyError(f"group column {group!r} not found")
    if data.groupby(entity)[group].nunique(dropna=False).max() > 1:
        raise ValueError("group must be time-invariant within entity")
    models: dict[Any, FixedEffectsResult] = {}
    tables: list[pd.DataFrame] = []
    for value, subset in data.groupby(group, dropna=False, sort=True):
        result = fit_fixed_effects(
            subset,
            outcome,
            predictors,
            entity=entity,
            time=time,
            entity_effects=entity_effects,
            time_effects=time_effects,
            covariance=covariance,
        )
        models[value] = result
        table = result.tidy()
        table.insert(0, "group_value", value)
        tables.append(table)
    return HeterogeneityResult(pd.concat(tables, ignore_index=True), models, group)
