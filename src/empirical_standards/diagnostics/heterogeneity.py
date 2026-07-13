"""Pre-specified subgroup analysis for panel models."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

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


@dataclass(frozen=True)
class InteractionHeterogeneityResult:
    group_effects: pd.DataFrame
    interaction_terms: pd.DataFrame
    joint_statistic: float
    joint_p_value: float
    reference_group: Any
    model: FixedEffectsResult

    def tidy(self) -> pd.DataFrame:
        return self.group_effects.copy()

    def glance(self) -> pd.Series:
        return pd.Series(
            {
                "estimator": "fe_heterogeneity",
                "nobs": self.model.nobs,
                "groups": len(self.group_effects),
                "reference_group": self.reference_group,
                "joint_statistic": self.joint_statistic,
                "joint_p_value": self.joint_p_value,
                "covariance": self.model.covariance,
            }
        )

    def model_spec(self) -> dict[str, Any]:
        spec = self.model.model_spec()
        spec["estimator"] = "fe_heterogeneity"
        spec["settings"] = {**spec["settings"], "reference_group": self.reference_group}
        return spec

    def sample_info(self) -> dict[str, Any]:
        return self.model.sample_info()

    def provenance(self) -> dict[str, Any]:
        return self.model.provenance()


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


def fit_fe_heterogeneity(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    *,
    entity: str,
    time: str,
    group: str,
    reference_group: Any | None = None,
    controls: list[str] | tuple[str, ...] = (),
    entity_effects: bool = True,
    time_effects: bool = True,
    covariance: PanelCovariance = "cluster_entity",
    confidence_level: float = 0.95,
) -> InteractionHeterogeneityResult:
    """Estimate categorical group effects and formally test equality across groups."""
    required = [entity, time, group, treatment, outcome, *controls]
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    if data[group].isna().any():
        raise ValueError("group must not contain missing values")
    if data.groupby(entity)[group].nunique().max() > 1:
        raise ValueError("group must be time-invariant within entity")
    groups = list(pd.unique(data[group]))
    with suppress(TypeError):
        groups = sorted(groups)
    if len(groups) < 2:
        raise ValueError("heterogeneity analysis requires at least two groups")
    reference = groups[0] if reference_group is None else reference_group
    if reference not in groups:
        raise ValueError("reference_group is not present in the data")
    sample = data.copy()
    interactions: list[tuple[Any, str]] = []
    nonreference = [value for value in groups if value != reference]
    for index, value in enumerate(nonreference):
        term = f"__heterogeneity_{index}"
        sample[term] = sample[treatment].astype(float) * (sample[group] == value).astype(float)
        interactions.append((value, term))
    interaction_terms = [term for _, term in interactions]
    model = fit_fixed_effects(
        sample,
        outcome,
        [treatment, *interaction_terms, *controls],
        entity=entity,
        time=time,
        entity_effects=entity_effects,
        time_effects=time_effects,
        covariance=covariance,
    )
    covariance_matrix = pd.DataFrame(model.raw_result.cov)
    covariance_values = covariance_matrix.to_numpy(dtype=float)
    covariance_index = covariance_matrix.index

    def covariance_value(left: str, right: str) -> float:
        return float(
            covariance_values[covariance_index.get_loc(left), covariance_index.get_loc(right)]
        )

    critical = float(norm.ppf(0.5 + confidence_level / 2))
    base = float(model.coefficients[treatment])
    base_variance = covariance_value(treatment, treatment)
    interaction_map = dict(interactions)
    rows: list[dict[str, Any]] = []
    for value in groups:
        if value == reference:
            estimate, variance = base, base_variance
        else:
            term = interaction_map[value]
            estimate = base + float(model.coefficients[term])
            variance = (
                base_variance + covariance_value(term, term) + 2 * covariance_value(treatment, term)
            )
        se = float(np.sqrt(max(variance, 0)))
        rows.append(
            {
                "group": value,
                "estimate": estimate,
                "std_error": se,
                "statistic": estimate / se,
                "p_value": float(2 * norm.sf(abs(estimate / se))),
                "conf_low": estimate - critical * se,
                "conf_high": estimate + critical * se,
                "is_reference": value == reference,
            }
        )
    restriction = np.zeros((len(interaction_terms), len(model.coefficients)))
    for row, term in enumerate(interaction_terms):
        restriction[row, model.coefficients.index.get_loc(term)] = 1
    test = model.raw_result.wald_test(restriction)
    interaction_table = model.tidy().set_index("term").loc[interaction_terms].reset_index()
    interaction_table.insert(0, "group", nonreference)
    return InteractionHeterogeneityResult(
        pd.DataFrame(rows), interaction_table, float(test.stat), float(test.pval), reference, model
    )
