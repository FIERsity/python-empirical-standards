"""Panel 2SLS with explicit fixed-effect indicator absorption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyhdfe

from empirical_standards.models.iv import IV2SLSResult, IVCovariance, fit_iv_2sls


@dataclass(frozen=True)
class PanelIV2SLSResult:
    iv_result: IV2SLSResult
    entity: str
    time: str
    entity_effects: bool
    time_effects: bool
    absorbed_indicator_count: int
    absorbed_degrees: int
    absorption_method: Literal["indicators", "within"]
    entities: int
    periods: int

    @property
    def first_stage(self) -> pd.DataFrame:
        return self.iv_result.first_stage

    def tidy(self) -> pd.DataFrame:
        structural_terms = [*self.iv_result.exogenous, *self.iv_result.endogenous]
        if self.iv_result.add_intercept:
            structural_terms.insert(0, "const")
        return self.iv_result.tidy().set_index("term").loc[structural_terms].reset_index()

    def glance(self) -> pd.Series:
        result = self.iv_result.glance().copy()
        result["estimator"] = "panel_iv_2sls"
        result["entities"] = self.entities
        result["periods"] = self.periods
        result["entity_effects"] = self.entity_effects
        result["time_effects"] = self.time_effects
        result["absorbed_indicator_count"] = self.absorbed_indicator_count
        result["absorbed_degrees"] = self.absorbed_degrees
        result["absorption_method"] = self.absorption_method
        return result

    def model_spec(self) -> dict[str, Any]:
        spec = self.iv_result.model_spec()
        spec["estimator"] = "panel_iv_2sls"
        spec["predictors"] = (*self.iv_result.exogenous, *self.iv_result.endogenous)
        spec["settings"] = {
            **spec["settings"],
            "exogenous": self.iv_result.exogenous,
            "entity": self.entity,
            "time": self.time,
            "entity_effects": self.entity_effects,
            "time_effects": self.time_effects,
            "absorption_implementation": self.absorption_method,
            "absorbed_indicator_count": self.absorbed_indicator_count,
            "absorbed_degrees": self.absorbed_degrees,
            "covariance_correction": (
                "finite_sample_including_indicators"
                if self.absorption_method == "indicators"
                else "asymptotic_after_within_transformation"
            ),
        }
        return spec

    def sample_info(self) -> dict[str, Any]:
        return self.iv_result.sample_info()

    def provenance(self) -> dict[str, Any]:
        return self.iv_result.provenance()


def _add_fixed_effect_indicators(
    data: pd.DataFrame,
    *,
    entity: str,
    time: str,
    entity_effects: bool,
    time_effects: bool,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    sample = data.copy()
    indicator_names: list[str] = []
    for column, enabled, prefix in (
        (entity, entity_effects, "__fe_entity"),
        (time, time_effects, "__fe_time"),
    ):
        if not enabled:
            continue
        indicators = pd.get_dummies(sample[column], prefix=prefix, drop_first=True, dtype=float)
        collisions = set(indicators.columns) & set(sample.columns)
        if collisions:
            raise ValueError(
                f"fixed-effect indicator names collide with data: {sorted(collisions)}"
            )
        sample = pd.concat([sample, indicators], axis=1)
        indicator_names.extend(map(str, indicators.columns))
    return sample, tuple(indicator_names)


def fit_panel_iv_2sls(
    data: pd.DataFrame,
    outcome: str,
    *,
    exogenous: list[str] | tuple[str, ...],
    endogenous: list[str] | tuple[str, ...],
    instruments: list[str] | tuple[str, ...],
    entity: str,
    time: str,
    entity_effects: bool = True,
    time_effects: bool = True,
    covariance: IVCovariance = "cluster",
    cluster: str | None = None,
    drop_missing: bool = False,
    absorption: Literal["indicators", "within"] = "indicators",
) -> PanelIV2SLSResult:
    """Fit panel 2SLS using explicit indicators or scalable HDFE residualization."""
    if not entity_effects and not time_effects:
        raise ValueError("at least one fixed-effect dimension must be enabled")
    for column in (entity, time):
        if column not in data:
            raise KeyError(f"panel key {column!r} not found")
    if data.duplicated([entity, time]).any():
        raise ValueError("entity-time pairs must be unique")
    if data[[entity, time]].isna().any().any():
        raise ValueError("panel keys must not contain missing values")
    if data[entity].nunique() < 2 or data[time].nunique() < 2:
        raise ValueError("panel IV requires at least two entities and two periods")
    if absorption not in {"indicators", "within"}:
        raise ValueError("absorption must be 'indicators' or 'within'")
    actual_cluster = entity if covariance == "cluster" and cluster is None else cluster
    if absorption == "indicators":
        sample, indicators = _add_fixed_effect_indicators(
            data,
            entity=entity,
            time=time,
            entity_effects=entity_effects,
            time_effects=time_effects,
        )
        result = fit_iv_2sls(
            sample,
            outcome,
            exogenous=[*exogenous, *indicators],
            endogenous=endogenous,
            instruments=instruments,
            covariance=covariance,
            cluster=actual_cluster,
            drop_missing=drop_missing,
        )
        absorbed_indicator_count = absorbed_degrees = len(indicators)
    else:
        required = list(
            dict.fromkeys(
                [entity, time, outcome, *exogenous, *endogenous, *instruments]
                + ([actual_cluster] if actual_cluster is not None else [])
            )
        )
        sample = data.loc[:, required].copy()
        missing_rows = sample.isna().any(axis=1)
        if missing_rows.any() and not drop_missing:
            raise ValueError("panel IV columns contain missing values; set drop_missing=True")
        sample = sample.loc[~missing_rows].copy()
        model_columns = [outcome, *exogenous, *endogenous, *instruments]
        if not np.isfinite(sample[model_columns].to_numpy(dtype=float)).all():
            raise ValueError("panel IV model columns must contain only finite values")
        effect_columns = [
            column
            for column, enabled in ((entity, entity_effects), (time, time_effects))
            if enabled
        ]
        algorithm = pyhdfe.create(sample[effect_columns].to_numpy(), drop_singletons=False)
        residualized = algorithm.residualize(sample[model_columns].to_numpy(dtype=float))
        transformed = pd.DataFrame(residualized, columns=model_columns, index=sample.index)
        if actual_cluster is not None:
            transformed[actual_cluster] = sample[actual_cluster]
        result = fit_iv_2sls(
            transformed,
            outcome,
            exogenous=exogenous,
            endogenous=endogenous,
            instruments=instruments,
            add_intercept=False,
            covariance=covariance,
            cluster=actual_cluster,
            drop_missing=False,
            debiased=False,
        )
        indicators = ()
        absorbed_indicator_count = 0
        absorbed_degrees = int(algorithm.degrees)
    # Expose substantive exogenous variables rather than internal indicator columns.
    object.__setattr__(result, "exogenous", tuple(exogenous))
    return PanelIV2SLSResult(
        result,
        entity,
        time,
        entity_effects,
        time_effects,
        absorbed_indicator_count,
        absorbed_degrees,
        absorption,
        int(data[entity].nunique()),
        int(data[time].nunique()),
    )
