"""Panel structure and estimability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelDiagnostics:
    """A transparent snapshot of panel coverage and variable variation."""

    entity: str
    time: str
    rows: int
    entities: int
    periods: int
    duplicate_key_rows: int
    missing_key_rows: int
    balanced: bool
    expected_balanced_rows: int
    coverage_rate: float
    observation_counts: pd.Series
    singleton_entities: tuple[object, ...]
    time_coverage: pd.DataFrame
    variable_variation: pd.DataFrame
    absorbed_by_entity: tuple[str, ...]
    absorbed_by_time: tuple[str, ...]

    def summary(self) -> pd.Series:
        """Return model-level diagnostics in a compact labeled series."""
        return pd.Series(
            {
                "rows": self.rows,
                "entities": self.entities,
                "periods": self.periods,
                "duplicate_key_rows": self.duplicate_key_rows,
                "missing_key_rows": self.missing_key_rows,
                "balanced": self.balanced,
                "expected_balanced_rows": self.expected_balanced_rows,
                "coverage_rate": self.coverage_rate,
                "singleton_entities": len(self.singleton_entities),
            }
        )


def diagnose_panel(
    data: pd.DataFrame,
    *,
    entity: str,
    time: str,
    variables: list[str] | tuple[str, ...] = (),
) -> PanelDiagnostics:
    """Diagnose panel keys, coverage, singletons, and within/between variation."""
    required = [entity, time, *variables]
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    duplicate_rows = int(data.duplicated([entity, time], keep=False).sum())
    missing_keys = int(data[[entity, time]].isna().any(axis=1).sum())
    valid = data.loc[~data[[entity, time]].isna().any(axis=1)].copy()
    entities = int(valid[entity].nunique())
    periods = int(valid[time].nunique())
    expected = entities * periods
    unique_cells = len(valid.drop_duplicates([entity, time]))
    coverage = float(unique_cells / expected) if expected else float("nan")
    counts = valid.groupby(entity, sort=True).size().rename("observations")
    singletons = tuple(counts[counts == 1].index.tolist())
    time_coverage = (
        valid.groupby(time, sort=True)
        .agg(observations=(entity, "size"), entities=(entity, "nunique"))
        .reset_index()
    )
    if entities:
        time_coverage["entity_coverage_rate"] = time_coverage["entities"] / entities
    else:
        time_coverage["entity_coverage_rate"] = np.nan

    rows: list[dict[str, object]] = []
    absorbed_entity: list[str] = []
    absorbed_time: list[str] = []
    for variable in variables:
        if not pd.api.types.is_numeric_dtype(valid[variable]):
            raise TypeError(f"column {variable!r} must be numeric")
        values = valid[[entity, time, variable]].dropna()
        overall_variance = float(values[variable].var(ddof=1))
        entity_means = values.groupby(entity)[variable].transform("mean")
        time_means = values.groupby(time)[variable].transform("mean")
        within_entity = float((values[variable] - entity_means).var(ddof=1))
        within_time = float((values[variable] - time_means).var(ddof=1))
        between_entity = float(values.groupby(entity)[variable].mean().var(ddof=1))
        between_time = float(values.groupby(time)[variable].mean().var(ddof=1))
        entity_absorbed = bool(np.isclose(within_entity, 0.0, atol=1e-14))
        time_absorbed = bool(np.isclose(within_time, 0.0, atol=1e-14))
        if entity_absorbed:
            absorbed_entity.append(variable)
        if time_absorbed:
            absorbed_time.append(variable)
        rows.append(
            {
                "variable": variable,
                "non_missing": len(values),
                "missing": len(valid) - len(values),
                "overall_variance": overall_variance,
                "within_entity_variance": within_entity,
                "between_entity_variance": between_entity,
                "within_time_variance": within_time,
                "between_time_variance": between_time,
                "absorbed_by_entity": entity_absorbed,
                "absorbed_by_time": time_absorbed,
            }
        )
    variation = pd.DataFrame(rows)
    return PanelDiagnostics(
        entity=entity,
        time=time,
        rows=len(data),
        entities=entities,
        periods=periods,
        duplicate_key_rows=duplicate_rows,
        missing_key_rows=missing_keys,
        balanced=bool(duplicate_rows == 0 and missing_keys == 0 and unique_cells == expected),
        expected_balanced_rows=expected,
        coverage_rate=coverage,
        observation_counts=counts,
        singleton_entities=singletons,
        time_coverage=time_coverage,
        variable_variation=variation,
        absorbed_by_entity=tuple(absorbed_entity),
        absorbed_by_time=tuple(absorbed_time),
    )
