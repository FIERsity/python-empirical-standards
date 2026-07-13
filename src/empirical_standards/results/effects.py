"""Standard long-form outputs for coefficients and plot-ready effect data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

EFFECT_COLUMNS = (
    "estimand",
    "term",
    "estimate",
    "std_error",
    "statistic",
    "p_value",
    "conf_low",
    "conf_high",
)


def effect_data(
    table: pd.DataFrame,
    *,
    estimand: str,
    term: str = "term",
    x: str | None = None,
    support_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Normalize an estimator table without drawing or choosing a visual style."""
    aliases = {"att": "estimate", "standard_error": "std_error"}
    normalized = table.rename(
        columns={old: new for old, new in aliases.items() if old in table and new not in table}
    ).copy()
    if "estimate" not in normalized:
        raise ValueError("effect table is missing required column: estimate")
    if term not in normalized:
        if x is None or x not in normalized:
            normalized[term] = estimand
        else:
            normalized[term] = normalized[x].map(lambda value: f"{estimand}[{value}]")
    normalized.insert(0, "estimand", estimand)
    if term != "term":
        normalized = normalized.rename(columns={term: "term"})
    numeric = [
        column
        for column in ("estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high")
        if column in normalized
    ]
    for column in numeric:
        original = normalized[column]
        values = pd.to_numeric(original, errors="coerce")
        if (values.isna() & original.notna()).any():
            raise ValueError(f"effect column {column!r} must be numeric")
        finite = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f"effect column {column!r} must contain finite values")
        normalized[column] = values
    ordered = [column for column in EFFECT_COLUMNS if column in normalized]
    if x is not None and x in normalized and x not in ordered:
        ordered.insert(2, x)
    for column in support_columns:
        if column in normalized and column not in ordered:
            ordered.append(column)
    remaining = [column for column in normalized if column not in ordered]
    return normalized.loc[:, [*ordered, *remaining]].reset_index(drop=True)
