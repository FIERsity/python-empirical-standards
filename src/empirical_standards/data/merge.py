"""Validated merges that make key relationships and unmatched rows explicit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

MergeRelationship = Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
MergeHow = Literal["left", "right", "inner", "outer"]


@dataclass(frozen=True)
class MergeReport:
    """Counts and key samples describing a validated merge."""

    relationship: MergeRelationship
    how: MergeHow
    keys: tuple[str, ...]
    left_rows: int
    right_rows: int
    result_rows: int
    matched_rows: int
    left_only_rows: int
    right_only_rows: int
    left_only_keys: pd.DataFrame
    right_only_keys: pd.DataFrame


@dataclass(frozen=True)
class MergeResult:
    data: pd.DataFrame
    report: MergeReport


def merge_validated(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str | list[str] | tuple[str, ...],
    relationship: MergeRelationship,
    how: MergeHow = "left",
    require_all_left: bool = False,
    require_all_right: bool = False,
    key_sample_size: int = 10,
) -> MergeResult:
    """Merge two frames after declaring cardinality and unmatched-key requirements."""
    keys = (on,) if isinstance(on, str) else tuple(on)
    if not keys:
        raise ValueError("on must contain at least one key")
    if relationship not in {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}:
        raise ValueError("unsupported merge relationship")
    if how not in {"left", "right", "inner", "outer"}:
        raise ValueError("unsupported merge method")
    if key_sample_size < 0:
        raise ValueError("key_sample_size must be non-negative")
    missing_left = [key for key in keys if key not in left]
    missing_right = [key for key in keys if key not in right]
    if missing_left or missing_right:
        raise KeyError(f"merge keys missing; left={missing_left}, right={missing_right}")
    if left[list(keys)].isna().any().any() or right[list(keys)].isna().any().any():
        raise ValueError("merge keys must not contain missing values")

    audited = left.merge(
        right,
        on=list(keys),
        how="outer",
        validate=relationship,
        indicator="__merge_status",
        suffixes=("_left", "_right"),
    )
    left_only = audited.loc[audited["__merge_status"] == "left_only", list(keys)]
    right_only = audited.loc[audited["__merge_status"] == "right_only", list(keys)]
    if require_all_left and not left_only.empty:
        raise ValueError(f"{len(left_only)} left rows have no match")
    if require_all_right and not right_only.empty:
        raise ValueError(f"{len(right_only)} right rows have no match")

    result = left.merge(
        right,
        on=list(keys),
        how=how,
        validate=relationship,
        suffixes=("_left", "_right"),
    )
    status = audited["__merge_status"].value_counts()
    report = MergeReport(
        relationship=relationship,
        how=how,
        keys=keys,
        left_rows=len(left),
        right_rows=len(right),
        result_rows=len(result),
        matched_rows=int(status.get("both", 0)),
        left_only_rows=int(status.get("left_only", 0)),
        right_only_rows=int(status.get("right_only", 0)),
        left_only_keys=left_only.drop_duplicates().head(key_sample_size).reset_index(drop=True),
        right_only_keys=right_only.drop_duplicates().head(key_sample_size).reset_index(drop=True),
    )
    return MergeResult(result, report)
