"""Production wild-cluster inference backed by R ``fwildclusterboot``."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from empirical_standards.backends.r import run_r_backend
from empirical_standards.results import build_metadata

RWildWeight = Literal["rademacher", "mammen", "webb", "normal"]


@dataclass(frozen=True)
class RWildClusterBootstrapResult:
    coefficient: str
    estimate: float
    bootstrap_p_value: float
    conf_low: float
    conf_high: float
    metadata: dict[str, object]

    def tidy(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "term": [self.coefficient],
                "estimate": [self.estimate],
                "bootstrap_p_value": [self.bootstrap_p_value],
                "conf_low": [self.conf_low],
                "conf_high": [self.conf_high],
            }
        )

    def glance(self) -> pd.Series:
        return pd.Series(self.metadata)

    def model_spec(self) -> dict[str, object]:
        return dict(cast(dict[str, object], self.metadata["specification"]))

    def sample_info(self) -> dict[str, object]:
        return dict(cast(dict[str, object], self.metadata["sample"]))

    def provenance(self) -> dict[str, object]:
        return {
            key: self.metadata[key]
            for key in ("backend", "package", "r_version", "package_versions", "script")
        }


def _script() -> Path:
    return Path(str(files("empirical_standards.backends.r_scripts").joinpath("wild_cluster.R")))


def wild_cluster_bootstrap_fe_r(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str] | tuple[str, ...],
    *,
    coefficient: str,
    fixed_effects: list[str] | tuple[str, ...],
    cluster: str,
    replications: int = 9999,
    weight_distribution: RWildWeight = "rademacher",
    impose_null: bool = True,
    null_value: float = 0.0,
    confidence_level: float = 0.95,
    random_state: int = 0,
) -> RWildClusterBootstrapResult:
    """Run null-imposed wild-cluster bootstrap-t inference through R."""
    terms, effects = tuple(predictors), tuple(fixed_effects)
    if coefficient not in terms:
        raise ValueError("coefficient must be included in predictors")
    if not effects:
        raise ValueError("fixed_effects must be non-empty")
    if replications < 100:
        raise ValueError("replications must be at least 100")
    if weight_distribution not in {"rademacher", "mammen", "webb", "normal"}:
        raise ValueError("unsupported wild weight distribution")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    required = list(dict.fromkeys([outcome, *terms, *effects, cluster]))
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    sample = data.loc[:, required].copy()
    if sample.isna().any().any():
        raise ValueError("R wild-cluster bootstrap currently requires complete data")
    if any(not pd.api.types.is_numeric_dtype(sample[column]) for column in [outcome, *terms]):
        raise TypeError("outcome and predictors must be numeric")
    if not np.isfinite(sample[[outcome, *terms]].to_numpy(dtype=float)).all():
        raise ValueError("model columns must contain only finite values")
    if sample[cluster].nunique() < 2:
        raise ValueError("wild-cluster bootstrap requires at least two clusters")
    specification: dict[str, object] = {
        "outcome": outcome,
        "predictors": list(terms),
        "coefficient": coefficient,
        "fixed_effects": list(effects),
        "cluster": cluster,
        "replications": replications,
        "weight_distribution": weight_distribution,
        "impose_null": impose_null,
        "null_value": null_value,
        "confidence_level": confidence_level,
        "random_state": random_state,
    }
    metadata, _ = run_r_backend(
        sample,
        specification,
        script=_script(),
        required_packages=("fixest", "fwildclusterboot", "jsonlite"),
    )
    shared = build_metadata(
        estimator="r_wild_cluster_bootstrap_fe",
        outcome=outcome,
        predictors=terms,
        settings=specification,
        sample=sample,
        original_nobs=len(data),
    )
    metadata["specification"] = specification
    metadata["sample"] = shared.sample.to_dict()
    metadata["python_provenance"] = shared.provenance.to_dict()
    return RWildClusterBootstrapResult(
        coefficient,
        float(cast(float, metadata["estimate"])),
        float(cast(float, metadata["bootstrap_p_value"])),
        float(cast(float, metadata["conf_low"])),
        float(cast(float, metadata["conf_high"])),
        metadata,
    )
