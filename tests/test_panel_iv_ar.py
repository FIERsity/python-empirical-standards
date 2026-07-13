from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from empirical_standards import (
    anderson_rubin_confidence_set,
    anderson_rubin_test,
    fit_panel_iv_2sls,
    fit_panel_iv_2sls_r,
)
from empirical_standards.backends import check_r_environment


def make_panel_iv(strength: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(1201)
    entities, periods = 50, 8
    entity = np.repeat(np.arange(entities), periods)
    time = np.tile(np.arange(periods), entities)
    z = rng.normal(size=len(entity))
    x = rng.normal(size=len(entity))
    structural_error = rng.normal(size=len(entity))
    endogenous = strength * z + 0.35 * x + 0.6 * structural_error + rng.normal(size=len(entity))
    y = (
        rng.normal(size=entities)[entity]
        + rng.normal(scale=0.3, size=periods)[time]
        + 0.7 * x
        + 1.8 * endogenous
        + structural_error
    )
    return pd.DataFrame(
        {"id": entity, "time": time, "z": z, "x": x, "endogenous": endogenous, "y": y}
    )


def test_panel_iv_recovers_effect_and_hides_indicators() -> None:
    result = fit_panel_iv_2sls(
        make_panel_iv(),
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z"],
        entity="id",
        time="time",
        time_effects=True,
    )
    assert result.tidy().set_index("term").loc["endogenous", "estimate"] == pytest.approx(
        1.8, abs=0.15
    )
    assert list(result.tidy()["term"]) == ["const", "x", "endogenous"]
    assert result.absorbed_indicator_count == 56
    assert result.glance()["estimator"] == "panel_iv_2sls"
    assert result.model_spec()["settings"]["absorption_implementation"] == "indicators"


def test_scalable_within_absorption_matches_indicator_coefficients() -> None:
    data = make_panel_iv()
    indicators = fit_panel_iv_2sls(
        data,
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z"],
        entity="id",
        time="time",
        time_effects=True,
        covariance="robust",
        absorption="indicators",
    )
    within = fit_panel_iv_2sls(
        data,
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z"],
        entity="id",
        time="time",
        time_effects=True,
        covariance="robust",
        absorption="within",
    )
    left = indicators.tidy().set_index("term").loc[["x", "endogenous"], "estimate"]
    right = within.tidy().set_index("term").loc[["x", "endogenous"], "estimate"]
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-10)
    assert within.absorbed_indicator_count == 0
    assert within.absorbed_degrees == 57
    assert within.model_spec()["settings"]["covariance_correction"] == "asymptotic"


def test_within_absorbed_df_covariance_matches_indicator_backend() -> None:
    data = make_panel_iv()
    indicators = fit_panel_iv_2sls(
        data,
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z"],
        entity="id",
        time="time",
        covariance="unadjusted",
        absorption="indicators",
    )
    within = fit_panel_iv_2sls(
        data,
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z"],
        entity="id",
        time="time",
        covariance="unadjusted",
        absorption="within",
        within_covariance_correction="absorbed_df",
    )
    terms = ["x", "endogenous"]
    indicator_table = indicators.tidy().set_index("term").loc[terms]
    within_table = within.tidy().set_index("term").loc[terms]
    np.testing.assert_allclose(
        indicator_table[["estimate", "std_error"]],
        within_table[["estimate", "std_error"]],
        rtol=0,
        atol=1e-10,
    )
    assert within.absorbed_degrees == 57
    assert within.glance()["within_covariance_correction"] == "absorbed_df"
    assert within.model_spec()["settings"]["covariance_correction"] == "absorbed_df"


def test_absorbed_df_covariance_scope_is_explicit() -> None:
    with pytest.raises(ValueError, match="unadjusted"):
        fit_panel_iv_2sls(
            make_panel_iv(),
            "y",
            exogenous=["x"],
            endogenous=["endogenous"],
            instruments=["z"],
            entity="id",
            time="time",
            covariance="robust",
            absorption="within",
            within_covariance_correction="absorbed_df",
        )


def test_anderson_rubin_true_and_false_nulls() -> None:
    data = make_panel_iv()
    true_null = anderson_rubin_test(
        data,
        "y",
        "endogenous",
        ["z"],
        null_value=1.8,
        exogenous=["x"],
        fixed_effects=["id", "time"],
        covariance="cluster",
        cluster="id",
    )
    false_null = anderson_rubin_test(
        data,
        "y",
        "endogenous",
        ["z"],
        null_value=0.0,
        exogenous=["x"],
        fixed_effects=["id", "time"],
        covariance="cluster",
        cluster="id",
    )
    assert true_null.p_value > 0.05
    assert false_null.p_value < 0.01
    assert true_null.instrument_count == 1


def test_anderson_rubin_grid_inversion() -> None:
    data = make_panel_iv(0.15)
    result = anderson_rubin_confidence_set(
        data,
        "y",
        "endogenous",
        ["z"],
        grid=np.linspace(-1, 4, 26),
        exogenous=["x"],
        fixed_effects=["id", "time"],
        covariance="cluster",
        cluster="id",
    )
    assert any(abs(value - 1.8) <= 0.11 for value in result.accepted_values)
    assert result.lower_bound <= 1.8 <= result.upper_bound
    assert len(result.grid_results) == 26
    assert list(result.plot_data().columns) == [
        "parameter_value",
        "statistic",
        "p_value",
        "accepted",
    ]


@pytest.mark.skipif(
    not check_r_environment(("fixest", "jsonlite")).available,
    reason="optional R fixest backend is unavailable",
)
def test_r_panel_iv_matches_python_within_coefficients() -> None:
    data = make_panel_iv()
    python_result = fit_panel_iv_2sls(
        data,
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z"],
        entity="id",
        time="time",
        covariance="cluster",
        absorption="within",
    )
    r_result = fit_panel_iv_2sls_r(
        data,
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z"],
        fixed_effects=["id", "time"],
        covariance="cluster",
        cluster="id",
    )
    r_coefficients = r_result.tidy().set_index("term")["estimate"].rename(
        index={"fit_endogenous": "endogenous"}
    )
    np.testing.assert_allclose(
        python_result.iv_result.coefficients.loc[["x", "endogenous"]],
        r_coefficients.loc[["x", "endogenous"]],
        rtol=0,
        atol=1e-10,
    )
    assert {"ivf", "ivwald"} <= {
        metric.split(".", 1)[0] for metric in r_result.first_stage_diagnostics["metric"]
    }


def test_panel_iv_and_ar_validation() -> None:
    data = make_panel_iv()
    with pytest.raises(ValueError, match="unique"):
        fit_panel_iv_2sls(
            pd.concat([data, data.iloc[[0]]]),
            "y",
            exogenous=["x"],
            endogenous=["endogenous"],
            instruments=["z"],
            entity="id",
            time="time",
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        anderson_rubin_confidence_set(data, "y", "endogenous", ["z"], grid=[0, 2, 1])
