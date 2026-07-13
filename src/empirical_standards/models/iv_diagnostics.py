"""Interpret first-stage statistics without overstating weak-identification diagnostics."""

from __future__ import annotations

import pandas as pd

from empirical_standards.models.iv import IV2SLSResult


def summarize_first_stage(result: IV2SLSResult) -> pd.DataFrame:
    """Label conventional F and robust Wald first-stage statistics explicitly.

    The normalized robust Wald statistic is descriptive and is deliberately not labeled as a
    Kleibergen-Paap statistic.
    """
    table = result.first_stage.copy()
    instrument_count = len(result.instruments)
    distributions = table["excluded_instruments_distribution"].astype(str)
    conventional = distributions.str.startswith("F(")
    table["statistic_kind"] = conventional.map(
        {True: "conventional_partial_F", False: "robust_excluded_instrument_Wald"}
    )
    table["excluded_instrument_count"] = instrument_count
    table["wald_per_excluded_instrument"] = table["excluded_instruments_statistic"]
    robust_rows = ~conventional
    table.loc[robust_rows, "wald_per_excluded_instrument"] = (
        table.loc[robust_rows, "excluded_instruments_statistic"] / instrument_count
    )
    table["is_kleibergen_paap"] = False
    table["interpretation_note"] = (
        "Inspect the distribution. Robust Wald/q is not a Kleibergen-Paap statistic."
    )
    return table
