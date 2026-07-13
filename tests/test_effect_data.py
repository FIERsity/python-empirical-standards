from __future__ import annotations

import pandas as pd
import pytest

from empirical_standards.results import effect_data


def test_effect_data_normalizes_att_and_preserves_support() -> None:
    table = pd.DataFrame(
        {
            "event_time": [-1, 0],
            "att": [0.0, 1.2],
            "std_error": [0.1, 0.2],
            "observations": [20, 18],
        }
    )
    result = effect_data(
        table,
        estimand="event_time_att",
        x="event_time",
        support_columns=("observations",),
    )
    assert list(result["estimate"]) == [0.0, 1.2]
    assert list(result["term"]) == ["event_time_att[-1]", "event_time_att[0]"]
    assert "observations" in result


def test_effect_data_rejects_nonfinite_estimates() -> None:
    with pytest.raises(ValueError, match="finite"):
        effect_data(pd.DataFrame({"estimate": [float("inf")]}), estimand="bad")
