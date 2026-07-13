"""Run with: uv run python examples/data_validation_example.py"""

import numpy as np
import pandas as pd

from empirical_standards.data import diagnose_panel, merge_validated


def main() -> None:
    panel = pd.DataFrame(
        {
            "city": np.repeat(["A", "B", "C"], 3),
            "year": np.tile([2020, 2021, 2022], 3),
            "outcome": [1.0, 1.4, 1.8, 2.0, 2.5, 2.7, 1.2, 1.3, 1.7],
            "coastal": np.repeat([1.0, 0.0, 1.0], 3),
        }
    )
    attributes = pd.DataFrame(
        {"city": ["A", "B", "C", "D"], "region": ["east", "west", "east", "north"]}
    )
    merged = merge_validated(panel, attributes, on="city", relationship="many_to_one")
    diagnostics = diagnose_panel(
        merged.data, entity="city", time="year", variables=["outcome", "coastal"]
    )
    print("Merge report:\n", merged.report)
    print("\nPanel summary:\n", diagnostics.summary().to_string())
    print("\nVariable variation:\n", diagnostics.variable_variation.to_string(index=False))


if __name__ == "__main__":
    main()
