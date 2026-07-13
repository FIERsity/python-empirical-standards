"""Generate Python 2SLS benchmark output."""

from pathlib import Path

import pandas as pd

from empirical_standards import fit_iv_2sls


def main() -> None:
    directory = Path(__file__).parent
    data = pd.read_csv(directory / "iv_fixture.csv")
    result = fit_iv_2sls(
        data,
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z1", "z2"],
        covariance="unadjusted",
    )
    result.tidy().to_csv(directory / "python_iv_results.csv", index=False, float_format="%.15g")


if __name__ == "__main__":
    main()
