"""Generate absorbed-DF panel-IV benchmark output."""

from pathlib import Path

import numpy as np
import pandas as pd

from empirical_standards import fit_panel_iv_2sls


def main() -> None:
    directory = Path(__file__).parent
    data = pd.read_csv(directory / "iv_fixture.csv")
    data["id"] = np.repeat(np.arange(60), 10)
    data["time"] = np.tile(np.arange(10), 60)
    result = fit_panel_iv_2sls(
        data,
        "y",
        exogenous=["x"],
        endogenous=["endogenous"],
        instruments=["z1", "z2"],
        entity="id",
        time="time",
        covariance="unadjusted",
        absorption="within",
        within_covariance_correction="absorbed_df",
    )
    result.tidy().to_csv(
        directory / "python_panel_iv_results.csv", index=False, float_format="%.15g"
    )


if __name__ == "__main__":
    main()
