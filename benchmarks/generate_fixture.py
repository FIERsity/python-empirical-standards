"""Generate the deterministic cross-software panel fixture."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    entities, periods = 30, 8
    entity = np.repeat(np.arange(entities), periods)
    time = np.tile(np.arange(periods), entities)
    x = np.sin(entity * 0.7 + time * 0.4) + (entity % 4) * 0.1
    treated = (entity < 15).astype(int)
    post = (time >= 4).astype(int)
    noise = 0.08 * np.cos(entity * 1.3 + time * 0.9)
    y = (entity % 7) * 0.2 + time * 0.15 + 1.25 * x + 2.0 * treated * post + noise
    fixture = pd.DataFrame({"id": entity, "time": time, "x": x, "did": treated * post, "y": y})
    path = Path(__file__).with_name("panel_fixture.csv")
    fixture.to_csv(path, index=False, float_format="%.15g")

    observations = np.arange(600)
    z1 = np.sin(observations * 0.17)
    z2 = np.cos(observations * 0.11)
    control = np.sin(observations * 0.07 + 0.4)
    structural_error = np.cos(observations * 0.23) + 0.2 * np.sin(observations * 0.31)
    endogenous = 0.9 * z1 + 0.6 * z2 + 0.4 * control + 0.5 * structural_error
    iv_outcome = 1.0 + 0.8 * control + 2.0 * endogenous + structural_error
    iv_fixture = pd.DataFrame(
        {"y": iv_outcome, "x": control, "endogenous": endogenous, "z1": z1, "z2": z2}
    )
    iv_fixture.to_csv(Path(__file__).with_name("iv_fixture.csv"), index=False, float_format="%.15g")


if __name__ == "__main__":
    main()
