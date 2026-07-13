# Result and metadata contract

Every public estimator returns an object with five common views:

- `tidy()`: parameter-level or group-time estimates.
- `glance()`: one-dimensional model summary.
- `model_spec()`: estimator, outcome, predictors, and consequential settings.
- `sample_info()`: original/used rows, included columns, and a SHA-256 sample fingerprint.
- `provenance()`: Python, platform, and core dependency versions.

Dynamic estimators and grid-inversion results additionally expose `plot_data()`. This returns a
long-form numerical table containing the horizontal coordinate, estimate or test statistic,
intervals, and available support fields. It does not render a figure or choose labels, colors,
fonts, axes, or a plotting library.

The fingerprint covers the exact ordered estimation data, including its index, column names,
and dtypes. It detects accidental changes; it is not a privacy guarantee and should not be
used as a substitute for archiving source data. Platform metadata may differ across otherwise
numerically equivalent runs.

Model metadata and plot-ready tables are intended to travel with exported estimates. Reporting
code should consume this contract rather than inspect estimator-specific internals.
