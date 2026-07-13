# Panel IV and Anderson-Rubin inference

`fit_panel_iv_2sls` adds entity and/or time fixed effects to the explicit 2SLS interface. The
current implementation creates reference-category indicators and includes them in both 2SLS
stages. This is algebraically exact and gives transparent finite-sample degrees of freedom,
but it is intentionally not advertised as a high-dimensional implementation. Very large
fixed-effect sets require a later absorbed-IV covariance implementation and an external
benchmark before becoming public API.

Panel keys must be complete and unique. Entity clustering is the default when clustered
covariance is selected without an explicit cluster column. Public coefficient tables hide the
internal indicator coefficients, while model metadata records the implementation and number
of absorbed indicators.

`anderson_rubin_test` tests one endogenous coefficient by subtracting its hypothesized value
from the outcome and jointly testing the excluded instruments in the resulting reduced-form
regression. It supports exogenous controls, categorical fixed effects, HC1 covariance, and
one-way clustering. Unlike the usual 2SLS t-test, its size does not rely on strong first-stage
identification under the standard AR assumptions.

`anderson_rubin_confidence_set` inverts the test over a user-supplied increasing grid. The full
accepted grid is retained because weak-identification-robust confidence sets can be
disconnected or unbounded. `lower_bound` and `upper_bound` are only the envelope of accepted
grid points; the `bounded_below` and `bounded_above` flags indicate whether the accepted set
reaches a grid edge. A wider grid and finer spacing are required before reporting endpoints.

The current AR implementation covers one endogenous regressor. Multi-endogenous-variable
weak-identification diagnostics and Kleibergen-Paap procedures remain future work.

