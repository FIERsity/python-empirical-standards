# Panel IV and Anderson-Rubin inference

## Recommended panel-IV backend

Use `fit_panel_iv_2sls_r` for IV models with one or more high-dimensional fixed effects. It builds
a `fixest::feols` IV formula from explicit Python arguments; it does not accept arbitrary formula
strings. The result contains second-stage coefficients, available `fixest` first-stage and IV
diagnostics, the generated formula, covariance choice, clustering, fixed effects, package versions,
and the exact estimation-sample fingerprint.

The wrapper accepts `unadjusted`, heteroskedastic-robust, or one-way clustered covariance. It keeps
the documented `fixest` small-sample defaults and records them. Cross-language comparisons must
align these conventions before comparing standard errors.

## Python verification backends

`fit_panel_iv_2sls` remains available for transparent verification:

- `absorption="indicators"` creates exact fixed-effect indicators and is suitable for small data;
- `absorption="within"` residualizes with `pyhdfe` and defaults to asymptotic covariance;
- homoskedastic within estimation can opt into `within_covariance_correction="absorbed_df"`.

The indicator and within coefficients are benchmarked against each other and R. This path is
frozen: robust or clustered absorbed-degree corrections and general multi-way Python HDFE are not
V1.0 development targets.

## Anderson-Rubin inference

`anderson_rubin_test` tests one structural coefficient by regressing
`outcome - null_value * endogenous` on the controls, excluded instruments, and explicit fixed-effect
indicators, then jointly testing the instruments. It supports classical, HC1, or one-way clustered
covariance.

`anderson_rubin_confidence_set` inverts this test over an explicit increasing grid. The full
`parameter_value`, statistic, p-value, and acceptance table is available through `plot_data()`.
Do not assume the accepted set is connected. If accepted values touch a grid boundary, expand the
grid before interpreting the bounds.

The implementation covers one endogenous coefficient. Multiple-endogenous AR regions, LIML,
Fuller, and general weak-identification regions are deferred.

## Diagnostic naming

Python conditional first-stage Wald/F tests and matrix-rank summaries retain their literal names.
They are not relabeled as KP statistics. The R `fixest` backend may return a package-native `kpr`
diagnostic when it is defined for the fitted specification; the result records the backend and
metric name rather than claiming a universal custom KP implementation.

## 中文

- 复杂面板 IV 优先使用 `fit_panel_iv_2sls_r`，由明确的变量角色生成 R `fixest` 模型，
  返回二阶段系数、第一阶段诊断、协方差、固定效应、版本和样本指纹。
- Python 指示变量与 `pyhdfe` within 后端保留作透明核验；同方差模型仍可启用已验证的
  吸收自由度修正，但不再扩展稳健/聚类修正或通用 HDFE。
- AR 检验支持单内生变量、控制变量、显式固定效应及经典/HC1/单向聚类协方差；置信集合
  保留完整网格，并通过 `plot_data()` 提供绘图前数据。
- Python 条件第一阶段和秩摘要不称为 KP。R 后端只有在 `fixest` 对具体模型成功返回时才
  保留原生 `kpr` 指标。
