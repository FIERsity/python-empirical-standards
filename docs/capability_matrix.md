# V1.0 capability matrix

This matrix separates stable estimators, design-matched diagnostics, experimental references,
and verification infrastructure. Passing tests establish reproducible numerical behavior, not a
valid research design.

## Stable estimation APIs

| Area | Public implementation | Main boundary |
|---|---|---|
| Data audit | Schema checks, validated merges, panel diagnostics | No data-version registry |
| OLS | Classical, HC1, one-way cluster covariance | No formula interface or multi-way cluster |
| Fixed effects | Entity, time, and two-way panel FE | General multi-dimensional HDFE uses R |
| Classic DID | Treated-by-post TWFE with explicit controls | Requires a defensible common-timing design |
| TWFE event study | Dynamic coefficients, joint pretrend test, support and plot-ready data | Not the primary estimator under heterogeneous staggered adoption |
| Staggered DID | R `did::att_gt` and `aggte`; support, weights, bootstrap bands | Requires the declared R environment |
| Cohort-interacted event study | R `fixest::sunab`; support and aggregation output | Requires an explicit reference cohort and period |
| IV/2SLS | Explicit roles, covariance choices, first-stage and specification tests | Identification remains substantive; no LIML/Fuller/JIVE |
| HDFE panel IV | R `fixest::feols` with explicit roles and fixed effects | R small-sample conventions are recorded, not silently changed |
| Anderson-Rubin | One endogenous coefficient, robust/cluster test and grid inversion | No multi-endogenous confidence region |

## Stable supporting tools

- Formal categorical FE interactions and joint equality tests.
- Covariance sensitivity, declared placebo timing, and leave-one-cluster-out diagnostics.
- R `fwildclusterboot` null-imposed wild-cluster inference when its optional package is installed.
- Bonferroni, Holm, and Benjamini-Hochberg adjustment for a declared hypothesis family.
- First-stage partial and Shea R-squared, excluded-instrument F/Wald tests, conditional relevance
  checks, and sample-rank validation. Rank summaries are input diagnostics, not weak-ID tests.
- Standard `tidy`, `glance`, specification, sample, provenance, and long-form `plot_data` outputs.
  The project prepares plotting data and does not draw figures.

## Experimental compatibility namespace

Import these only from `empirical_standards.experimental`:

- balanced-panel Python cohort-time DID reference;
- Python cohort-interaction event-study reference;
- subgroup-by-subgroup FE exploration;
- unrestricted-assignment DID permutation helper;
- restricted Python wild-cluster bootstrap reference.

They remain available for teaching and reproducibility but are not V1.0 production estimators.
The public workflow uses R `did`, `fixest`, and `fwildclusterboot` instead.

## Internal verification backends

The Python panel-IV indicator and `pyhdfe` within implementations remain available as transparent
small-sample and cross-language numerical oracles. Their homoskedastic absorbed-degree correction
is tested, but they are frozen rather than expanded into a general HDFE inference framework.

Deterministic Python-R benchmarks are verification infrastructure, not separate estimators.

## Deferred

- General Python multi-way HDFE and custom KP implementations.
- LIML, Fuller, JIVE, and multi-parameter weak-identification-robust regions.
- Spatial econometrics, machine-learning validation, and data-version management.
- Figure rendering and a complete publication-table system.

## Maturity rule

Use stable APIs for reusable workflows. Use experimental references only when their narrower
assumptions are the object of study. Every result must preserve the sample, estimand, treatment or
instrument roles, fixed effects, comparison group, covariance convention, and backend provenance.

## 中文

V1.0 将功能分为稳定接口、辅助诊断、实验参考和验证基础设施：

- 稳定估计包括 OLS、面板固定效应、经典 DID、TWFE 事件研究、R 交错 DID、R
  Sun–Abraham、基础 IV/2SLS、R 高维固定效应面板 IV 和单内生变量 AR 检验。
- 稳定辅助工具包括正式交互异质性、协方差敏感性、安慰剂、LOCO、R wild cluster
  bootstrap、多重检验校正及明确命名的第一阶段诊断。
- Python 交错 DID、Python cohort-interaction、分组回归、通用置换和受限 Python wild
  bootstrap 移至 `empirical_standards.experimental`，仅用于教学和兼容。
- Python 指示变量与 `pyhdfe` 面板 IV 后端保留作数值核验，不再扩展为通用 HDFE 框架。
- 项目输出标准化系数、支持度、检验和 `plot_data` 长表，不负责实际绘图。
- 空间计量、机器学习、数据版本管理和完整论文制表暂缓。
