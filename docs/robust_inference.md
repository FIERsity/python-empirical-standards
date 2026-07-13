# Robust inference and multiple testing

## Production wild-cluster inference

`wild_cluster_bootstrap_fe_r` fits the declared fixed-effects model with R `fixest` and calls
`fwildclusterboot::boottest`. The caller must state the tested coefficient, fixed effects,
clustering variable, bootstrap weights, null value, replication count, confidence level, and
random seed. The result records all of them and returns the bootstrap p-value and inverted
confidence interval.

This backend is optional and fails explicitly when `fwildclusterboot` is unavailable. It never
falls back to a different bootstrap algorithm.

The former Python `wild_cluster_bootstrap_fe` is retained under
`empirical_standards.experimental`. It is limited to one FE coefficient and entity/time
clustering and should not be extended as the production implementation.

## Permutation inference

The generic `permutation_did` helper is experimental. Shuffling a time-invariant treatment across
entities is valid only when that operation represents the actual assignment mechanism. It is not
valid by default for geographic adoption, staggered cohorts, stratified assignment, or irreversible
policy timing. A future stable randomization API must require explicit permutation blocks or an
assignment generator.

## Influence and specification sensitivity

`leave_one_cluster_out_fe` reports how a declared coefficient changes after omitting every cluster.
It is a sensitivity diagnostic, not an alternative sampling distribution.

`covariance_sensitivity` holds the coefficient specification fixed while changing declared
covariance estimators. `placebo_did` evaluates pre-specified placebo dates. Neither validates the
identification assumptions on its own.

## Multiple testing

`adjust_pvalues` implements Bonferroni, Holm, and Benjamini-Hochberg adjustment. Define the
hypothesis family before looking at the results and report raw and adjusted p-values together.

## 中文

- 正式 wild cluster 推断使用 `wild_cluster_bootstrap_fe_r`，对接 R `fixest` 与
  `fwildclusterboot`，并记录系数、聚类、权重、原假设、重复次数和随机种子。
- 原 Python wild bootstrap 和通用 DID 置换移入 `experimental`；后者只有在置换方式等于
  真实处理分配机制时才有效。
- LOCO、协方差敏感性和安慰剂是稳健性诊断，不能单独证明识别成立。
- 多重检验支持 Bonferroni、Holm 和 BH；应预先声明检验族，同时报告原始与调整后 p 值。
