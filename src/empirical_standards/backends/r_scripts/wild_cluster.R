args <- commandArgs(trailingOnly = TRUE)
input_path <- args[[1]]
spec_path <- args[[2]]
output_dir <- args[[3]]

suppressPackageStartupMessages(library(fixest))
suppressPackageStartupMessages(library(fwildclusterboot))
suppressPackageStartupMessages(library(jsonlite))

data <- read.csv(input_path, check.names = FALSE)
spec <- fromJSON(spec_path, simplifyVector = TRUE)
quote_name <- function(x) paste0("`", gsub("`", "``", x, fixed = TRUE), "`")
join_terms <- function(x) paste(vapply(x, quote_name, character(1)), collapse = " + ")
formula_text <- paste0(
  quote_name(spec$outcome), " ~ ", join_terms(spec$predictors), " | ",
  join_terms(spec$fixed_effects)
)
cluster_formula <- as.formula(paste0("~", quote_name(spec$cluster)))
fit <- feols(as.formula(formula_text), data = data, vcov = cluster_formula)

set.seed(spec$random_state)
boot <- boottest(
  fit,
  param = spec$coefficient,
  B = spec$replications,
  clustid = spec$cluster,
  type = spec$weight_distribution,
  impose_null = spec$impose_null,
  r = spec$null_value,
  sign_level = 1 - spec$confidence_level,
  conf_int = TRUE,
  sampling = "standard"
)

metadata <- list(
  estimator = "r_wild_cluster_bootstrap_fe",
  backend = "Rscript",
  package = "fwildclusterboot",
  nobs = nobs(fit),
  formula = formula_text,
  coefficient = spec$coefficient,
  estimate = unname(coef(fit)[[spec$coefficient]]),
  bootstrap_p_value = unname(boot$p_val),
  conf_low = unname(boot$conf_int[[1]]),
  conf_high = unname(boot$conf_int[[2]]),
  replications = boot$boot_iter,
  cluster = spec$cluster,
  weight_distribution = spec$weight_distribution,
  impose_null = spec$impose_null,
  null_value = spec$null_value,
  random_state = spec$random_state
)
write_json(metadata, file.path(output_dir, "result.json"), auto_unbox = TRUE, pretty = TRUE)
