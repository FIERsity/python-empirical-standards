args <- commandArgs(trailingOnly = TRUE)
input_path <- args[[1]]
spec_path <- args[[2]]
output_dir <- args[[3]]

suppressPackageStartupMessages(library(fixest))
suppressPackageStartupMessages(library(jsonlite))

data <- read.csv(input_path, check.names = FALSE)
spec <- fromJSON(spec_path, simplifyVector = TRUE)
quote_name <- function(x) paste0("`", gsub("`", "``", x, fixed = TRUE), "`")
join_terms <- function(x) {
  if (length(x) == 0) "1" else paste(vapply(x, quote_name, character(1)), collapse = " + ")
}

formula_text <- paste0(
  quote_name(spec$outcome), " ~ ", join_terms(spec$exogenous), " | ",
  join_terms(spec$fixed_effects), " | ", join_terms(spec$endogenous), " ~ ",
  join_terms(spec$instruments)
)
model_formula <- as.formula(formula_text)
vcov_value <- switch(
  spec$covariance,
  unadjusted = "iid",
  robust = "hetero",
  cluster = as.formula(paste0("~", quote_name(spec$cluster)))
)
fit <- feols(model_formula, data = data, vcov = vcov_value)

coefs <- as.data.frame(coeftable(fit), check.names = FALSE)
names(coefs) <- c("estimate", "std_error", "statistic", "p_value")
coefs$term <- rownames(coefs)
rownames(coefs) <- NULL
ci <- as.data.frame(confint(fit, level = spec$confidence_level), check.names = FALSE)
names(ci) <- c("conf_low", "conf_high")
coefs$conf_low <- ci[[1]]
coefs$conf_high <- ci[[2]]
coefs <- coefs[, c("term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high")]
write.csv(coefs, file.path(output_dir, "coefficients.csv"), row.names = FALSE)

stat_types <- c("ivf", "ivwald", "cd", "kpr", "wh", "sargan")
flatten <- numeric(0)
for (stat_type in stat_types) {
  value <- tryCatch(
    fitstat(fit, stat_type, verbose = FALSE),
    error = function(error) NULL
  )
  if (!is.null(value)) {
    current <- unlist(value, recursive = TRUE, use.names = TRUE)
    current <- current[is.finite(suppressWarnings(as.numeric(current)))]
    if (length(current) > 0) {
      names(current) <- paste0(stat_type, ".", names(current))
      flatten <- c(flatten, current)
    }
  }
}
first_stage <- data.frame(metric = names(flatten), value = as.numeric(flatten))
write.csv(first_stage, file.path(output_dir, "first_stage_diagnostics.csv"), row.names = FALSE)

metadata <- list(
  estimator = "r_panel_iv_2sls",
  backend = "Rscript",
  package = "fixest",
  nobs = nobs(fit),
  formula = formula_text,
  covariance = spec$covariance,
  cluster = if (is.null(spec$cluster)) NA else spec$cluster,
  fixed_effects = spec$fixed_effects,
  small_sample_correction = list(
    K_adj = TRUE, K_fixef = "nonnested", G_adj = TRUE, G_df = "min", t_df = "min"
  )
)
write_json(metadata, file.path(output_dir, "result.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
