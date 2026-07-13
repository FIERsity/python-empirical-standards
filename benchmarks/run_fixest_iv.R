args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg))
directory <- dirname(script_path)
data <- read.csv(file.path(directory, "iv_fixture.csv"))
model <- fixest::feols(y ~ x | endogenous ~ z1 + z2, data = data, vcov = "iid")
table <- fixest::coeftable(model, stage = 2)
output <- data.frame(
  term = rownames(table),
  estimate = as.numeric(table[, "Estimate"]),
  std_error = as.numeric(table[, "Std. Error"]),
  statistic = as.numeric(table[, "t value"]),
  p_value = as.numeric(table[, "Pr(>|t|)"])
)
write.csv(output, file.path(directory, "fixest_iv_results.csv"), row.names = FALSE)
