args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
directory <- dirname(normalizePath(sub("^--file=", "", script_arg)))
data <- read.csv(file.path(directory, "iv_fixture.csv"))
data$id <- rep(0:59, each = 10)
data$time <- rep(0:9, 60)

fixed_effects <- model.matrix(~ factor(id) + factor(time), data = data)
residualize <- function(values) {
  values - fixed_effects %*% qr.coef(qr(fixed_effects), values)
}
y <- residualize(as.matrix(data$y))
x <- residualize(as.matrix(data[c("x", "endogenous")]))
z <- residualize(as.matrix(data[c("x", "z1", "z2")]))
projection_cross <- crossprod(x, z) %*% solve(crossprod(z)) %*% crossprod(z, x)
beta <- solve(projection_cross, crossprod(x, z) %*% solve(crossprod(z)) %*% crossprod(z, y))
residual <- y - x %*% beta
absorbed_degrees <- qr(fixed_effects)$rank
residual_df <- nrow(data) - ncol(x) - absorbed_degrees
sigma2 <- as.numeric(crossprod(residual) / residual_df)
standard_error <- sqrt(diag(sigma2 * solve(projection_cross)))
output <- data.frame(
  term = c("x", "endogenous"),
  estimate = as.numeric(beta),
  std_error = standard_error
)
write.csv(output, file.path(directory, "r_panel_iv_results.csv"), row.names = FALSE)
