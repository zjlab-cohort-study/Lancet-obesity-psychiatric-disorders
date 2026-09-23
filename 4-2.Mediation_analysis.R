########################################################################
# Fully runnable: strict (full) bootstrap mediation analysis script
# - Exposure: categorical (binary)
# - Mediators: list of continuous vars (will do log(x+1) + z)
# - Outcome: survival (time + event)
# - Bootstrap: fully re-fit a and b paths in each bootstrap sample
# - Multiple testing: both FDR (BH) and Bonferroni
#
########################################################################

library(dplyr)
library(survival)
setwd(dir = "your_working_directory")

# -------------------------------
# 0. User configuration
# -------------------------------
data_path <- "your_input_file.csv"
exposure  <- "obesity"
time_var  <- "follow_up_time"
event_var <- "event_label"

# Mediator column names (original names):
mediators <- c(
  "mediators"
)

# Covariates (numeric or factor both work)
continuous_vars <- c("age", "TDI", "sleep_duration")

categorical_vars <- c(
  "sex", "race", "college", "employment", "family_income",
  "alcohol", "smoking", "physical_activity", "assessment_center"
)

# Number of bootstrap replicates
B <- 2000

# Random seed (for reproducibility)
set.seed(12345)

# -------------------------------
# 1. Read data & basic checks
# -------------------------------
dat <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)

# Check that all required columns exist
covars <- c(continuous_vars, categorical_vars)
need_cols <- c(exposure, time_var, event_var, mediators, covars)
miss <- setdiff(need_cols, names(dat))
if (length(miss) > 0) stop("Missing required columns: ", paste(miss, collapse = ", "))

dat[categorical_vars] <- lapply(dat[categorical_vars], as.factor)
dat[continuous_vars] <- lapply(dat[continuous_vars], as.numeric)

event_values <- sort(unique(na.omit(dat[[event_var]])))
if (!all(event_values %in% c(0, 1))) {
  stop("event_var must be coded 0/1; current values: ", paste(event_values, collapse = ", "))
}

exposure_values <- sort(unique(na.omit(dat[[exposure]])))
if (length(exposure_values) < 2) {
  stop("exposure needs at least two levels; current values of ", exposure, ": ",
       paste(exposure_values, collapse = ", "))
}
dat[[event_var]] <- as.numeric(as.character(dat[[event_var]]))
dat[[exposure]] <- factor(dat[[exposure]], levels = exposure_values)

extract_named_coef <- function(model, target_name) {
  coef_vec <- coef(model)
  coef_names <- names(coef_vec)

  if (is.null(coef_names)) {
    return(NA_real_)
  }

  clean_names <- gsub("`", "", coef_names)
  match_idx <- match(target_name, clean_names)

  if (is.na(match_idx)) {
    return(NA_real_)
  }

  as.numeric(coef_vec[[match_idx]])
}

# -------------------------------
# 2. Standardize mediator columns
# -------------------------------
check_normality <- function(x, sample_n = 5000, alpha = 0.05) {
  x <- x[!is.na(x)]

  out <- list(
    normal = FALSE,
    p_value = NA_real_,
    n_non_missing = length(x),
    n_tested = NA_integer_
  )

  if (length(x) < 3) return(out)
  if (length(unique(x)) < 3) return(out)

  x_test <- x
  if (length(x_test) > sample_n) {
    x_test <- sample(x_test, sample_n)
  }

  p_val <- tryCatch(
    shapiro.test(x_test)$p.value,
    error = function(e) NA_real_
  )

  out$p_value <- p_val
  out$n_tested <- length(x_test)
  out$normal <- !is.na(p_val) && p_val >= alpha
  out
}

mediator_distribution_summary <- data.frame(
  mediator = mediators,
  n_non_missing = NA_integer_,
  n_tested = NA_integer_,
  shapiro_p = NA_real_,
  normal_before = NA,
  transform = NA_character_,
  stringsAsFactors = FALSE
)

for (i in seq_along(mediators)) {
  m <- mediators[i]

  if (is.factor(dat[[m]])) {
    dat[[m]] <- as.numeric(as.character(dat[[m]]))
  } else if (is.character(dat[[m]])) {
    dat[[m]] <- as.numeric(dat[[m]])
  }

  x <- as.numeric(dat[[m]])
  normality_res <- check_normality(x)
  normal_flag <- normality_res$normal
  x_trans <- x
  method <- "zscore"

  if (!normal_flag) {
    method <- "log_then_zscore"
    min_x <- suppressWarnings(min(x, na.rm = TRUE))

    if (is.finite(min_x) && min_x <= 0) {
      # Shift to positive values before taking the log, to avoid log(<= 0)
      x_trans <- log(x - min_x + 1)
    } else {
      x_trans <- log(x)
    }
  }

  m_mean <- mean(x_trans, na.rm = TRUE)
  m_sd <- sd(x_trans, na.rm = TRUE)

  if (is.na(m_sd) || m_sd == 0) {
    dat[[m]] <- ifelse(is.na(x_trans), NA_real_, 0)
  } else {
    dat[[m]] <- (x_trans - m_mean) / m_sd
  }

  mediator_distribution_summary$n_non_missing[i] <- normality_res$n_non_missing
  mediator_distribution_summary$n_tested[i] <- normality_res$n_tested
  mediator_distribution_summary$shapiro_p[i] <- normality_res$p_value
  mediator_distribution_summary$normal_before[i] <- normal_flag
  mediator_distribution_summary$transform[i] <- method
}

write.csv(mediator_distribution_summary,
          paste0("Mediator_distribution_summary_", event_var, ".csv"),
          row.names = FALSE)
cat("Mediator distribution summary saved.\n")
cat("Normal -> zscore:", sum(mediator_distribution_summary$transform == "zscore"), "\n")
cat("Non-normal -> log_then_zscore:", sum(mediator_distribution_summary$transform == "log_then_zscore"), "\n")

# -------------------------------
# 3. Full (non-simplified) mediation analysis with bootstrap, per mediator
#    Returns: a, b, c_total(c), c_prime, ACME (obs), ACME CI (boot), ACME p (boot)
# -------------------------------
results <- list()

cat("Starting full bootstrap mediation analysis per mediator: B =", B, "\n\n")

for (i in seq_along(mediators)) {
  med <- mediators[i]
  cat("===== Processing mediator:", med, "=====\n")

  # Drop rows with NA in required variables (estimation on the full sample)
  df_full <- dat %>%
    dplyr::select(all_of(c(exposure, time_var, event_var, covars, med))) %>%
    na.omit()

  # Skip if sample too small
  if (nrow(df_full) < 50) {
    warning("Sample size < 50, skipping ", med)
    next
  }

  # -------------------
  # a: lm(M ~ X + covars)
  # -------------------
  f_a <- as.formula(paste0("`", med, "` ~ ", exposure, " + ", paste(covars, collapse = " + ")))
  fit_a <- tryCatch(lm(f_a, data = df_full), error = function(e) NULL)
  if (is.null(fit_a)) {
    warning("a-path lm failed, skipping ", med); next
  }
  exposure_terms <- grep(paste0("^", exposure), names(coef(fit_a)), value = TRUE)
  if (length(exposure_terms) == 0) {
    warning("a-path produced no exposure coefficient, skipping ", med); next
  }

  # -------------------
  # c: total effect Cox (Y ~ X + covars)
  # -------------------
  f_c <- as.formula(paste0("Surv(", time_var, ",", event_var, ") ~ ", exposure, " + ", paste(covars, collapse = " + ")))
  fit_c <- tryCatch(coxph(f_c, data = df_full), error = function(e) NULL)
  if (is.null(fit_c)) {
    warning("c (total effect) Cox failed, skipping ", med); next
  }
  coef_c_all <- coef(fit_c)

  # -------------------
  # b & c': Cox (Y ~ X + M + covars)
  # -------------------
  f_b <- as.formula(paste0("Surv(", time_var, ",", event_var, ") ~ ", exposure, " + `", med, "` + ", paste(covars, collapse = " + ")))
  fit_b <- tryCatch(coxph(f_b, data = df_full), error = function(e) NULL)
  if (is.null(fit_b)) {
    warning("b/c' Cox failed, skipping ", med); next
  }
  b_obs <- extract_named_coef(fit_b, med)
  cprime_all <- coef(fit_b)
  if (is.na(b_obs)) {
    warning("b-path mediator coefficient not found, skipping ", med); next
  }

  # -------------------
  # Bootstrap: resample all rows (with replacement) and refit a and b
  # in each bootstrap sample
  # -------------------
  cat("  Running bootstrap ...\n")
  # Pre-allocate for speed
  ACME_boot_list <- setNames(vector("list", length(exposure_terms)), exposure_terms)
  for (term_name in exposure_terms) {
    ACME_boot_list[[term_name]] <- rep(NA_real_, B)
  }

  for (bi in 1:B) {
    # Sample row indices
    sidx <- sample.int(nrow(df_full), size = nrow(df_full), replace = TRUE)
    df_b <- df_full[sidx, , drop = FALSE]

    # Fit a (lm)
    fa_b <- tryCatch(lm(f_a, data = df_b), error = function(e) NULL)
    if (is.null(fa_b)) { next }
    coef_a_b <- coef(fa_b)

    # Fit b (cox)
    fb_b <- tryCatch(coxph(f_b, data = df_b), error = function(e) NULL)
    if (is.null(fb_b)) { next }

    # Skip if b coefficient missing
    b_b <- extract_named_coef(fb_b, med)
    if (is.na(b_b)) { next }

    for (term_name in exposure_terms) {
      a_b <- coef_a_b[term_name]
      if (is.na(a_b)) next
      ACME_boot_list[[term_name]][bi] <- as.numeric(a_b * b_b)
    }
  } # end bootstrap loop

  for (term_name in exposure_terms) {
    a_obs <- coef(fit_a)[term_name]
    c_total_obs <- coef_c_all[term_name]
    cprime_obs <- cprime_all[term_name]
    if (is.na(a_obs) || is.na(c_total_obs) || is.na(cprime_obs) || is.na(b_obs)) next

    ACME_obs <- as.numeric(a_obs * b_obs)
    ACME_boot_valid <- ACME_boot_list[[term_name]][!is.na(ACME_boot_list[[term_name]])]
    n_valid <- length(ACME_boot_valid)

    if (n_valid < max(50, floor(B * 0.5))) {
      warning(sprintf("Too few valid bootstrap replicates (%d/%d) for %s - %s, results may be unstable",
                      n_valid, B, med, term_name))
    }

    if (n_valid > 0) {
      CI_low  <- as.numeric(quantile(ACME_boot_valid, probs = 0.025, na.rm = TRUE))
      CI_high <- as.numeric(quantile(ACME_boot_valid, probs = 0.975, na.rm = TRUE))
      p_boot <- 2 * min(mean(ACME_boot_valid <= 0), mean(ACME_boot_valid >= 0))
      p_boot <- min(p_boot, 1)
    } else {
      CI_low <- NA_real_
      CI_high <- NA_real_
      p_boot <- NA_real_
    }

    Prop_med <- ifelse(is.na(c_total_obs) | c_total_obs == 0, NA, ACME_obs / c_total_obs)

    results[[paste(med, term_name, sep = "__")]] <- list(
      mediator = med,
      contrast = term_name,
      a = as.numeric(a_obs),
      b = as.numeric(b_obs),
      c_total = as.numeric(c_total_obs),
      c_prime = as.numeric(cprime_obs),
      ACME = as.numeric(ACME_obs),
      ACME_CI_low = CI_low,
      ACME_CI_high = CI_high,
      ACME_p = p_boot,
      ACME_boot_valid = n_valid,
      Prop_mediated = as.numeric(Prop_med)
    )

    cat(sprintf("  Done %s [%s]: a=%.4g, b=%.4g, ACME=%.4g, CI=[%.4g, %.4g], p=%.4g (valid %d/%d)\n",
                med, term_name, a_obs, b_obs, ACME_obs, CI_low, CI_high, p_boot, n_valid, B))
  }

} # end mediator loop

# -------------------------------
# 4. Aggregate results into a data.frame and apply
#    multiple-testing correction (FDR & Bonferroni)
# -------------------------------
if (length(results) == 0) {
  stop("No mediator completed fitting; check exposure coding, missing values, and model formulas.")
}

res_df <- do.call(rbind, lapply(results, function(x) {
  data.frame(
    mediator = x$mediator,
    contrast = x$contrast,
    a = x$a, b = x$b,
    c_total = x$c_total,
    c_prime = x$c_prime,
    ACME = x$ACME,
    ACME_CI_low = x$ACME_CI_low,
    ACME_CI_high = x$ACME_CI_high,
    ACME_p = x$ACME_p,
    ACME_boot_valid = x$ACME_boot_valid,
    Prop_mediated = x$Prop_mediated,
    stringsAsFactors = FALSE
  )
}))

# Adjusted p-values
res_df$ACME_p_FDR <- p.adjust(res_df$ACME_p, method = "BH")
res_df$ACME_p_Bonferroni <- p.adjust(res_df$ACME_p, method = "bonferroni")

# Save CSV
outname <- paste0("Mediation_bootstrap_results_B", B, "_obesity_blood_anxiety.csv")
write.csv(res_df, outname, row.names = FALSE)
cat("\nResults saved to:", outname, "\n")

# -------------------------------
# 5. Brief summary - print to console
# -------------------------------
print(res_df)
