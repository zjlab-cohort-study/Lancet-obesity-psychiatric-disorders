# ==========================================
# UKB Blood Cell Markers x Obesity: Linear Regression
# Separate handling of continuous/categorical covariates
# + Benjamini-Hochberg multiple-testing correction
# ==========================================

library(data.table)
library(dplyr)
library(broom)
library(parallel)

# -------------------------------
# 1. Read data
# -------------------------------
df <- fread("your_input_file.csv")
output_file <- "output.csv"

# -------------------------------
# 2. Variable definitions
# -------------------------------

# Exposure (binary: 0 = control, 1 = case)
exposure_var <- "obesity"

# Continuous covariates
continuous_vars <- c("age", "TDI", "sleep_duration")

# Categorical covariates
categorical_vars <- c(
  "sex", "race", "college", "employment", "family_income",
  "alcohol", "smoking", "physical_activity", "assessment_center"
)

# Mediator variables (blood cell markers): specify column names manually
blood_vars <- c(
  "Platelet_count",
  "White_blood_cell_count",
  "Red_blood_cell_count"
  # ... add the remaining blood cell marker column names here
)

# Check that all required columns exist
required_vars <- unique(c(exposure_var, continuous_vars, categorical_vars, blood_vars))
missing_vars <- setdiff(required_vars, names(df))
if (length(missing_vars) > 0) {
  stop("Missing required columns: ", paste(missing_vars, collapse = ", "))
}

# -------------------------------
# 3. Type conversion
# -------------------------------

# Exposure as factor with "0" as the reference level
df[[exposure_var]] <- as.factor(df[[exposure_var]])
df[[exposure_var]] <- relevel(df[[exposure_var]], ref = "0")

# Categorical covariates -> factor
df[, (categorical_vars) := lapply(.SD, as.factor), .SDcols = categorical_vars]

# Continuous covariates -> numeric
df[, (continuous_vars) := lapply(.SD, as.numeric), .SDcols = continuous_vars]

# Blood cell markers -> numeric
df[, (blood_vars) := lapply(.SD, as.numeric), .SDcols = blood_vars]

# -------------------------------
# 4. Standardize blood cell markers
# -------------------------------

# Shapiro-Wilk test on a subsample (at most 5000 obs for large samples)
# to decide whether a marker is approximately normal
is_approximately_normal <- function(x, sample_n = 5000, alpha = 0.05) {
  x <- x[!is.na(x)]

  if (length(x) < 3) return(FALSE)
  if (length(unique(x)) < 3) return(FALSE)

  if (length(x) > sample_n) {
    x <- sample(x, sample_n)
  }

  p_val <- tryCatch(
    shapiro.test(x)$p.value,
    error = function(e) NA_real_
  )

  if (is.na(p_val)) return(FALSE)
  p_val >= alpha
}

transform_summary <- data.table(
  blood = blood_vars,
  normal_before = NA,
  transform = NA_character_
)

for (i in seq_along(blood_vars)) {
  v <- blood_vars[i]
  x <- as.numeric(df[[v]])

  normal_flag <- is_approximately_normal(x)
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

  m <- mean(x_trans, na.rm = TRUE)
  s <- sd(x_trans, na.rm = TRUE)

  if (is.na(s) || s == 0) {
    x_z <- ifelse(is.na(x_trans), NA_real_, 0)
  } else {
    x_z <- (x_trans - m) / s
  }

  df[[v]] <- x_z
  transform_summary$normal_before[i] <- normal_flag
  transform_summary$transform[i] <- method
}

# -------------------------------
# 5. Covariate formula
# -------------------------------
covariates_formula <- paste(c(continuous_vars, categorical_vars), collapse = " + ")

# -------------------------------
# 6. Fit one linear model per marker
# -------------------------------
run_lm <- function(outcome) {

  formula_str <- paste0(
    "`", outcome, "` ~ ", exposure_var, " + ", covariates_formula
  )

  sub_df <- df[, c(outcome, exposure_var, continuous_vars, categorical_vars), with = FALSE]
  sub_df <- na.omit(sub_df)

  if (nrow(sub_df) < 100) return(NULL)

  model <- tryCatch(
    lm(as.formula(formula_str), data = sub_df),
    error = function(e) return(NULL)
  )

  if (is.null(model)) return(NULL)

  res <- tryCatch(
    broom::tidy(model),
    error = function(e) return(NULL)
  )

  if (is.null(res)) return(NULL)

  # Keep only the exposure coefficient (binary exposure -> exactly one term)
  res <- res[grepl(paste0("^", exposure_var), res$term), , drop = FALSE]

  if (nrow(res) == 0) return(NULL)

  res$outcome <- outcome
  res$n <- nrow(sub_df)

  return(res)
}

# -------------------------------
# 7. Run for all markers (parallel optional)
# -------------------------------

# Single core
# results_list <- lapply(blood_vars, run_lm)

# Multicore (recommended)
results_list <- mclapply(blood_vars, run_lm, mc.cores = 4)

results_list <- results_list[!sapply(results_list, is.null)]

results <- bind_rows(results_list)

# Benjamini-Hochberg FDR and Bonferroni correction across markers
results$p_BH <- p.adjust(results$p.value, method = "BH")
results$p_bonferroni <- p.adjust(results$p.value, method = "bonferroni")

print(head(results))

fwrite(results, output_file)
