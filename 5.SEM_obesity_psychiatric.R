# ==============================================================================
# Structural Equation Model (SEM) analysis script
# Supports: latent variable construction (CFA), data preprocessing (log/z-score),
# categorical outcome variables (WLSMV)
# ==============================================================================

# 0. Setup (make sure lavaan and lavaanPlot are installed)
# install.packages(c("lavaan", "lavaanPlot", "tidyverse"))
library(lavaan)
library(lavaanPlot)
library(dplyr)

make_indicator_aliases <- function(var_names, prefix) {
  stats::setNames(paste0(prefix, seq_along(var_names)), var_names)
}

make_dummy_covariates <- function(data, vars, outcome_var = NULL, min_level_n = 100, min_outcome_cell_n = 10) {
  dummy_names <- character(0)

  for (var_name in vars) {
    factor_values <- as.factor(data[[var_name]])
    factor_levels <- levels(factor_values)

    if (length(factor_levels) <= 1) {
      message("Skipping categorical covariate (<= 1 valid level): ", var_name)
      next
    }

    reference_level <- factor_levels[[1]]
    new_dummy_names <- character(0)

    for (level_name in factor_levels[-1]) {
      level_index <- !is.na(factor_values) & factor_values == level_name
      level_n <- sum(level_index)

      if (level_n < min_level_n) {
        message("Skipping sparse level: ", var_name, " = ", level_name, " (n = ", level_n, ")")
        next
      }

      if (!is.null(outcome_var)) {
        outcome_counts <- table(data[[outcome_var]][level_index], useNA = "no")
        if (length(outcome_counts) < 2 || any(outcome_counts < min_outcome_cell_n)) {
          message(
            "Skipping level with sparse outcome distribution: ",
            var_name,
            " = ",
            level_name
          )
          next
        }
      }

      dummy_name <- make.names(paste(var_name, level_name, sep = "__"))
      data[[dummy_name]] <- ifelse(is.na(factor_values), NA_real_, as.numeric(factor_values == level_name))
      dummy_names <- c(dummy_names, dummy_name)
      new_dummy_names <- c(new_dummy_names, dummy_name)
    }

    message(
      "Categorical covariate ",
      var_name,
      " converted to ",
      length(new_dummy_names),
      " dummy variables; reference level = ",
      reference_level
    )
  }

  list(data = data, dummy_names = dummy_names)
}

build_sem_model <- function(mri_names, blood_names, exposure_names, outcome_name, covariate_names, extra_residual_lines = character(0)) {
  mri_rhs <- c(
    paste0("a_pre*", exposure_names[[1]]),
    paste0("a_clin*", exposure_names[[2]]),
    covariate_names
  )
  blood_rhs <- c(
    paste0("b_pre*", exposure_names[[1]]),
    paste0("b_clin*", exposure_names[[2]]),
    covariate_names
  )
  outcome_rhs <- c(
    "d_mri*MRI_latent",
    "d_blood*Blood_latent",
    paste0("c_pre*", exposure_names[[1]]),
    paste0("c_clin*", exposure_names[[2]]),
    covariate_names
  )

  paste0(
    "\n  # a. Measurement model (CFA defining the latent variables)\n",
    "  MRI_latent   =~ ", paste(mri_names, collapse = " + "), "\n",
    "  Blood_latent =~ ", paste(blood_names, collapse = " + "), "\n\n",
    "  # b. Structural model (path analysis)\n",
    "  MRI_latent   ~ ", paste(mri_rhs, collapse = " + "), "\n",
    "  Blood_latent ~ ", paste(blood_rhs, collapse = " + "), "\n\n",
    "  ", outcome_name, " ~ ", paste(outcome_rhs, collapse = " + "), "\n\n",
    "  # c. Residual covariance (allows the two mediator latents to covary; optional, theory-driven)\n",
    "  MRI_latent ~~ Blood_latent\n",
    if (length(extra_residual_lines) > 0) paste0("  ", paste(extra_residual_lines, collapse = "\n  "), "\n\n") else "\n",
    "  # d. Indirect and total effects\n",
    "  indirect_pre_mri := a_pre * d_mri\n",
    "  indirect_pre_blood := b_pre * d_blood\n",
    "  indirect_pre_total := indirect_pre_mri + indirect_pre_blood\n",
    "  total_pre := c_pre + indirect_pre_total\n",
    "  indirect_clin_mri := a_clin * d_mri\n",
    "  indirect_clin_blood := b_clin * d_blood\n",
    "  indirect_clin_total := indirect_clin_mri + indirect_clin_blood\n",
    "  total_clin := c_clin + indirect_clin_total\n"
  )
}

extract_fit_measures <- function(fit) {
  measure_names <- c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr")
  measure_values <- tryCatch(
    fitMeasures(fit, measure_names),
    error = function(e) setNames(rep(NA_real_, length(measure_names)), measure_names)
  )
  as.data.frame(as.list(measure_values))
}

format_pvalue <- function(pvalue) {
  if (is.na(pvalue)) {
    return("NA")
  }
  if (pvalue < 0.001) {
    return("<0.001")
  }
  sprintf("%.3f", pvalue)
}

build_exposure_label <- function(pe, lhs_name) {
  pre_row <- pe[pe$lhs == lhs_name & pe$op == "~" & pe$rhs == "preclinical", , drop = FALSE]
  clin_row <- pe[pe$lhs == lhs_name & pe$op == "~" & pe$rhs == "clinical", , drop = FALSE]

  paste0(
    "Pre: ", sprintf("%.3f", pre_row$std.all[[1]]), " (p=", format_pvalue(pre_row$pvalue[[1]]), ")",
    "\nClin: ", sprintf("%.3f", clin_row$std.all[[1]]), " (p=", format_pvalue(clin_row$pvalue[[1]]), ")"
  )
}

build_single_path_label <- function(pe, lhs_name, rhs_name) {
  row <- pe[pe$lhs == lhs_name & pe$op == "~" & pe$rhs == rhs_name, , drop = FALSE]
  paste0(sprintf("%.3f", row$std.all[[1]]), " (p=", format_pvalue(row$pvalue[[1]]), ")")
}

save_widget_outputs <- function(widget, base_path) {
  htmlwidgets::saveWidget(
    widget = widget,
    file = paste0(base_path, ".html"),
    selfcontained = TRUE
  )

  if (requireNamespace("DiagrammeRsvg", quietly = TRUE) && requireNamespace("rsvg", quietly = TRUE)) {
    widget_svg <- DiagrammeRsvg::export_svg(widget)
    rsvg::rsvg_pdf(charToRaw(widget_svg), file = paste0(base_path, ".pdf"))
  }
}

build_simplified_structure_graph <- function(parameter_estimates_df, outcome_name) {
  sem_pe <- parameter_estimates_df
  exposure_to_mri <- build_exposure_label(sem_pe, "MRI_latent")
  exposure_to_blood <- build_exposure_label(sem_pe, "Blood_latent")
  exposure_to_outcome <- build_exposure_label(sem_pe, outcome_name)
  mri_to_outcome <- build_single_path_label(sem_pe, outcome_name, "MRI_latent")
  blood_to_outcome <- build_single_path_label(sem_pe, outcome_name, "Blood_latent")

  dot_code <- paste0(
    "digraph sem_main {\n",
    "graph [layout = dot, rankdir = LR, fontsize = 20, labelloc = t];\n",
    "node [shape = box, style = filled, fontname = Helvetica, fontsize = 18, color = '#2F4858', fillcolor = '#EAF2F8'];\n",
    "edge [fontname = Helvetica, fontsize = 13, color = '#4F5D75'];\n",
    "exposure [label = 'Obesity status\\n(0=reference; preclinical/clinical)'];\n",
    "mri [label = 'MRI latent'];\n",
    "blood [label = 'Blood latent'];\n",
    "outcome [label = '", outcome_name, "'];\n",
    "exposure -> mri [label = '", exposure_to_mri, "'];\n",
    "exposure -> blood [label = '", exposure_to_blood, "'];\n",
    "mri -> outcome [label = '", mri_to_outcome, "'];\n",
    "blood -> outcome [label = '", blood_to_outcome, "'];\n",
    "exposure -> outcome [label = '", exposure_to_outcome, "'];\n",
    "mri -> blood [dir = both, arrowtail = none, arrowhead = none, style = dashed, label = 'latent covariance'];\n",
    "}\n"
  )

  DiagrammeR::grViz(dot_code)
}

## =================== 1. Configuration (modify to match your data) ===================

# 1.1 File paths
data_path <- "your_input_file.csv"
out_dir <- "output_dir"

# 1.2 Core variables
expo_var <- "obesity"
outcome_var <- "event_label"

# 1.3 Mediator indicator lists
mri_indicators <- c("mri_mediators")
blood_indicators <- c("blood_mediators")

mri_indicator_alias <- make_indicator_aliases(mri_indicators, "mri_")
blood_indicator_alias <- make_indicator_aliases(blood_indicators, "blood_")

# 1.4 Covariates
cov_continuous <- c("age", "TDI", "sleep_duration", "scale_factor")
cov_categorical <- c(
  "sex", "smoking", "alcohol", "physical_activity", "race",
  "family_income", "college", "employment", "assessment_center"
)

## =================== 2. Read and preprocess data ===================

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# 2.1 Read data
df_raw <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)

# 2.2 Dummy-code the exposure (0/1/2 -> two dummies, 0 as baseline)
# Assume 0 = normal, 1 = overweight, 2 = obese
df_processed <- df_raw %>%
  mutate(
    preclinical = ifelse(.data[[expo_var]] == 1, 1, 0),
    clinical = ifelse(.data[[expo_var]] == 2, 1, 0)
  )
expo_dummies <- c("preclinical", "clinical")

# 2.3 Transform mediator variables
# MRI: Z-score, with lavaan-safe variable names.
for (source_name in names(mri_indicator_alias)) {
  alias_name <- mri_indicator_alias[[source_name]]
  df_processed[[alias_name]] <- as.numeric(scale(df_processed[[source_name]]))
}

# Blood: log1p (guards against zeros) + Z-score, with lavaan-safe variable names.
for (source_name in names(blood_indicator_alias)) {
  alias_name <- blood_indicator_alias[[source_name]]
  df_processed[[alias_name]] <- as.numeric(scale(log1p(df_processed[[source_name]])))
}

# 2.4 Declare covariate types (ensure categorical vars are factors)
df_processed <- df_processed %>%
  mutate(across(all_of(cov_categorical), as.factor))

# 2.5 Declare the outcome as ordered categorical and dummy-code categorical covariates.
df_processed[[outcome_var]] <- ordered(df_processed[[outcome_var]])
dummy_result <- make_dummy_covariates(df_processed, cov_categorical, outcome_var = outcome_var)
df_processed <- dummy_result$data
cov_categorical_dummies <- dummy_result$dummy_names

# 2.6 Keep only analysis variables and drop missing values
# (SEM is sensitive to missing data; WLSMV does not support FIML)
all_needed_vars <- c(outcome_var, expo_dummies,
                     unname(mri_indicator_alias), unname(blood_indicator_alias),
                     cov_continuous, cov_categorical_dummies)

df_analytic <- df_processed %>%
  select(all_of(all_needed_vars)) %>%
  na.omit()

message("Preprocessing done. Raw N = ", nrow(df_raw), "; complete-case N = ", nrow(df_analytic))


## =================== 3. Build SEM syntax and fit candidate models ===================

assessment_center_dummies <- grep("^assessment_center__", cov_categorical_dummies, value = TRUE)

fit_candidates <- list(
  list(
    name = "full_covariates",
    covariates = c(cov_continuous, cov_categorical_dummies),
    extra_residual_lines = character(0)
  ),
  list(
    name = "full_covariates_correlated_residuals",
    covariates = c(cov_continuous, cov_categorical_dummies),
    extra_residual_lines = c(
      "mri_3 ~~ mri_5",
      "mri_1 ~~ mri_2",
      "blood_2 ~~ blood_4"
    )
  ),
  list(
    name = "without_assessment_center",
    covariates = c(cov_continuous, setdiff(cov_categorical_dummies, assessment_center_dummies)),
    extra_residual_lines = character(0)
  ),
  list(
    name = "continuous_only",
    covariates = cov_continuous,
    extra_residual_lines = character(0)
  )
)

candidate_summaries <- list()
candidate_fits <- list()
candidate_models <- list()

for (candidate in fit_candidates) {
  sem_model_candidate <- build_sem_model(
    mri_names = unname(mri_indicator_alias),
    blood_names = unname(blood_indicator_alias),
    exposure_names = expo_dummies,
    outcome_name = outcome_var,
    covariate_names = candidate$covariates,
    extra_residual_lines = candidate$extra_residual_lines
  )

  cat(
    "### Fitting model:",
    candidate$name,
    "###\n",
    sem_model_candidate,
    "\n-------------------------------\n"
  )

  # WLSMV estimator for the ordered (binary) outcome;
  # candidates fall back automatically to simpler covariate sets
  fit_attempt <- tryCatch(
    sem(
      model = sem_model_candidate,
      data = df_analytic,
      ordered = outcome_var,
      estimator = "WLSMV"
    ),
    error = function(e) e
  )

  if (inherits(fit_attempt, "error")) {
    candidate_summaries[[candidate$name]] <- data.frame(
      candidate = candidate$name,
      converged = FALSE,
      n_covariates = length(candidate$covariates),
      chisq = NA_real_,
      df = NA_real_,
      pvalue = NA_real_,
      cfi = NA_real_,
      tli = NA_real_,
      rmsea = NA_real_,
      srmr = NA_real_,
      error_message = conditionMessage(fit_attempt),
      stringsAsFactors = FALSE
    )
    message("Model fit failed (", candidate$name, "): ", conditionMessage(fit_attempt))
    next
  }

  candidate_measure_df <- extract_fit_measures(fit_attempt)
  candidate_summaries[[candidate$name]] <- data.frame(
    candidate = candidate$name,
    converged = lavInspect(fit_attempt, "converged"),
    n_covariates = length(candidate$covariates),
    candidate_measure_df,
    error_message = "",
    stringsAsFactors = FALSE
  )

  if (lavInspect(fit_attempt, "converged")) {
    candidate_fits[[candidate$name]] <- fit_attempt
    candidate_models[[candidate$name]] <- sem_model_candidate
    message("Model converged; recording candidate: ", candidate$name)
  } else {
    message("Model did not converge; recorded, continuing with simpler covariate sets: ", candidate$name)
  }
}

candidate_summary_df <- bind_rows(candidate_summaries)

if (length(candidate_fits) == 0) {
  if (nrow(candidate_summary_df) > 0) {
    write.csv(candidate_summary_df, file.path(out_dir, "candidate_model_comparison.csv"), row.names = FALSE)
  }
  stop("None of the preset models converged; simplify the covariates further or adjust the model structure.")
}

best_candidate <- candidate_summary_df %>%
  filter(converged) %>%
  arrange(rmsea, desc(cfi), srmr, n_covariates) %>%
  slice(1) %>%
  pull(candidate)

fit <- candidate_fits[[best_candidate]]
fit_strategy <- best_candidate
sem_model <- candidate_models[[best_candidate]]

message("Final model strategy: ", fit_strategy)
write.csv(candidate_summary_df, file.path(out_dir, "candidate_model_comparison.csv"), row.names = FALSE)
writeLines(sem_model, file.path(out_dir, "selected_model_syntax.txt"))


## =================== 4. Results and evaluation ===================

# 4.1 Overall fit indices and parameter estimates
# standardized = TRUE gives fully standardized coefficients (std.all)
fit_summary_text <- capture.output(summary(fit, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE))
cat(paste(fit_summary_text, collapse = "\n"), "\n")
writeLines(fit_summary_text, file.path(out_dir, "fit_summary.txt"))

fit_measure_df <- extract_fit_measures(fit)
fit_measure_df$fit_strategy <- fit_strategy
write.csv(fit_measure_df, file.path(out_dir, "fit_measures.csv"), row.names = FALSE)

# 4.2 Extract the key path coefficient table
path_coefficients <- parameterEstimates(fit, standardized = TRUE) %>%
  filter(op == "~" | op == "=~") %>%
  select(lhs, op, rhs, est, se, z, pvalue, std.all)

parameter_estimates_df <- parameterEstimates(fit, standardized = TRUE) %>%
  as.data.frame()

indirect_effects_df <- parameter_estimates_df %>%
  filter(op == ":=")

write.csv(path_coefficients, file.path(out_dir, "path_coefficients.csv"), row.names = FALSE)
write.csv(parameter_estimates_df, file.path(out_dir, "parameter_estimates.csv"), row.names = FALSE)
write.csv(indirect_effects_df, file.path(out_dir, "indirect_effects.csv"), row.names = FALSE)

print("### Key path coefficients (standardized) ###")
print(path_coefficients)


## =================== 5. Path diagram visualization (lavaanPlot) ===================

tryCatch({
  sem_graph <- lavaanPlot(
    model = fit,
    labels = c(MRI_latent = "MRI brain structure", Blood_latent = "Blood biomarkers"), # rename latents if desired
    coefs = TRUE,          # show path coefficients
    stand = TRUE,          # show standardized coefficients
    stars = c("regress"),  # significance stars on regression paths
    sig = 0.05,            # show only paths with p < 0.05 (optional)
    graph_options = list(rankdir = "LR") # left-to-right layout
  )

  save_widget_outputs(sem_graph, file.path(out_dir, "sem_full_diagram"))

  simplified_graph <- build_simplified_structure_graph(parameter_estimates_df, outcome_var)
  save_widget_outputs(simplified_graph, file.path(out_dir, "sem_main_structure"))
}, error = function(e) {
  warning("Plotting failed; check variable names or graphics device settings. Error: ", e)
})
