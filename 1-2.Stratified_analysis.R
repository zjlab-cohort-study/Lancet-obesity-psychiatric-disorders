library(survival)
library(dplyr)
library(readr)
library(broom)
library(tibble)

# -----------------------------
# Input/output paths
# -----------------------------
input_file <- "your_input_file.csv"
output_dir <- "output"
stratified_output_file <- file.path(output_dir, "Cox_obesity_psychiatric_stratified_results.csv")
interaction_output_file <- file.path(output_dir, "Cox_obesity_psychiatric_interaction_pvalues.csv")
interaction_terms_output_file <- file.path(output_dir, "Cox_obesity_psychiatric_interaction_terms.csv")

# -----------------------------
# Variables
# -----------------------------
exposure <- "obesity"
event <- "psychiatric_label"
time <- "follow_up_time"

continuous_vars <- c("age", "sleep_duration")
categorical_vars <- c(
  "sex", "race", "college", "employment",
  "family_income", "alcohol", "smoking", "physical_activity"
)

strata_vars <- c("age", "sleep_duration", categorical_vars)
all_covariates <- c(continuous_vars, categorical_vars)
required_vars <- unique(c(exposure, event, time, all_covariates))

# -----------------------------
# Helper functions
# -----------------------------
check_required_vars <- function(data, vars) {
  missing_vars <- setdiff(vars, names(data))
  if (length(missing_vars) > 0) {
    stop("Missing required variables: ", paste(missing_vars, collapse = ", "))
  }
}

make_binary_strata <- function(data, var_name) {
  if (var_name == "age") {
    return(if_else(data[[var_name]] < 60, "<60", ">=60", missing = NA_character_))
  }

  if (var_name == "sleep_duration") {
    return(if_else(data[[var_name]] < 8, "<8", ">=8", missing = NA_character_))
  }

  as.character(data[[var_name]])
}

build_formula <- function(exposure_var, covariates, time_var, event_var, interaction_var = NULL) {
  rhs_terms <- c(exposure_var, covariates)

  if (!is.null(interaction_var)) {
    rhs_terms <- c(rhs_terms, interaction_var, paste0(exposure_var, "*", interaction_var))
  }

  rhs_terms <- rhs_terms[!duplicated(rhs_terms)]

  as.formula(
    paste0("Surv(", time_var, ", ", event_var, ") ~ ", paste(rhs_terms, collapse = " + "))
  )
}

extract_exposure_results <- function(model, exposure_var) {
  tidy(model, exponentiate = TRUE, conf.int = TRUE) %>%
    filter(grepl(paste0("^", exposure_var), term)) %>%
    transmute(
      term,
      HR = estimate,
      CI_lower = conf.low,
      CI_upper = conf.high,
      p = p.value
    )
}

extract_exposure_term_names <- function(model, exposure_var, interaction_var = ".interaction_group") {
  coef_names <- names(stats::coef(model))
  coef_names[
    startsWith(coef_names, exposure_var) &
      !grepl(interaction_var, coef_names, fixed = TRUE)
  ]
}

compute_joint_wald_p <- function(model, term_names) {
  if (length(term_names) == 0) {
    return(NA_real_)
  }

  coef_vec <- stats::coef(model)[term_names]
  vcov_mat <- stats::vcov(model)[term_names, term_names, drop = FALSE]

  if (any(is.na(coef_vec)) || any(is.na(vcov_mat))) {
    return(NA_real_)
  }

  wald_stat <- tryCatch(
    as.numeric(t(coef_vec) %*% solve(vcov_mat, coef_vec)),
    error = function(e) NA_real_
  )

  if (is.na(wald_stat)) {
    return(NA_real_)
  }

  stats::pchisq(wald_stat, df = length(term_names), lower.tail = FALSE)
}

build_interaction_summary_row <- function(strata_var, interaction_model, overall_p, n, events, note = NA_character_) {
  summary_row <- list(
    strata_var = strata_var,
    p_interaction = overall_p,
    n = n,
    events = events,
    note = note
  )

  if (!is.null(interaction_model)) {
    exposure_terms <- extract_exposure_term_names(interaction_model, exposure)
    coef_names <- names(stats::coef(interaction_model))

    for (exposure_term in exposure_terms) {
      interaction_terms <- coef_names[
        startsWith(coef_names, paste0(exposure_term, ":.interaction_group")) |
          startsWith(coef_names, paste0(".interaction_group:", exposure_term))
      ]

      summary_row[[paste0("p_interaction_", exposure_term)]] <- compute_joint_wald_p(
        interaction_model,
        interaction_terms
      )
    }
  }

  as_tibble(summary_row)
}

empty_result_row <- function(strata_var, strata_level, n, events, note) {
  tibble(
    strata_var = strata_var,
    strata_level = strata_level,
    n = n,
    events = events,
    term = NA_character_,
    HR = NA_real_,
    CI_lower = NA_real_,
    CI_upper = NA_real_,
    p = NA_real_,
    note = note
  )
}

# -----------------------------
# Read and preprocess data
# -----------------------------
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data <- read_csv(input_file, show_col_types = FALSE)
check_required_vars(data, required_vars)

data_filt <- data %>%
  filter(.data[[time]] >= 0) %>%
  select(all_of(required_vars)) %>%
  mutate(
    across(all_of(categorical_vars), as.factor),
    across(all_of(event), as.numeric)
  )

data_filt[[exposure]] <- factor(data_filt[[exposure]], levels = sort(unique(data_filt[[exposure]])))

# -----------------------------
# Stratified analysis
# -----------------------------
stratified_results <- list()
interaction_summary_results <- list()
interaction_term_results <- list()

for (strata_var in strata_vars) {
  cat("Processing strata variable:", strata_var, "\n")

  analysis_data <- data_filt
  analysis_data$.strata_group <- make_binary_strata(analysis_data, strata_var)
  analysis_data <- analysis_data %>%
    filter(!is.na(.data[[event]]), !is.na(.data[[time]]), !is.na(.data[[exposure]]), !is.na(.strata_group))

  adjust_covariates <- setdiff(all_covariates, strata_var)

  for (level_name in unique(analysis_data$.strata_group)) {
    subgroup_data <- analysis_data %>%
      filter(.strata_group == level_name) %>%
      filter(if_all(all_of(adjust_covariates), ~ !is.na(.x)))

    n_obs <- nrow(subgroup_data)
    n_events <- sum(subgroup_data[[event]] == 1, na.rm = TRUE)

    if (n_obs == 0 || n_events == 0 || dplyr::n_distinct(subgroup_data[[exposure]]) < 2) {
      stratified_results[[length(stratified_results) + 1]] <- empty_result_row(
        strata_var, level_name, n_obs, n_events,
        note = "Insufficient sample, no events, or no variation in exposure"
      )
      next
    }

    cox_model <- tryCatch(
      coxph(build_formula(exposure, adjust_covariates, time, event), data = subgroup_data),
      error = function(e) e
    )

    if (inherits(cox_model, "error")) {
      stratified_results[[length(stratified_results) + 1]] <- empty_result_row(
        strata_var, level_name, n_obs, n_events,
        note = cox_model$message
      )
      next
    }

    model_results <- extract_exposure_results(cox_model, exposure)

    if (nrow(model_results) == 0) {
      stratified_results[[length(stratified_results) + 1]] <- empty_result_row(
        strata_var, level_name, n_obs, n_events,
        note = "No exposure terms in model output"
      )
      next
    }

    stratified_results[[length(stratified_results) + 1]] <- model_results %>%
      mutate(
        strata_var = strata_var,
        strata_level = level_name,
        n = n_obs,
        events = n_events,
        note = NA_character_
      ) %>%
      select(strata_var, strata_level, n, events, term, HR, CI_lower, CI_upper, p, note)
  }

  interaction_data <- analysis_data %>%
    mutate(.interaction_group = factor(.strata_group)) %>%
    filter(if_all(all_of(adjust_covariates), ~ !is.na(.x)))

  if (nrow(interaction_data) == 0 ||
      dplyr::n_distinct(interaction_data[[exposure]]) < 2 ||
      dplyr::n_distinct(interaction_data$.interaction_group) < 2) {
    interaction_summary_results[[length(interaction_summary_results) + 1]] <- build_interaction_summary_row(
      strata_var = strata_var,
      interaction_model = NULL,
      overall_p = NA_real_,
      n = nrow(interaction_data),
      events = sum(interaction_data[[event]] == 1, na.rm = TRUE),
      note = "Insufficient sample or no variation for interaction model"
    )
    next
  }

  interaction_formula <- build_formula(exposure, adjust_covariates, time, event, interaction_var = ".interaction_group")
  base_formula <- build_formula(exposure, c(adjust_covariates, ".interaction_group"), time, event)

  base_model <- tryCatch(coxph(base_formula, data = interaction_data), error = function(e) e)
  interaction_model <- tryCatch(coxph(interaction_formula, data = interaction_data), error = function(e) e)

  if (inherits(base_model, "error") || inherits(interaction_model, "error")) {
    error_message <- if (inherits(base_model, "error")) base_model$message else interaction_model$message
    interaction_summary_results[[length(interaction_summary_results) + 1]] <- build_interaction_summary_row(
      strata_var = strata_var,
      interaction_model = NULL,
      overall_p = NA_real_,
      n = nrow(interaction_data),
      events = sum(interaction_data[[event]] == 1, na.rm = TRUE),
      note = error_message
    )
    next
  }

  lrt_table <- anova(base_model, interaction_model, test = "Chisq")
  interaction_p <- lrt_table[2, "Pr(>|Chi|)"]

  interaction_summary_results[[length(interaction_summary_results) + 1]] <- build_interaction_summary_row(
    strata_var = strata_var,
    interaction_model = interaction_model,
    overall_p = interaction_p,
    n = nrow(interaction_data),
    events = sum(interaction_data[[event]] == 1, na.rm = TRUE)
  )

  interaction_term_results[[length(interaction_term_results) + 1]] <- tidy(interaction_model, exponentiate = TRUE, conf.int = TRUE) %>%
    filter(grepl(":.interaction_group", term, fixed = TRUE) | grepl(".interaction_group:", term, fixed = TRUE)) %>%
    transmute(
      strata_var = strata_var,
      term,
      HR = estimate,
      CI_lower = conf.low,
      CI_upper = conf.high,
      p = p.value
    )
}

write_csv(bind_rows(stratified_results), stratified_output_file)
write_csv(bind_rows(interaction_summary_results), interaction_output_file)
write_csv(bind_rows(interaction_term_results), interaction_terms_output_file)
