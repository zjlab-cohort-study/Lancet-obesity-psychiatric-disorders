# ============================================
#   Dose-Response Analysis — illness_score x psychiatric risk
#   Restricted to excess_body_fat == 1
# ============================================

library(survival)
library(dplyr)
library(readr)
library(broom)
library(ggplot2)

# -----------------------------
# Paths
# -----------------------------
input_file  <- "your_input_file.csv"
output_dir  <- "output_dir"

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# -----------------------------
# Variable definitions (same covariate set as Cox model-3)
# -----------------------------
exposure <- "illness_score_cat"
event    <- "psychiatric_label"
time     <- "follow_up_time"

continuous_vars <- c("age", "TDI", "sleep_duration")
categorical_vars <- c("sex", "race", "college", "employment",
                      "family_income", "alcohol", "smoking", "physical_activity")
all_covariates <- c(continuous_vars, categorical_vars)

# ============================================================
#  1. Read data & restrict to excess_body_fat == 1
# ============================================================
data <- read_csv(input_file, show_col_types = FALSE)

required_vars <- unique(c("excess_body_fat", "illness_score", event, time, all_covariates))
missing_vars <- setdiff(required_vars, names(data))
if (length(missing_vars) > 0) {
  stop("Missing required columns: ", paste(missing_vars, collapse = ", "))
}

data <- data %>% filter(excess_body_fat == 1)

cat("Sample size with excess_body_fat == 1:", nrow(data), "\n")

# -----------------------------
# Build the grouping variable illness_score_cat
#   0 = score 0, 1 = score 1, 2+ = score >= 2
# -----------------------------
data <- data %>%
  mutate(
    illness_score_cat = case_when(
      illness_score == 0 ~ "0",
      illness_score == 1 ~ "1",
      illness_score >= 2 ~ "2+"
    ),
    illness_score_cat = factor(illness_score_cat, levels = c("0", "1", "2+"))
  )

cat("illness_score_cat distribution:\n")
print(table(data$illness_score_cat))

# Convert categorical covariates to factors
data[categorical_vars] <- lapply(data[categorical_vars], factor)

# ============================================================
#  2. Cox dose-response analysis (illness_score_cat)
# ============================================================
formula_str <- paste0(
  "Surv(", time, ", ", event, ") ~ ",
  exposure, " + ",
  paste(all_covariates, collapse = " + ")
)
cox_formula <- as.formula(formula_str)

cox_model <- coxph(cox_formula, data = data)

# Extract results
results <- tidy(cox_model, exponentiate = TRUE, conf.int = TRUE) %>%
  select(term, estimate, conf.low, conf.high, p.value) %>%
  rename(HR = estimate, CI_lower = conf.low, CI_upper = conf.high, p = p.value)

write_csv(results, file.path(output_dir, "Cox_dose_response_illness_score.csv"))
cat("Cox dose-response results saved\n")

# -----------------------------
#  P for trend (treat illness_score_cat as a continuous 0-2 score)
# -----------------------------
data$illness_score_num <- as.numeric(data$illness_score_cat) - 1  # 0, 1, 2

trend_formula <- as.formula(paste0(
  "Surv(", time, ", ", event, ") ~ illness_score_num + ",
  paste(all_covariates, collapse = " + ")
))
cox_trend <- coxph(trend_formula, data = data)
trend_res <- tidy(cox_trend, exponentiate = TRUE, conf.int = TRUE) %>%
  filter(term == "illness_score_num")

cat("\n--- P for trend ---\n")
cat("HR per level:", round(trend_res$estimate, 4),
    "95%CI:", round(trend_res$conf.low, 4), "-", round(trend_res$conf.high, 4),
    "p =", signif(trend_res$p.value, 4), "\n")

# ============================================================
#  3. Forest plot
# ============================================================
reference_level <- "0"

plot_data <- results %>%
  filter(grepl(paste0("^", exposure), term)) %>%
  mutate(group = sub(paste0("^", exposure), "", term)) %>%
  select(group, HR, CI_lower, CI_upper, p)

# Add the reference group
plot_data <- bind_rows(
  tibble(group = reference_level, HR = 1, CI_lower = 1, CI_upper = 1, p = NA),
  plot_data
) %>%
  mutate(group = factor(group, levels = c("0", "1", "2+"))) %>%
  arrange(group)

# Add label columns
plot_data <- plot_data %>%
  mutate(
    label = sprintf("%.2f (%.2f-%.2f)",
                    HR, CI_lower, CI_upper),
    p_label = ifelse(is.na(p), "Ref",
                     ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
  )

forest_plot <- ggplot(plot_data, aes(x = group, y = HR)) +
  geom_point(size = 4, color = "#4C78A8") +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                width = 0.15, linewidth = 0.8, color = "#4C78A8") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "#666666") +
  geom_text(aes(label = label), vjust = -1.2, size = 3.2) +
  geom_text(aes(label = p_label), vjust = -2.5, size = 3, color = "#666666",
            fontface = "italic") +
  scale_y_continuous(limits = c(0.5, max(plot_data$CI_upper, na.rm = TRUE) * 1.35),
                     breaks = seq(0.5, 5, by = 0.5)) +
  labs(
    title = "Dose-Response: Illness Score & Psychiatric Risk\n(excess body fat = 1)",
    x = "Illness Score Category",
    y = "HR (95% CI)"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold"),
    axis.title  = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(output_dir, "dose_response_forest_plot.pdf"),
       forest_plot, width = 7, height = 5)
ggsave(file.path(output_dir, "dose_response_forest_plot.png"),
       forest_plot, width = 7, height = 5, dpi = 300)
cat("Forest plot saved\n")

# ============================================================
#  4. Nonlinearity test (LRT: linear vs categorical)
# ============================================================
# illness_score_num has only 3 discrete values (0, 1, 2) with a highly
# skewed distribution, so automatic RCS knot placement fails. A likelihood
# ratio test (LRT) is used instead to assess nonlinearity:
#   H0: linear trend model vs H1: categorical (unrestricted) model
# i.e. testing departure from linearity of the exposure-outcome relation.

cov_str <- paste(all_covariates, collapse = " + ")

# Linear model (illness_score_num continuous)
lin_formula <- as.formula(paste0(
  "Surv(", time, ", ", event, ") ~ illness_score_num + ", cov_str
))
fit_linear <- coxph(lin_formula, data = data)

# Categorical model (illness_score_cat factor)
cat_formula <- as.formula(paste0(
  "Surv(", time, ", ", event, ") ~ illness_score_cat + ", cov_str
))
fit_categorical <- coxph(cat_formula, data = data)

# LRT: categorical vs linear
lrt <- anova(fit_linear, fit_categorical)
cat("\n--- Nonlinearity test (LRT: categorical vs linear) ---\n")
print(lrt)

# Extract the p-value
nonlinear_p <- lrt[2, "Pr(>|Chi|)"]
cat("P_nonlinearity =", signif(nonlinear_p, 4), "\n")

# Overall p: joint Wald test of the illness_score_cat coefficients
# (2 coefficients in cox_model from step 2)
beta_idx <- which(grepl("^illness_score_cat", names(coef(cox_model))))
if (length(beta_idx) > 0) {
  beta_vec <- coef(cox_model)[beta_idx]
  var_mat  <- vcov(cox_model)[beta_idx, beta_idx]
  wald_chi <- as.numeric(t(beta_vec) %*% solve(var_mat) %*% beta_vec)
  overall_p <- pchisq(wald_chi, df = length(beta_idx), lower.tail = FALSE)
} else {
  overall_p <- NA_real_
}
cat("P_overall (Wald) =", signif(overall_p, 4), "\n")

# -----------------------------
#  Line plot (categorical HR + 95% CI connected)
# -----------------------------
# Reuses plot_data from step 3 (already includes reference group 0)
rcs_plot_data <- plot_data %>%
  mutate(x_num = as.numeric(group) - 1)  # 0, 1, 2

# p-value text
p_overall_txt <- ifelse(overall_p < 0.001, "P_overall < 0.001",
                        paste0("P_overall = ", signif(overall_p, 3)))
p_nonlin_txt  <- ifelse(nonlinear_p < 0.001, "P_nonlinear < 0.001",
                        paste0("P_nonlinear = ", signif(nonlinear_p, 3)))
annotate_txt  <- paste0(p_overall_txt, "\n", p_nonlin_txt)

rcs_plot <- ggplot(rcs_plot_data, aes(x = x_num, y = HR)) +
  geom_line(linewidth = 1.2, color = "#E45756") +
  geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper),
              alpha = 0.2, fill = "#E45756") +
  geom_point(size = 3, color = "#E45756") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "#666666") +
  annotate("text", x = max(rcs_plot_data$x_num),
           y = max(rcs_plot_data$CI_upper) * 1.08,
           label = annotate_txt, hjust = 1.05, vjust = -0.3,
           size = 3.5, fontface = "italic") +
  scale_x_continuous(breaks = 0:2, labels = c("0", "1", ">=2")) +
  labs(
    title = "Dose-Response: Illness Score & Psychiatric Risk\n(excess body fat = 1)",
    x = "Illness Score Category",
    y = "HR (95% CI)"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold"),
    axis.title  = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(output_dir, "RCS_illness_score_plot.pdf"),
       rcs_plot, width = 7, height = 5)
ggsave(file.path(output_dir, "RCS_illness_score_plot.png"),
       rcs_plot, width = 7, height = 5, dpi = 300)
cat("Line plot saved\n")

# ============================================================
#  5. Summary output
# ============================================================
summary_file <- file.path(output_dir, "dose_response_summary.txt")
sink(summary_file)

cat("Dose-Response Analysis: illness_score and psychiatric risk\n")
cat("Restricted to excess_body_fat == 1; N =", nrow(data), "\n\n")

cat("--- Cox dose-response results ---\n")
print(results)

cat("\n--- P for trend ---\n")
cat("HR per level:", round(trend_res$estimate, 4),
    "95%CI:", round(trend_res$conf.low, 4), "-", round(trend_res$conf.high, 4),
    "p =", signif(trend_res$p.value, 4), "\n")

cat("\n--- Nonlinearity test (LRT: categorical vs linear) ---\n")
print(lrt)
cat("P_nonlinearity =", signif(nonlinear_p, 4), "\n")

cat("\nP_overall (joint Wald test) =", signif(overall_p, 4), "\n")

sink()
cat("Summary saved:", summary_file, "\n")
