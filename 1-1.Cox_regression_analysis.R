# ============================================
#      Cox Regression Script (with Covariates)
# ============================================

library(survival)
library(dplyr)
library(readr)
library(broom)
library(ggplot2)

# -----------------------------
# Input/output paths
# -----------------------------
input_file <- "your_input_file.csv"
output_file <- "output.csv"
plot_file <- sub("\\.csv$", "_HR_barplot.pdf", output_file)

# -----------------------------
# Define variables
# -----------------------------
exposure <- "obesity"
event <- "psychiatric_label"
time <- "follow_up_time"

# continuous_vars <- c("age")
# categorical_vars <- c("sex", "race") #model-1

# continuous_vars <- c("age", "TDI")
# categorical_vars <- c("sex", "race", "college", "employment", "family_income") #model-2

continuous_vars <- c("age", "TDI", "sleep_duration")
categorical_vars <- c("sex", "race", "college", "employment", "family_income", "alcohol", "smoking", "physical_activity") #model-3


all_covariates <- c(continuous_vars, categorical_vars)

# -----------------------------
# Read data and filter on follow-up time
# -----------------------------
data <- read_csv(input_file)
data_filt <- data %>% filter(data[[time]] >= 0)
# -----------------------------
# Convert categorical variables to factors
# -----------------------------
data_filt[[exposure]] <- factor(data_filt[[exposure]], levels = sort(unique(data_filt[[exposure]])))
data_filt[categorical_vars] <- lapply(data_filt[categorical_vars], factor)

# -----------------------------
# Build model formula
# -----------------------------
formula_str <- paste0(
  "Surv(", time, ", ", event, ") ~ ",
  exposure, " + ",
  paste(all_covariates, collapse = " + ")
)
cox_formula <- as.formula(formula_str)

# -----------------------------
# Fit Cox regression
# -----------------------------
cox_model <- coxph(cox_formula, data = data_filt)

# -----------------------------
# Extract HR, 95% CI, and p-value
# -----------------------------
results <- tidy(cox_model, exponentiate = TRUE, conf.int = TRUE) %>%
  select(term, estimate, conf.low, conf.high, p.value) %>%
  rename(
    HR = estimate,
    CI_lower = conf.low,
    CI_upper = conf.high,
    p = p.value
  )

# -----------------------------
# Write results to CSV
# -----------------------------
print(results)
write_csv(results, output_file)

cat("Cox regression results saved to:", output_file, "\n")

# -----------------------------
# Plot HR bar chart for the exposure and save as PDF
# -----------------------------
reference_level <- levels(data_filt[[exposure]])[1]

plot_data <- results %>%
  filter(grepl(paste0("^", exposure), term)) %>%
  mutate(group = sub(paste0("^", exposure), "", term)) %>%
  select(group, HR, CI_lower, CI_upper)

plot_data <- bind_rows(
  tibble(
    group = reference_level,
    HR = 1,
    CI_lower = 1,
    CI_upper = 1
  ),
  plot_data
) %>%
  mutate(group = factor(group, levels = levels(data_filt[[exposure]]))) %>%
  arrange(group)

hr_plot <- ggplot(plot_data, aes(x = group, y = HR)) +
  geom_col(fill = "#4C78A8", width = 0.65) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.15) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "#666666") +
  scale_y_continuous(limits = c(0, 2.5), breaks = seq(0, 2.5, by = 0.5)) +
  labs(
    title = paste0("Hazard Ratios for ", exposure),
    x = exposure,
    y = "HR (95% CI)"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    panel.grid.minor = element_blank()
  )

ggsave(plot_file, hr_plot, width = 7, height = 5)

cat("HR bar chart saved to:", plot_file, "\n")
