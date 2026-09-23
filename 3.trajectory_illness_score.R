########################################################################
# Pseudo-trajectory analysis of illness_score before psychiatric diagnosis
#
# Three steps:
#   Step 1: All samples pooled -> Cases vs Controls
#   Step 2: EBF=0 subset      -> Cases vs Controls
#   Step 3: EBF=1 subset      -> Cases vs Controls
#
# Method:
#   - Residualize illness_score for covariates
#   - Use Controls as reference to compute Z-scores (age-sex stratified)
#   - Both Cases and Controls are Z-scored against the same reference
#   - LOESS for both groups, plotted together (no scatter points)
#   - Controls include ALL disease-free participants (no >=2yr filter)
#     to ensure time-axis alignment
#
# Heatmaps (pooled / EBF=0 / EBF=1), original style (ordered by Z near
# diagnosis). Unsupervised clustering of LOESS trend curves (k=2, Ward,
# z-normalized shapes) reported separately per group: dendrogram PDF,
# cluster assignment CSV, and per-cluster trend-curve panels (clustering
# is NOT shown in the heatmaps).
########################################################################

rm(list = ls())

library(dplyr)
library(data.table)
library(ggplot2)
library(grid)
library(gridExtra)

# ---- Configuration ----
data_path  <- "your_input_file.csv"
output_dir <- "output_dir"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

covars_str <- paste(c("age", "sex", "TDI", "sleep_duration", "race", "college", "employment",
                      "family_income", "alcohol", "smoking", "physical_activity"),
                    collapse = " + ")

disorders <- c("anxiety", "bipolar", "dissociative", "feeding", "depression",
               "SUD", "neurodevelopmental", "OCD", "personality", "schizophrenia",
               "sleepwake", "somaticSymp", "truama")

disorder_names <- c(
  anxiety            = "Anxiety Disorders",
  bipolar            = "Bipolar Disorder",
  dissociative       = "Dissociative Disorders",
  feeding            = "Feeding & Eating Disorders",
  depression         = "Depressive Disorders",
  SUD                = "Substance Use Disorders",
  neurodevelopmental = "Neurodevelopmental Disorders",
  OCD                = "OCD",
  personality        = "Personality Disorders",
  schizophrenia      = "Schizophrenia Spectrum",
  sleepwake          = "Sleep-Wake Disorders",
  somaticSymp        = "Somatic Symptom Disorders",
  truama             = "Trauma & Stressor-Related Disorders"
)

loess_span <- 0.75
min_cases  <- 30

# ---- Read data ----
cat("============================================\n")
cat("Reading data...\n")
cat("============================================\n")
dat0 <- fread(data_path)

cat_vars <- c("sex", "race", "college", "employment",
              "family_income", "alcohol", "smoking", "physical_activity")
for (v in cat_vars) dat0[[v]] <- as.factor(dat0[[v]])
cat(sprintf("Total N = %d\n\n", nrow(dat0)))

# Check that all required columns exist
required_cols <- unique(c(
  "illness_score", "excess_body_fat", "age", "TDI", "sleep_duration", cat_vars,
  unlist(lapply(disorders, function(d) c(paste0(d, "_label"), paste0(d, "_followup_time"))))
))
missing_cols <- setdiff(required_cols, names(dat0))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

# All subsequent outputs are written into output_dir
setwd(output_dir)

########################################################################
# Core function: Case vs Control trajectory for one disorder x one sample
#
# 1. Residualize illness_score for covariates
# 2. Use controls as reference: compute age-sex stratified mean/SD
# 3. Z-score BOTH cases and controls
# 4. Fit LOESS for both
########################################################################
fit_case_control <- function(dat, disorder, sample_label) {

  label_col <- paste0(disorder, "_label")
  time_col  <- paste0(disorder, "_followup_time")

  # Step 1: Residualize
  lm_form <- as.formula(paste("illness_score ~", covars_str))
  fit <- lm(lm_form, data = as.data.frame(dat))
  dat$illness_resid <- residuals(fit)

  # Split: cases & controls (ALL disease-free, no >=2yr filter)
  cases_dt    <- dat[get(label_col) == 1]
  controls_dt <- dat[get(label_col) == 0]

  n_cases <- nrow(cases_dt)
  n_ctrls <- nrow(controls_dt)

  if (n_cases < min_cases) {
    cat(sprintf("  %-22s %-20s | cases=%d (<%d) SKIP\n",
                disorder, paste0("[", sample_label, "]"), n_cases, min_cases))
    return(NULL)
  }

  cat(sprintf("  %-22s %-20s | cases=%d, controls=%d\n",
              disorder, paste0("[", sample_label, "]"), n_cases, n_ctrls))

  # Step 2: Reference stats from controls (age-sex stratified)
  ref_stats <- controls_dt[, .(
    ref_mean = mean(illness_resid, na.rm = TRUE),
    ref_sd   = sd(illness_resid, na.rm = TRUE),
    ref_n    = .N
  ), by = .(sex, age)]

  # Merge to both cases and controls
  cases_dt    <- merge(cases_dt,    ref_stats, by = c("sex", "age"), all.x = TRUE)
  controls_dt <- merge(controls_dt, ref_stats, by = c("sex", "age"), all.x = TRUE)

  # Handle sparse cells: 3-year window
  for (dt in list(cases_dt, controls_dt)) {
    missing_idx <- which(is.na(dt$ref_mean))
    if (length(missing_idx) > 0) {
      for (i in missing_idx) {
        refs_w <- controls_dt[sex == dt$sex[i] &
                                age >= (dt$age[i] - 1) &
                                age <= (dt$age[i] + 1)]
        if (nrow(refs_w) >= 5) {
          dt$ref_mean[i] <- mean(refs_w$illness_resid, na.rm = TRUE)
          dt$ref_sd[i]   <- sd(refs_w$illness_resid, na.rm = TRUE)
        }
      }
    }
    # Remaining: sex-wide
    still_missing <- which(is.na(dt$ref_mean))
    if (length(still_missing) > 0) {
      for (i in still_missing) {
        refs_sex <- controls_dt[sex == dt$sex[i]]
        dt$ref_mean[i] <- mean(refs_sex$illness_resid, na.rm = TRUE)
        dt$ref_sd[i]   <- sd(refs_sex$illness_resid, na.rm = TRUE)
      }
    }
  }

  # Step 3: Z-score both groups
  cases_dt[, z_score := (illness_resid - ref_mean) / pmax(ref_sd, 1e-8)]
  controls_dt[, z_score := (illness_resid - ref_mean) / pmax(ref_sd, 1e-8)]
  cases_dt[, years_before := get(time_col)]
  controls_dt[, years_before := get(time_col)]

  # Step 4a: Cases LOESS
  span_c <- ifelse(n_cases < 100, 0.9,
            ifelse(n_cases < 500, 0.85, loess_span))
  loess_c <- tryCatch({
    loess(z_score ~ years_before, data = as.data.frame(cases_dt), span = span_c)
  }, error = function(e) NULL)
  if (is.null(loess_c)) return(NULL)

  xr_c <- range(cases_dt$years_before, na.rm = TRUE)
  grid_c <- data.frame(years_before = seq(xr_c[1], xr_c[2], length.out = 300))
  pred_c <- predict(loess_c, newdata = grid_c, se = TRUE)
  grid_c$y_pred  <- pred_c$fit
  grid_c$y_lower <- pred_c$fit - 1.96 * pred_c$se.fit
  grid_c$y_upper <- pred_c$fit + 1.96 * pred_c$se.fit
  grid_c <- grid_c[!is.na(grid_c$y_pred), ]

  # Step 4b: Controls LOESS (subsample if large)
  if (n_ctrls > 10000) {
    set.seed(42)
    controls_sub <- controls_dt[sample.int(n_ctrls, 10000)]
  } else {
    controls_sub <- controls_dt
  }
  loess_ctrl <- tryCatch({
    loess(z_score ~ years_before, data = as.data.frame(controls_sub), span = loess_span)
  }, error = function(e) NULL)

  grid_ctrl <- NULL
  if (!is.null(loess_ctrl)) {
    xr_ctrl <- range(controls_sub$years_before, na.rm = TRUE)
    grid_ctrl <- data.frame(years_before = seq(xr_ctrl[1], xr_ctrl[2], length.out = 300))
    pred_ctrl <- predict(loess_ctrl, newdata = grid_ctrl, se = TRUE)
    grid_ctrl$y_pred  <- pred_ctrl$fit
    grid_ctrl$y_lower <- pred_ctrl$fit - 1.96 * pred_ctrl$se.fit
    grid_ctrl$y_upper <- pred_ctrl$fit + 1.96 * pred_ctrl$se.fit
    grid_ctrl <- grid_ctrl[!is.na(grid_ctrl$y_pred), ]
  }

  # Statistics
  trend_fit  <- lm(z_score ~ years_before, data = as.data.frame(cases_dt))
  trend_coef <- coef(summary(trend_fit))["years_before", ]

  cases_near <- cases_dt[years_before <= 2]
  mean_near  <- ifelse(nrow(cases_near) > 0,
                       mean(cases_near$z_score, na.rm = TRUE), NA)

  return(list(
    disorder       = disorder,
    sample_label   = sample_label,
    n_cases        = n_cases,
    n_controls     = n_ctrls,
    cases_loess    = grid_c,
    controls_loess = grid_ctrl,
    trend_slope    = unname(trend_coef["Estimate"]),
    trend_p        = unname(trend_coef["Pr(>|t|)"]),
    mean_z_near    = mean_near,
    mean_z_controls = mean(controls_dt$z_score, na.rm = TRUE)
  ))
}

########################################################################
# Run all analyses: 3 steps
########################################################################
cat("============================================\n")
cat("Running trajectory analyses...\n")
cat("============================================\n\n")

all_results  <- list()
summary_rows <- list()

sample_defs <- list(
  list(key = "pooled", label = "All",              dat = dat0),
  list(key = "EBF0",   label = "No Excess Body Fat", dat = dat0[excess_body_fat == 0]),
  list(key = "EBF1",   label = "Excess Body Fat",     dat = dat0[excess_body_fat == 1])
)

for (samp in sample_defs) {
  cat(sprintf("--- %s (N=%d) ---\n", samp$label, nrow(samp$dat)))
  for (dis in disorders) {
    res <- fit_case_control(samp$dat, dis, samp$label)
    if (is.null(res)) next
    rk <- paste0(dis, "_", samp$key)
    all_results[[rk]] <- res
    summary_rows[[rk]] <- data.frame(
      disorder        = dis,
      disorder_name   = disorder_names[dis],
      sample          = samp$label,
      n_cases         = res$n_cases,
      n_controls      = res$n_controls,
      trend_slope     = res$trend_slope,
      trend_p         = res$trend_p,
      mean_z_cases_near = res$mean_z_near,
      mean_z_controls   = res$mean_z_controls
    )
  }
  cat("\n")
}

summary_df <- bind_rows(summary_rows)
summary_df$trend_p_fdr <- p.adjust(summary_df$trend_p, method = "BH")
write.csv(summary_df, "trajectory_summary_statistics.csv", row.names = FALSE)
cat("Summary statistics saved.\n")

########################################################################
# Plotting: clean LOESS curves only, no scatter points
########################################################################
cat("\n============================================\n")
cat("Generating plots...\n")
cat("============================================\n\n")

make_plot <- function(res, case_color, title_extra = "") {
  p <- ggplot()

  # Controls: gray dashed + CI ribbon
  if (!is.null(res$controls_loess)) {
    p <- p +
      geom_ribbon(data = res$controls_loess,
                  aes(x = years_before, ymin = y_lower, ymax = y_upper),
                  alpha = 0.10, fill = "gray60") +
      geom_line(data = res$controls_loess,
                aes(x = years_before, y = y_pred),
                linewidth = 1.0, linetype = "dashed", color = "gray50")
  }

  # Cases: colored solid + CI ribbon
  p <- p +
    geom_ribbon(data = res$cases_loess,
                aes(x = years_before, ymin = y_lower, ymax = y_upper),
                alpha = 0.20, fill = case_color) +
    geom_line(data = res$cases_loess,
              aes(x = years_before, y = y_pred),
              linewidth = 1.4, color = case_color) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "gray70", linewidth = 0.3) +
    scale_x_reverse(name = "Years Before Diagnosis") +
    labs(
      title    = paste0(disorder_names[res$disorder], title_extra),
      subtitle = paste0("Cases: n=", res$n_cases,
                        " | Controls: n=", res$n_controls,
                        " | Cases near diag: Z=", round(res$mean_z_near, 3)),
      y = "Z-score of Illness Score (vs. age-sex matched controls)"
    ) +
    annotate("text", x = Inf, y = Inf, hjust = 1.1, vjust = 1.5,
             label = "Cases (solid)", size = 3, color = case_color) +
    annotate("text", x = Inf, y = Inf, hjust = 1.1, vjust = 2.8,
             label = "Controls (dashed)", size = 3, color = "gray50") +
    theme_bw(base_size = 11) +
    theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(color = "gray40", size = 9),
      panel.grid.minor = element_blank()
    )

  return(p)
}

# ---- Per-step plots (Step 1 pooled / Step 2 EBF=0 / Step 3 EBF=1) ----
plot_specs <- list(
  list(step = "Step 1: Pooled", key = "pooled", prefix = "Step1_pooled",
       color = "#D95F02", extra = ""),
  list(step = "Step 2: EBF=0", key = "EBF0", prefix = "Step2_EBF0",
       color = "#4A90D9", extra = " (No Excess Body Fat)"),
  list(step = "Step 3: EBF=1", key = "EBF1", prefix = "Step3_EBF1",
       color = "#D95F02", extra = " (Excess Body Fat)")
)

for (spec in plot_specs) {
  cat(spec$step, "plots...\n")
  for (dis in disorders) {
    res <- all_results[[paste0(dis, "_", spec$key)]]
    if (is.null(res)) next
    p <- make_plot(res, case_color = spec$color, title_extra = spec$extra)
    ggsave(sprintf("%s_%s.pdf", spec$prefix, dis), p, width = 8, height = 5.5)
  }
}

# ---- Summary panel (pooled) ----
cat("\nSummary panel...\n")
panel_list <- list()
for (dis in disorders) {
  res <- all_results[[paste0(dis, "_pooled")]]
  if (is.null(res)) next

  p <- ggplot()
  if (!is.null(res$controls_loess)) {
    p <- p +
      geom_line(data = res$controls_loess,
                aes(x = years_before, y = y_pred),
                linewidth = 0.5, linetype = "dashed", color = "gray60")
  }
  p <- p +
    geom_ribbon(data = res$cases_loess,
                aes(x = years_before, ymin = y_lower, ymax = y_upper),
                alpha = 0.15, fill = "#D95F02") +
    geom_line(data = res$cases_loess,
              aes(x = years_before, y = y_pred),
              linewidth = 0.8, color = "#D95F02") +
    geom_hline(yintercept = 0, linetype = "dotted", color = "gray70", linewidth = 0.2) +
    scale_x_reverse(name = "Years Before Diagnosis") +
    labs(title = disorder_names[dis], y = "Z-score") +
    theme_bw(base_size = 7) +
    theme(
      plot.title = element_text(face = "bold", size = 8),
      panel.grid.minor = element_blank(),
      legend.position = "none",
      axis.text = element_text(size = 6),
      axis.title = element_text(size = 7)
    )
  panel_list[[dis]] <- p
}

if (length(panel_list) > 0) {
  n_p <- length(panel_list); nc <- ceiling(sqrt(n_p)); nr <- ceiling(n_p / nc)
  fig <- arrangeGrob(
    grobs = panel_list, ncol = nc,
    top = textGrob("Cases (solid) vs Controls (dashed) - All Pooled",
                   gp = gpar(fontsize = 13, fontface = "bold")),
    bottom = textGrob("Orange = Cases | Gray dashed = Controls | Dotted = Z=0",
                      gp = gpar(fontsize = 10, col = "gray40")),
    padding = unit(1, "line")
  )
  ggsave("summary_panel_pooled.pdf", fig, width = nc * 3.5, height = nr * 2.8 + 2)
  cat("  Panel saved.\n")
}

# ======================================================================
# Heatmaps (pooled / EBF=0 / EBF=1), original style: ordered by Z near
# diagnosis, no facets / no extra labels.
#
# Unsupervised clustering of the LOESS trend curves (separate output,
# NOT shown in the heatmaps):
#   - curves interpolated onto a common time grid (rule = 2 clamping)
#   - mean-centered and unit-scaled per disorder -> distance reflects
#     trajectory SHAPE (direction of change), invariant to burden level
#     and amplitude
#   - hierarchical clustering (Ward linkage), cut at k = 2
#   - outputs per group: dendrogram PDF, cluster assignment CSV, and a
#     trend-curve figure (one panel per cluster); clusters described
#     post-hoc by mean dZ (near minus far Z)
# ======================================================================
cat("\nHeatmaps + unsupervised curve clustering (pooled / EBF=0 / EBF=1)...\n")

common_time <- seq(0, 17, length.out = 50)
near_win    <- c(0, 2)

heat_specs <- list(
  list(key = "pooled", label = "All samples (pooled)"),
  list(key = "EBF0",   label = "No Excess Body Fat"),
  list(key = "EBF1",   label = "Excess Body Fat")
)

build_heat_data <- function(sample_key) {
  heat_list <- list()
  stat_list <- list()
  for (dis in disorders) {
    res <- all_results[[paste0(dis, "_", sample_key)]]
    if (is.null(res)) next

    yb <- res$cases_loess$years_before
    zp <- res$cases_loess$y_pred

    # Interpolate onto common grid (rule = 2 clamps at curve ends)
    heat_list[[dis]] <- data.frame(disorder     = disorder_names[dis],
                                   years_before = common_time,
                                   z_score      = approx(x = yb, y = zp,
                                                         xout = common_time,
                                                         rule = 2)$y)

    near_idx <- yb <= near_win[2]
    if (!any(near_idx)) near_idx <- yb <= min(yb) + 2   # short-range fallback
    stat_list[[dis]] <- data.frame(
      disorder = disorder_names[dis],
      z_near   = mean(zp[near_idx]),
      z_far    = mean(zp[yb >= max(yb) - 3])            # own last 3 years
    )
  }
  if (length(heat_list) == 0) return(NULL)

  stats_df  <- bind_rows(stat_list) %>% mutate(delta_z = z_near - z_far)
  curve_mat <- t(vapply(heat_list, function(d) d$z_score,
                        numeric(length(common_time))))
  rownames(curve_mat) <- vapply(heat_list, function(d) d$disorder[1],
                                character(1))
  list(df = bind_rows(heat_list), curve_mat = curve_mat, stats = stats_df)
}

heat_data <- setNames(lapply(heat_specs, function(hs) build_heat_data(hs$key)),
                      vapply(heat_specs, function(hs) hs$key, character(1)))
has_data <- !vapply(heat_data, is.null, logical(1))

if (!any(has_data)) {
  cat("  No eligible disorders in any group; heatmaps skipped.\n")
} else {

  # Shared color scale so heatmaps are comparable across groups
  z_limits <- range(unlist(lapply(heat_data[has_data],
                                  function(h) h$df$z_score)),
                    na.rm = TRUE, finite = TRUE)

  for (hs in heat_specs[has_data]) {
    hd      <- heat_data[[hs$key]]
    heat_df <- hd$df
    stats   <- hd$stats

    cat(sprintf("  [%s]\n", hs$label))

    # ---- Unsupervised k = 2 clustering of the trend curves ----
    # Center AND unit-scale each curve so distance reflects trajectory
    # SHAPE (direction of change), invariant to burden level & amplitude
    curve_centered <- hd$curve_mat - rowMeans(hd$curve_mat)
    curve_scaled   <- curve_centered / pmax(apply(curve_centered, 1, sd), 1e-8)
    if (nrow(curve_scaled) >= 2) {
      hc <- hclust(dist(curve_scaled), method = "ward.D2")
      cl <- cutree(hc, k = 2)
    } else {
      hc <- NULL
      cl <- setNames(rep(1L, nrow(curve_scaled)), rownames(curve_scaled))
    }
    stats$cluster <- unname(cl[stats$disorder])

    # Post-hoc cluster description (descriptive only, not assignment)
    cl_desc <- stats %>%
      group_by(cluster) %>%
      summarise(n = n(), mean_dZ = mean(delta_z),
                mean_z_near = mean(z_near), .groups = "drop") %>%
      mutate(pattern = ifelse(mean_dZ >= 0, "rising toward diagnosis",
                              "declining toward diagnosis"))
    for (i in seq_len(nrow(cl_desc))) {
      members <- stats$disorder[stats$cluster == cl_desc$cluster[i]]
      cat(sprintf("    Cluster %d (n=%d, mean dZ=%+.2f, %s): %s\n",
                  cl_desc$cluster[i], cl_desc$n[i], cl_desc$mean_dZ[i],
                  cl_desc$pattern[i], paste(members, collapse = ", ")))
    }

    cl_out <- stats %>%
      left_join(cl_desc %>% select(cluster, pattern), by = "cluster") %>%
      transmute(sample = hs$label, disorder, cluster, pattern,
                delta_z, z_near, z_far)
    write.csv(cl_out, sprintf("trajectory_curve_clustering_%s.csv", hs$key),
              row.names = FALSE)
    cat(sprintf("    Saved: trajectory_curve_clustering_%s.csv\n", hs$key))

    if (!is.null(hc)) {
      pdf(sprintf("curve_clustering_dendrogram_%s.pdf", hs$key),
          width = 9, height = 5)
      par(mar = c(9, 4, 3, 1))
      plot(hc, labels = rownames(curve_centered), ann = FALSE, hang = -1)
      title(main = paste0("Unsupervised Clustering of Illness-Score ",
                          "Trajectories: ", hs$label),
            ylab = "Euclidean distance (z-normalized curves)")
      rect.hclust(hc, k = 2, border = c("#D95F02", "#4A90D9"))
      dev.off()
      cat(sprintf("    Saved: curve_clustering_dendrogram_%s.pdf\n", hs$key))
    }

    # ---- Trend curves by cluster: one panel per cluster ----
    curve_list <- list()
    for (dis in disorders) {
      res <- all_results[[paste0(dis, "_", hs$key)]]
      if (is.null(res)) next
      curve_list[[dis]] <- data.frame(
        disorder     = disorder_names[dis],
        years_before = res$cases_loess$years_before,
        y_pred       = res$cases_loess$y_pred
      )
    }
    curve_df <- bind_rows(curve_list)
    curve_df$cluster <- stats$cluster[match(curve_df$disorder, stats$disorder)]

    facet_keys <- setNames(
      sprintf("Cluster %d (n=%d, %s)", cl_desc$cluster, cl_desc$n,
              sub(" toward diagnosis", "", cl_desc$pattern)),
      cl_desc$cluster)
    curve_df$cluster_lab <- unname(facet_keys[as.character(curve_df$cluster)])
    curve_df$cluster_lab <- factor(curve_df$cluster_lab,
                                   levels = unname(facet_keys))

    p_clust <- ggplot(curve_df, aes(x = years_before, y = y_pred,
                                    color = disorder)) +
      geom_hline(yintercept = 0, linetype = "dotted",
                 color = "gray70", linewidth = 0.3) +
      geom_line(linewidth = 0.8) +
      scale_x_reverse(name = "Years Before Diagnosis") +
      facet_wrap(~ cluster_lab) +
      labs(
        title    = paste0("Illness-Score Trajectories by Cluster: ", hs$label),
        subtitle = paste0("Cases' LOESS Z-score curves, one panel per ",
                          "unsupervised cluster (k=2)"),
        y = "Z-score of Illness Score", color = NULL
      ) +
      theme_bw(base_size = 11) +
      theme(
        plot.title      = element_text(face = "bold", size = 13),
        plot.subtitle   = element_text(color = "gray40", size = 9),
        legend.position = "bottom",
        axis.text.x     = element_text(size = 9)
      ) +
      guides(color = guide_legend(nrow = 3, byrow = TRUE))

    ggsave(sprintf("trajectory_clusters_%s.pdf", hs$key), p_clust,
           width = 12, height = 7)
    cat(sprintf("    Saved: trajectory_clusters_%s.pdf\n", hs$key))

    # ---- Heatmap: original style, ordered by Z near diagnosis ----
    dis_order <- stats %>% arrange(desc(z_near)) %>% pull(disorder)
    heat_df$disorder <- factor(heat_df$disorder, levels = rev(dis_order))

    p_heat <- ggplot(heat_df, aes(x = years_before, y = disorder,
                                  fill = z_score)) +
      geom_tile(width = 0.35, height = 0.9) +
      scale_fill_gradient2(
        low = "#4A90D9", mid = "white", high = "#D95F02",
        midpoint = 0, limits = z_limits,
        name = "Z-score\n(Cases vs Controls)"
      ) +
      scale_x_reverse(name = "Years Before Diagnosis",
                      breaks = seq(0, 16, 2)) +
      labs(
        title    = paste0("Illness Score Z-score Heatmap: ", hs$label),
        subtitle = "Orange = higher illness score in cases | Blue = lower",
        y = ""
      ) +
      theme_minimal(base_size = 10) +
      theme(
        plot.title      = element_text(face = "bold", size = 13),
        plot.subtitle   = element_text(color = "gray40", size = 9),
        axis.text.y     = element_text(size = 9),
        axis.text.x     = element_text(size = 8),
        panel.grid      = element_blank(),
        legend.position = "right",
        legend.text     = element_text(size = 8),
        legend.title    = element_text(size = 9)
      )

    ggsave(sprintf("heatmap_%s.pdf", hs$key), p_heat, width = 10, height = 6.5)
    cat(sprintf("    Saved: heatmap_%s.pdf\n", hs$key))
  }
}

cat("\n============================================\n")
cat("All analyses complete!\n")
cat("Output directory:", output_dir, "\n")
cat("============================================\n")
