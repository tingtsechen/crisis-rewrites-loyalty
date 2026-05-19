#!/usr/bin/env Rscript
# ===========================================================================
# GRF BLP Production Script — Main G03 Analysis
#
# Causal Forest with Best Linear Projection (BLP)
# Treatment: healthrisk_impact (COVID-19 period 2020 vs pre-pandemic 2019)
# Interaction of interest: republicanStd_x_brand
#
# 13 Controls:
#   Prescription: dspns_qty_mean
#   Healthcare:   pharmacy_density, insurance_coverage_mean,
#                 patient_coverage_mean, loc_id_nunique
#   Demographics: female_percentage, over_65_percentage
#   Education:    edu_bachelors
#   Income:       income_middle_percentage
#   Employment:   unemployment_rate
#   Race:         race_white
#   Culture:      evangelical_pct
#   Time:         week
#
# 7 Heterogeneity features:
#   republican_percentage_std, is_brand, republicanStd_x_brand,
#   drug_ANDROGENS_ANABOLIC_sum, drug_CONTRACEPTIVES_sum,
#   drug_ESTROGENS_sum, drug_PROGESTINS_sum
#
# BLP A-matrix: republican_percentage_std + republicanStd_x_brand
# ===========================================================================

library(grf)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
    BASE_DIR <- dirname(dirname(normalizePath(script_path)))
} else {
    BASE_DIR <- getwd()
}
DATA_DIR <- file.path(BASE_DIR, "data")
OUTPUT_BASE <- file.path(BASE_DIR, "data_output")

TIMESTAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
OUTPUT_DIR <- file.path(OUTPUT_BASE, paste0("grf_blp_main_G03_", TIMESTAMP))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("============================================================\n")
cat("GRF BLP PRODUCTION — G03 (Sex Hormones)\n")
cat(sprintf("  Output: %s\n", OUTPUT_DIR))
cat("============================================================\n\n")

# ---------------------------------------------------------------------------
# Load and merge data
# ---------------------------------------------------------------------------
main_file <- file.path(DATA_DIR, "final_data",
    "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv")
df <- read.csv(main_file)

# Healthcare indicators
hc_file <- file.path(DATA_DIR, "processed", "county_healthcare_indicators_2020.csv")
if (file.exists(hc_file)) {
    hc <- read.csv(hc_file)
    df <- merge(df, hc, by = "fips", all.x = TRUE)
} else {
    stop("Healthcare indicators file not found: ", hc_file)
}

# Religiosity
relig_file <- file.path(DATA_DIR, "external", "county_religiosity_2020.csv")
if (file.exists(relig_file)) {
    relig <- read.csv(relig_file)
    relig$fips <- as.integer(relig$fips)
    df <- merge(df, relig[, c("fips", "evangelical_pct")],
                by = "fips", all.x = TRUE)
} else {
    stop("Religiosity file not found: ", relig_file)
}

# ---------------------------------------------------------------------------
# Create derived variables
# ---------------------------------------------------------------------------
df$avg_price <- df$rx_prc_amt_sum / df$transaction_count
df$log_avg_price <- log1p(df$avg_price)
if (!"log_sales" %in% names(df)) df$log_sales <- log1p(df$rx_prc_amt_sum)
if (!"log_transactions" %in% names(df)) df$log_transactions <- log1p(df$transaction_count)

df$republican_percentage_std <- scale(df$republican_percentage)[, 1]
df$republicanStd_x_brand <- df$republican_percentage_std * df$is_brand

cat(sprintf("Data loaded: %d rows, %d counties\n\n",
            nrow(df), length(unique(df$fips))))

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
hetero_features <- c(
    "republican_percentage_std", "is_brand", "republicanStd_x_brand",
    "drug_ANDROGENS_ANABOLIC_sum", "drug_CONTRACEPTIVES_sum",
    "drug_ESTROGENS_sum", "drug_PROGESTINS_sum"
)

controls <- c(
    "dspns_qty_mean",              # Prescription volume baseline
    "pharmacy_density",            # Drug supply access
    "insurance_coverage_mean",     # Insurance coverage
    "patient_coverage_mean",       # Patient cost burden
    "loc_id_nunique",              # Provider diversity
    "female_percentage",           # G03 sex hormone user base
    "over_65_percentage",          # Age-related drug demand
    "edu_bachelors",               # SES proxy
    "income_middle_percentage",    # Price sensitivity proxy
    "unemployment_rate",           # Economic pressure
    "race_white",                  # Demographic composition
    "evangelical_pct",             # Religious conservatism
    "week"                         # Temporal trend
)

x_features <- c(hetero_features, controls)

blp_vars <- c("republican_percentage_std", "republicanStd_x_brand")

# Model parameters
NUM_TREES <- 3000
MIN_NODE  <- 15
SEED      <- 42

# ---------------------------------------------------------------------------
# DVs to analyze
# ---------------------------------------------------------------------------
dvs <- c("log_sales", "log_avg_price", "log_transactions")

# ---------------------------------------------------------------------------
# Run analysis for each DV
# ---------------------------------------------------------------------------
all_blp_results <- list()
all_ate_results <- list()

for (dv in dvs) {
    cat("============================================================\n")
    cat(sprintf("DV: %s\n", dv))
    cat("============================================================\n")

    # Clean data
    required <- c("healthrisk_impact", dv, x_features, "fips")
    df_clean <- df[complete.cases(df[, required]), ]
    cat(sprintf("  Observations: %d (dropped %d with missing values)\n",
                nrow(df_clean), nrow(df) - nrow(df_clean)))
    cat(sprintf("  Clusters: %d counties\n", length(unique(df_clean$fips))))

    X <- as.matrix(df_clean[, x_features])
    W <- df_clean$healthrisk_impact
    Y <- df_clean[[dv]]
    W_hat <- rep(mean(W), length(W))

    # --- Causal Forest ---
    cat(sprintf("  Fitting causal forest (trees=%d, min_node=%d) ... ",
                NUM_TREES, MIN_NODE))
    flush.console()
    t0 <- Sys.time()

    cf <- causal_forest(
        X = X, Y = Y, W = W, W.hat = W_hat,
        num.trees = NUM_TREES, min.node.size = MIN_NODE,
        seed = SEED, clusters = df_clean$fips
    )

    t1 <- Sys.time()
    cat(sprintf("done (%.1f sec)\n", as.numeric(difftime(t1, t0, units = "secs"))))

    # --- ATE ---
    ate <- average_treatment_effect(cf)
    cat(sprintf("  ATE: %.4f (SE=%.4f, p=%.4f)\n",
                ate[1], ate[2], 2 * pnorm(-abs(ate[1] / ate[2]))))
    all_ate_results[[dv]] <- data.frame(
        dv = dv,
        ate = ate[1],
        se = ate[2],
        t_value = ate[1] / ate[2],
        p_value = 2 * pnorm(-abs(ate[1] / ate[2])),
        stringsAsFactors = FALSE
    )

    # --- BLP ---
    A <- as.matrix(df_clean[, blp_vars, drop = FALSE])
    blp <- best_linear_projection(cf, A)
    blp_mat <- as.matrix(blp)

    cat("\n  BLP (Best Linear Projection):\n")
    cat(sprintf("  %-35s %10s %10s %10s %10s\n",
                "Variable", "Estimate", "Std.Error", "t value", "Pr(>|t|)"))
    cat(paste(rep("-", 80), collapse = ""), "\n")

    for (i in seq_len(nrow(blp_mat))) {
        sig <- ""
        if (blp_mat[i, 4] < 0.001) sig <- "***"
        else if (blp_mat[i, 4] < 0.01) sig <- "**"
        else if (blp_mat[i, 4] < 0.05) sig <- "*"
        else if (blp_mat[i, 4] < 0.10) sig <- "."
        cat(sprintf("  %-35s %10.4f %10.4f %10.4f %10.4f %s\n",
                    rownames(blp_mat)[i],
                    blp_mat[i, 1], blp_mat[i, 2],
                    blp_mat[i, 3], blp_mat[i, 4], sig))
    }

    # Save BLP results
    blp_df <- data.frame(
        dv = dv,
        variable = rownames(blp_mat),
        estimate = blp_mat[, 1],
        std_error = blp_mat[, 2],
        t_value = blp_mat[, 3],
        p_value = blp_mat[, 4],
        stringsAsFactors = FALSE,
        row.names = NULL
    )
    all_blp_results[[dv]] <- blp_df

    # --- Feature Importance ---
    varimp <- variable_importance(cf)
    varimp_df <- data.frame(
        variable = colnames(X),
        importance = as.numeric(varimp),
        stringsAsFactors = FALSE
    )
    varimp_df <- varimp_df[order(-varimp_df$importance), ]

    cat("\n  Top 10 variable importance:\n")
    for (j in 1:min(10, nrow(varimp_df))) {
        cat(sprintf("    %2d. %-30s %.4f\n",
                    j, varimp_df$variable[j], varimp_df$importance[j]))
    }

    # --- Save per-DV outputs ---
    dv_dir <- file.path(OUTPUT_DIR, dv)
    dir.create(dv_dir, recursive = TRUE, showWarnings = FALSE)

    # Treatment effects (ITEs)
    tau_hat <- predict(cf)$predictions
    ite_df <- data.frame(
        fips = df_clean$fips,
        sld_dt = df_clean$sld_dt,
        republican_percentage = df_clean$republican_percentage,
        is_brand = df_clean$is_brand,
        ite = tau_hat,
        stringsAsFactors = FALSE
    )
    write.csv(ite_df, file.path(dv_dir, "treatment_effects.csv"),
              row.names = FALSE)

    # BLP results
    write.csv(blp_df, file.path(dv_dir, "blp_results.csv"),
              row.names = FALSE)

    # Variable importance
    write.csv(varimp_df, file.path(dv_dir, "variable_importance.csv"),
              row.names = FALSE)

    # ATE
    write.csv(all_ate_results[[dv]], file.path(dv_dir, "ate_results.csv"),
              row.names = FALSE)

    cat("\n")
}

# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------
cat("\n")
cat("============================================================\n")
cat("COMBINED RESULTS SUMMARY\n")
cat("============================================================\n\n")

# ATE summary
ate_df <- do.call(rbind, all_ate_results)
write.csv(ate_df, file.path(OUTPUT_DIR, "ate_summary.csv"), row.names = FALSE)

cat("ATE (Average Treatment Effect):\n")
cat(sprintf("  %-20s %10s %10s %10s\n", "DV", "ATE", "SE", "p-value"))
cat(paste(rep("-", 55), collapse = ""), "\n")
for (i in seq_len(nrow(ate_df))) {
    cat(sprintf("  %-20s %10.4f %10.4f %10.4f\n",
                ate_df$dv[i], ate_df$ate[i], ate_df$se[i], ate_df$p_value[i]))
}

# BLP summary (interaction term only)
cat("\nBLP — republicanStd_x_brand interaction:\n")
cat(sprintf("  %-20s %10s %10s %10s %10s\n",
            "DV", "Beta", "SE", "t", "p-value"))
cat(paste(rep("-", 65), collapse = ""), "\n")
blp_all <- do.call(rbind, all_blp_results)
write.csv(blp_all, file.path(OUTPUT_DIR, "blp_summary.csv"), row.names = FALSE)

for (dv in dvs) {
    row <- blp_all[blp_all$dv == dv & blp_all$variable == "republicanStd_x_brand", ]
    if (nrow(row) > 0) {
        sig <- ""
        if (row$p_value < 0.05) sig <- "*"
        else if (row$p_value < 0.10) sig <- "."
        cat(sprintf("  %-20s %10.4f %10.4f %10.4f %10.4f %s\n",
                    dv, row$estimate, row$std_error, row$t_value, row$p_value, sig))
    }
}

# ---------------------------------------------------------------------------
# Model specification log
# ---------------------------------------------------------------------------
spec_log <- paste0(
    "GRF BLP Production — G03 (Sex Hormones)\n",
    "=========================================\n\n",
    "Date: ", Sys.time(), "\n",
    "Drug category: G03\n",
    "Treatment: healthrisk_impact (2020 vs 2019)\n\n",
    "Causal Forest parameters:\n",
    "  num.trees = ", NUM_TREES, "\n",
    "  min.node.size = ", MIN_NODE, "\n",
    "  seed = ", SEED, "\n",
    "  clusters = fips\n",
    "  W.hat = constant (deterministic treatment)\n\n",
    "Heterogeneity features (7):\n",
    "  ", paste(hetero_features, collapse = ", "), "\n\n",
    "Controls (13):\n",
    "  ", paste(controls, collapse = ", "), "\n\n",
    "BLP A-matrix:\n",
    "  ", paste(blp_vars, collapse = ", "), "\n\n",
    "DVs: ", paste(dvs, collapse = ", "), "\n"
)
writeLines(spec_log, file.path(OUTPUT_DIR, "model_specification.txt"))

cat(sprintf("\nAll results saved to: %s\n", OUTPUT_DIR))
cat("DONE\n")
