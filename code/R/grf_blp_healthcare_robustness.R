#!/usr/bin/env Rscript
# ===========================================================================
# GRF BLP Robustness: 13 controls + 10 healthcare infrastructure vars
#
# Purpose: Confirm BLP results are stable when adding granular healthcare
#          infrastructure controls (physicians, hospitals, medicare).
#          Key test: log_transactions should remain null (~p=0.30).
#
# Comparison:
#   13-ctrl (production): log_sales p=0.044, log_price p=0.083, log_trans p=0.298
#   23-ctrl (this script): expected similar
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
OUTPUT_DIR <- file.path(OUTPUT_BASE, paste0("grf_blp_healthcare_robustness_", TIMESTAMP))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("============================================================\n")
cat("GRF BLP ROBUSTNESS — 13 + 10 Healthcare Infrastructure\n")
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

# Original 13 controls
controls_13 <- c(
    "dspns_qty_mean",
    "pharmacy_density",
    "insurance_coverage_mean",
    "patient_coverage_mean",
    "loc_id_nunique",
    "female_percentage",
    "over_65_percentage",
    "edu_bachelors",
    "income_middle_percentage",
    "unemployment_rate",
    "race_white",
    "evangelical_pct",
    "week"
)

# 10 additional healthcare infrastructure variables
healthcare_infra <- c(
    "physicians_family_medicine_f1200120",
    "physicians_medical_specialists_f0461820",
    "physicians_cardiovascular_f0463520",
    "physicians_psychiatry_f1013420",
    "hospitals_rural_f1403320",
    "hospitals_accreditation_f0887720",
    "hospitals_veterans_f0892920",
    "medicare_enrollment_f1554920",
    "medicare_advantage_f1319220",
    "medicare_prescription_drug_f1420720"
)

controls_23 <- c(controls_13, healthcare_infra)

blp_vars <- c("republican_percentage_std", "republicanStd_x_brand")

# Model parameters
NUM_TREES <- 3000
MIN_NODE  <- 15
SEED      <- 42

# ---------------------------------------------------------------------------
# DVs
# ---------------------------------------------------------------------------
dvs <- c("log_sales", "log_avg_price", "log_transactions")

# ---------------------------------------------------------------------------
# Run BLP for each DV
# ---------------------------------------------------------------------------
all_blp <- list()

for (dv in dvs) {
    cat("============================================================\n")
    cat(sprintf("DV: %s\n", dv))
    cat("============================================================\n")

    x_features <- c(hetero_features, controls_23)
    required <- c("healthrisk_impact", dv, x_features, "fips")
    df_clean <- df[complete.cases(df[, required]), ]
    cat(sprintf("  Observations: %d (dropped %d with NA)\n",
                nrow(df_clean), nrow(df) - nrow(df_clean)))
    cat(sprintf("  Clusters: %d counties\n", length(unique(df_clean$fips))))
    cat(sprintf("  Controls: %d (13 base + 10 healthcare infra)\n\n",
                length(controls_23)))

    X <- as.matrix(df_clean[, x_features])
    W <- df_clean$healthrisk_impact
    Y <- df_clean[[dv]]
    W_hat <- rep(mean(W), length(W))

    cat(sprintf("  Fitting causal forest (trees=%d) ... ", NUM_TREES))
    flush.console()
    t0 <- Sys.time()

    cf <- causal_forest(
        X = X, Y = Y, W = W, W.hat = W_hat,
        num.trees = NUM_TREES, min.node.size = MIN_NODE,
        seed = SEED, clusters = df_clean$fips
    )

    t1 <- Sys.time()
    cat(sprintf("done (%.1f sec)\n", as.numeric(difftime(t1, t0, units = "secs"))))

    # ATE
    ate <- average_treatment_effect(cf)
    cat(sprintf("  ATE: %.4f (SE=%.4f)\n", ate[1], ate[2]))

    # BLP
    A <- as.matrix(df_clean[, blp_vars, drop = FALSE])
    blp <- best_linear_projection(cf, A)
    blp_mat <- as.matrix(blp)

    cat("\n  BLP results:\n")
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
    all_blp[[dv]] <- blp_df
    cat("\n")
}

# ---------------------------------------------------------------------------
# Combined summary + comparison
# ---------------------------------------------------------------------------
cat("\n")
cat("============================================================\n")
cat("COMPARISON: 13-ctrl (production) vs 23-ctrl (+ healthcare)\n")
cat("============================================================\n\n")

blp_all <- do.call(rbind, all_blp)
write.csv(blp_all, file.path(OUTPUT_DIR, "blp_summary_23ctrl.csv"), row.names = FALSE)

# Production 13-ctrl results (hardcoded for comparison)
prod_13 <- data.frame(
    dv = c("log_sales", "log_avg_price", "log_transactions"),
    beta_13 = c(-0.0341, -0.0283, -0.0016),
    se_13 = c(0.0170, 0.0163, 0.0015),
    p_13 = c(0.044, 0.083, 0.298),
    stringsAsFactors = FALSE
)

cat(sprintf("%-18s | %10s %8s %8s | %10s %8s %8s\n",
            "DV", "beta_13", "SE_13", "p_13", "beta_23", "SE_23", "p_23"))
cat(paste(rep("-", 90), collapse = ""), "\n")

for (dv in dvs) {
    row23 <- blp_all[blp_all$dv == dv & blp_all$variable == "republicanStd_x_brand", ]
    row13 <- prod_13[prod_13$dv == dv, ]
    if (nrow(row23) > 0 && nrow(row13) > 0) {
        cat(sprintf("%-18s | %10.4f %8.4f %8.4f | %10.4f %8.4f %8.4f\n",
                    dv, row13$beta_13, row13$se_13, row13$p_13,
                    row23$estimate, row23$std_error, row23$p_value))
    }
}

cat(sprintf("\nResults saved to: %s\n", OUTPUT_DIR))
cat("DONE\n")
