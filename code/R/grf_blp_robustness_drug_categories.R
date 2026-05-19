#!/usr/bin/env Rscript
# ===========================================================================
# GRF BLP Robustness — Alternative Drug Categories (C01, C02, C10)
#
# C10 = Lipid-modifying agents (placebo test — no political salience)
# C01 = Cardiac therapy
# C02 = Antihypertensives
#
# Same methodology as main G03 analysis but:
#   - No drug subcategory columns (hetero_features = 3 vars only)
#   - Same 13 controls (minus evangelical_pct if missing data)
# ===========================================================================

library(grf)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
    BASE_DIR <- dirname(dirname(dirname(normalizePath(script_path))))
} else {
    BASE_DIR <- getwd()
}
DATA_DIR <- file.path(BASE_DIR, "data")
OUTPUT_BASE <- file.path(BASE_DIR, "data_output")

TIMESTAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
OUTPUT_DIR <- file.path(OUTPUT_BASE, paste0("grf_blp_robustness_", TIMESTAMP))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("============================================================\n")
cat("GRF BLP ROBUSTNESS — Alternative Drug Categories\n")
cat(sprintf("  Output: %s\n", OUTPUT_DIR))
cat("============================================================\n\n")

# ---------------------------------------------------------------------------
# Model parameters (same as main G03)
# ---------------------------------------------------------------------------
NUM_TREES <- 3000
MIN_NODE  <- 15
SEED      <- 42

# No drug subcategory columns for C01/C02/C10
hetero_features <- c(
    "republican_percentage_std", "is_brand", "republicanStd_x_brand"
)

controls <- c(
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

blp_vars <- c("republican_percentage_std", "republicanStd_x_brand")
dvs <- c("log_sales", "log_avg_price", "log_transactions")

# ---------------------------------------------------------------------------
# Drug categories to test
# ---------------------------------------------------------------------------
categories <- list(
    list(code = "C10", label = "Lipid-modifying agents (placebo)"),
    list(code = "C01", label = "Cardiac therapy"),
    list(code = "C02", label = "Antihypertensives")
)

# ---------------------------------------------------------------------------
# Load external data files (shared across categories)
# ---------------------------------------------------------------------------
hc_file <- file.path(DATA_DIR, "processed", "county_healthcare_indicators_2020.csv")
hc <- read.csv(hc_file)

relig_file <- file.path(DATA_DIR, "external", "county_religiosity_2020.csv")
relig <- read.csv(relig_file)
relig$fips <- as.integer(relig$fips)

# ---------------------------------------------------------------------------
# Master results collector
# ---------------------------------------------------------------------------
master_blp <- list()
master_ate <- list()

for (cat_info in categories) {
    cat_code <- cat_info$code
    cat_label <- cat_info$label

    cat("\n############################################################\n")
    cat(sprintf("  %s — %s\n", cat_code, cat_label))
    cat("############################################################\n\n")

    # Load data
    data_file <- file.path(DATA_DIR, "final_data",
        sprintf("merged_sldDt_%s_daily_covid_remove_healthRisk_addUrbanRural.csv",
                cat_code))
    if (!file.exists(data_file)) {
        cat(sprintf("  WARNING: Data file not found, skipping: %s\n", data_file))
        next
    }
    df <- read.csv(data_file)
    df <- merge(df, hc, by = "fips", all.x = TRUE)
    df <- merge(df, relig[, c("fips", "evangelical_pct")],
                by = "fips", all.x = TRUE)

    # Derived variables
    df$avg_price <- df$rx_prc_amt_sum / df$transaction_count
    df$log_avg_price <- log1p(df$avg_price)
    if (!"log_sales" %in% names(df)) df$log_sales <- log1p(df$rx_prc_amt_sum)
    if (!"log_transactions" %in% names(df)) {
        df$log_transactions <- log1p(df$transaction_count)
    }
    df$republican_percentage_std <- scale(df$republican_percentage)[, 1]
    df$republicanStd_x_brand <- df$republican_percentage_std * df$is_brand

    cat(sprintf("  Data: %d rows, %d counties\n",
                nrow(df), length(unique(df$fips))))

    # Check which controls are available
    avail_controls <- controls[controls %in% names(df)]
    missing <- setdiff(controls, avail_controls)
    if (length(missing) > 0) {
        cat(sprintf("  Missing controls (dropped): %s\n",
                    paste(missing, collapse = ", ")))
    }

    x_features <- c(hetero_features, avail_controls)

    # Create category output directory
    cat_dir <- file.path(OUTPUT_DIR, cat_code)
    dir.create(cat_dir, recursive = TRUE, showWarnings = FALSE)

    for (dv in dvs) {
        cat(sprintf("\n  --- %s: %s ---\n", cat_code, dv))

        required <- c("healthrisk_impact", dv, x_features, "fips")
        df_clean <- df[complete.cases(df[, required]), ]
        cat(sprintf("  Observations: %d, Clusters: %d\n",
                    nrow(df_clean), length(unique(df_clean$fips))))

        if (nrow(df_clean) < 100) {
            cat("  SKIP: Too few observations\n")
            next
        }

        X <- as.matrix(df_clean[, x_features])
        W <- df_clean$healthrisk_impact
        Y <- df_clean[[dv]]
        W_hat <- rep(mean(W), length(W))

        cat("  Fitting causal forest ... ")
        flush.console()
        t0 <- Sys.time()

        cf <- tryCatch(
            causal_forest(
                X = X, Y = Y, W = W, W.hat = W_hat,
                num.trees = NUM_TREES, min.node.size = MIN_NODE,
                seed = SEED, clusters = df_clean$fips
            ),
            error = function(e) {
                cat(sprintf("ERROR: %s\n", e$message))
                NULL
            }
        )
        if (is.null(cf)) next

        t1 <- Sys.time()
        cat(sprintf("done (%.1f sec)\n",
                    as.numeric(difftime(t1, t0, units = "secs"))))

        # ATE
        ate <- average_treatment_effect(cf)
        ate_p <- 2 * pnorm(-abs(ate[1] / ate[2]))
        cat(sprintf("  ATE: %.4f (SE=%.4f, p=%.4f)\n", ate[1], ate[2], ate_p))
        master_ate[[paste0(cat_code, "_", dv)]] <- data.frame(
            category = cat_code, dv = dv,
            ate = ate[1], se = ate[2], p_value = ate_p,
            n_obs = nrow(df_clean), n_clusters = length(unique(df_clean$fips)),
            stringsAsFactors = FALSE
        )

        # BLP
        A <- as.matrix(df_clean[, blp_vars, drop = FALSE])
        blp <- best_linear_projection(cf, A)
        blp_mat <- as.matrix(blp)

        cat("  BLP results:\n")
        for (i in seq_len(nrow(blp_mat))) {
            sig <- ifelse(blp_mat[i, 4] < 0.05, "*",
                          ifelse(blp_mat[i, 4] < 0.10, ".", " "))
            cat(sprintf("    %-35s beta=%.4f  p=%.4f %s\n",
                        rownames(blp_mat)[i],
                        blp_mat[i, 1], blp_mat[i, 4], sig))
        }

        blp_df <- data.frame(
            category = cat_code, dv = dv,
            variable = rownames(blp_mat),
            estimate = blp_mat[, 1], std_error = blp_mat[, 2],
            t_value = blp_mat[, 3], p_value = blp_mat[, 4],
            n_obs = nrow(df_clean), n_clusters = length(unique(df_clean$fips)),
            stringsAsFactors = FALSE, row.names = NULL
        )
        master_blp[[paste0(cat_code, "_", dv)]] <- blp_df

        # Save per-DV results
        dv_dir <- file.path(cat_dir, dv)
        dir.create(dv_dir, recursive = TRUE, showWarnings = FALSE)
        write.csv(blp_df, file.path(dv_dir, "blp_results.csv"), row.names = FALSE)
    }
}

# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------
cat("\n\n")
cat("============================================================\n")
cat("COMBINED ROBUSTNESS SUMMARY\n")
cat("============================================================\n\n")

if (length(master_blp) > 0) {
    all_blp <- do.call(rbind, master_blp)
    write.csv(all_blp, file.path(OUTPUT_DIR, "blp_all_categories.csv"),
              row.names = FALSE)

    # Show interaction term only
    interaction_rows <- all_blp[all_blp$variable == "republicanStd_x_brand", ]
    cat(sprintf("%-6s %-20s %10s %10s %10s %8s %8s\n",
                "Cat", "DV", "Beta", "SE", "p-value", "n_obs", "n_clust"))
    cat(paste(rep("-", 90), collapse = ""), "\n")
    for (i in seq_len(nrow(interaction_rows))) {
        r <- interaction_rows[i, ]
        sig <- ifelse(r$p_value < 0.05, "*",
                      ifelse(r$p_value < 0.10, ".", " "))
        cat(sprintf("%-6s %-20s %10.4f %10.4f %10.4f%s %8d %8d\n",
                    r$category, r$dv, r$estimate, r$std_error,
                    r$p_value, sig, r$n_obs, r$n_clusters))
    }
}

if (length(master_ate) > 0) {
    all_ate <- do.call(rbind, master_ate)
    write.csv(all_ate, file.path(OUTPUT_DIR, "ate_all_categories.csv"),
              row.names = FALSE)
}

cat(sprintf("\nResults saved to: %s\n", OUTPUT_DIR))
cat("DONE\n")
