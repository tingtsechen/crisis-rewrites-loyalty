#!/usr/bin/env Rscript
# ===========================================================================
# GRF BLP Subgroup Analysis — G03 Drug Subcategories
#
# Filters main G03 data by drug subcategory (drug_{cat}_sum > 0):
#   ANDROGENS_ANABOLIC, ESTROGENS, PROGESTINS, CONTRACEPTIVES
#
# Same methodology as main G03 production script.
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
OUTPUT_DIR <- file.path(OUTPUT_BASE, paste0("grf_blp_subgroup_G03_", TIMESTAMP))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("============================================================\n")
cat("GRF BLP SUBGROUP — G03 Drug Subcategories\n")
cat(sprintf("  Output: %s\n", OUTPUT_DIR))
cat("============================================================\n\n")

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
NUM_TREES <- 3000
MIN_NODE  <- 15
SEED      <- 42

hetero_features <- c(
    "republican_percentage_std", "is_brand", "republicanStd_x_brand",
    "drug_ANDROGENS_ANABOLIC_sum", "drug_CONTRACEPTIVES_sum",
    "drug_ESTROGENS_sum", "drug_PROGESTINS_sum"
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
# Drug subcategories
# ---------------------------------------------------------------------------
subcategories <- c(
    "ANDROGENS_ANABOLIC",
    "ESTROGENS",
    "PROGESTINS",
    "CONTRACEPTIVES"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
main_file <- file.path(DATA_DIR, "final_data",
    "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv")
df_full <- read.csv(main_file)

# Healthcare indicators
hc_file <- file.path(DATA_DIR, "processed", "county_healthcare_indicators_2020.csv")
hc <- read.csv(hc_file)
df_full <- merge(df_full, hc, by = "fips", all.x = TRUE)

# Religiosity
relig_file <- file.path(DATA_DIR, "external", "county_religiosity_2020.csv")
relig <- read.csv(relig_file)
relig$fips <- as.integer(relig$fips)
df_full <- merge(df_full, relig[, c("fips", "evangelical_pct")],
                 by = "fips", all.x = TRUE)

# Derived variables
df_full$avg_price <- df_full$rx_prc_amt_sum / df_full$transaction_count
df_full$log_avg_price <- log1p(df_full$avg_price)
if (!"log_sales" %in% names(df_full)) {
    df_full$log_sales <- log1p(df_full$rx_prc_amt_sum)
}
if (!"log_transactions" %in% names(df_full)) {
    df_full$log_transactions <- log1p(df_full$transaction_count)
}
df_full$republican_percentage_std <- scale(df_full$republican_percentage)[, 1]
df_full$republicanStd_x_brand <- df_full$republican_percentage_std * df_full$is_brand

cat(sprintf("Full G03 data: %d rows, %d counties\n\n",
            nrow(df_full), length(unique(df_full$fips))))

x_features <- c(hetero_features, controls)

# ---------------------------------------------------------------------------
# Master results collector
# ---------------------------------------------------------------------------
master_blp <- list()
master_ate <- list()

for (subcat in subcategories) {
    cat("\n############################################################\n")
    cat(sprintf("  Subcategory: %s\n", subcat))
    cat("############################################################\n\n")

    # Filter: rows where this drug subcategory has activity
    drug_col <- paste0("drug_", subcat, "_sum")
    if (!drug_col %in% names(df_full)) {
        cat(sprintf("  WARNING: Column %s not found, skipping\n", drug_col))
        next
    }
    df <- df_full[df_full[[drug_col]] > 0, ]
    cat(sprintf("  Filtered data: %d rows (%.1f%% of full), %d counties\n",
                nrow(df), 100 * nrow(df) / nrow(df_full),
                length(unique(df$fips))))

    if (nrow(df) < 200) {
        cat("  SKIP: Too few observations after filtering\n")
        next
    }

    # Create subcategory output directory
    sub_dir <- file.path(OUTPUT_DIR, subcat)
    dir.create(sub_dir, recursive = TRUE, showWarnings = FALSE)

    for (dv in dvs) {
        cat(sprintf("\n  --- %s: %s ---\n", subcat, dv))

        required <- c("healthrisk_impact", dv, x_features, "fips")
        df_clean <- df[complete.cases(df[, required]), ]
        n_clusters <- length(unique(df_clean$fips))
        cat(sprintf("  Observations: %d, Clusters: %d\n",
                    nrow(df_clean), n_clusters))

        if (nrow(df_clean) < 100 || n_clusters < 10) {
            cat("  SKIP: Too few observations or clusters\n")
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
        master_ate[[paste0(subcat, "_", dv)]] <- data.frame(
            subcategory = subcat, dv = dv,
            ate = ate[1], se = ate[2], p_value = ate_p,
            n_obs = nrow(df_clean), n_clusters = n_clusters,
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
            subcategory = subcat, dv = dv,
            variable = rownames(blp_mat),
            estimate = blp_mat[, 1], std_error = blp_mat[, 2],
            t_value = blp_mat[, 3], p_value = blp_mat[, 4],
            n_obs = nrow(df_clean), n_clusters = n_clusters,
            stringsAsFactors = FALSE, row.names = NULL
        )
        master_blp[[paste0(subcat, "_", dv)]] <- blp_df

        # Save per-DV
        dv_dir <- file.path(sub_dir, dv)
        dir.create(dv_dir, recursive = TRUE, showWarnings = FALSE)
        write.csv(blp_df, file.path(dv_dir, "blp_results.csv"), row.names = FALSE)
    }
}

# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------
cat("\n\n")
cat("============================================================\n")
cat("SUBGROUP ANALYSIS SUMMARY\n")
cat("============================================================\n\n")

if (length(master_blp) > 0) {
    all_blp <- do.call(rbind, master_blp)
    write.csv(all_blp, file.path(OUTPUT_DIR, "blp_all_subcategories.csv"),
              row.names = FALSE)

    # Show interaction term
    interaction_rows <- all_blp[all_blp$variable == "republicanStd_x_brand", ]
    cat(sprintf("%-22s %-20s %10s %10s %10s %8s %8s\n",
                "Subcategory", "DV", "Beta", "SE", "p-value", "n_obs", "n_clust"))
    cat(paste(rep("-", 100), collapse = ""), "\n")
    for (i in seq_len(nrow(interaction_rows))) {
        r <- interaction_rows[i, ]
        sig <- ifelse(r$p_value < 0.05, "*",
                      ifelse(r$p_value < 0.10, ".", " "))
        cat(sprintf("%-22s %-20s %10.4f %10.4f %10.4f%s %8d %8d\n",
                    r$subcategory, r$dv, r$estimate, r$std_error,
                    r$p_value, sig, r$n_obs, r$n_clusters))
    }
}

if (length(master_ate) > 0) {
    all_ate <- do.call(rbind, master_ate)
    write.csv(all_ate, file.path(OUTPUT_DIR, "ate_all_subcategories.csv"),
              row.names = FALSE)
}

cat(sprintf("\nResults saved to: %s\n", OUTPUT_DIR))
cat("DONE\n")
