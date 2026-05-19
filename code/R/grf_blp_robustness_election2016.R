#!/usr/bin/env Rscript
# Robustness: GRF + BLP using 2016 Election Results
# Tests whether results are robust to the specific election year used.
# Uses identical specification as production (13 controls, 3000 trees)
# but replaces 2020 Republican vote share with 2016.

suppressPackageStartupMessages({
    library(grf)
})

# === Paths ===
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("--file=", args, value = TRUE)
if (length(file_arg) > 0) {
    SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
} else {
    SCRIPT_DIR <- getwd()
}
BASE_DIR <- dirname(dirname(SCRIPT_DIR))

TIMESTAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
OUTPUT_DIR <- file.path(BASE_DIR, "data_output",
                        paste0("grf_blp_robustness_election2016_", TIMESTAMP))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=== GRF BLP Robustness: 2016 Election ===\n")
cat(sprintf("Output: %s\n\n", OUTPUT_DIR))

# === Load data ===
data_path <- file.path(BASE_DIR, "data", "final_data",
                       "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv")
election_path <- file.path(BASE_DIR, "data",
                           "2016_US_County_Level_Presidential_Results.csv")
relig_path <- file.path(BASE_DIR, "data", "external",
                        "county_religiosity_2020.csv")

hc_path <- file.path(BASE_DIR, "data", "processed",
                     "county_healthcare_indicators_2020.csv")

df <- read.csv(data_path)
election2016 <- read.csv(election_path)
relig <- read.csv(relig_path)
hc <- read.csv(hc_path)

cat(sprintf("Main data: %d rows\n", nrow(df)))

# === Create DVs ===
df$avg_price <- df$rx_prc_amt_sum / df$transaction_count
df$log_avg_price <- log1p(df$avg_price)
if (!"log_sales" %in% names(df)) df$log_sales <- log1p(df$rx_prc_amt_sum)
if (!"log_transactions" %in% names(df)) df$log_transactions <- log1p(df$transaction_count)

# === Merge healthcare ===
df$fips <- as.integer(df$fips)
hc$fips <- as.integer(hc$fips)
df <- merge(df, hc, by = "fips", all.x = TRUE)

# === Merge 2016 election ===
election2016$fips <- as.integer(election2016$combined_fips)
election2016$republican_pct_2016 <- election2016$per_gop * 100
election2016 <- election2016[, c("fips", "republican_pct_2016")]
election2016 <- election2016[!duplicated(election2016$fips), ]

df$fips <- as.integer(df$fips)
df <- merge(df, election2016, by = "fips", all.x = TRUE)

# Merge religiosity
relig$fips <- as.integer(relig$fips)
df <- merge(df, relig[, c("fips", "evangelical_pct")], by = "fips", all.x = TRUE)

# Drop rows missing 2016 data
n_before <- nrow(df)
df <- df[!is.na(df$republican_pct_2016), ]
cat(sprintf("After 2016 merge: %d rows (dropped %d)\n", nrow(df), n_before - nrow(df)))

# === Create 2016 variables ===
df$republican_2016_std <- scale(df$republican_pct_2016)[, 1]
df$republicanStd2016_x_brand <- df$republican_2016_std * df$is_brand

# === Specification (same as production but with 2016 political vars) ===
HETERO_FEATURES <- c(
    "republican_2016_std", "is_brand", "republicanStd2016_x_brand",
    "drug_ANDROGENS_ANABOLIC_sum", "drug_CONTRACEPTIVES_sum",
    "drug_ESTROGENS_sum", "drug_PROGESTINS_sum"
)

CONTROLS <- c(
    "dspns_qty_mean", "pharmacy_density", "insurance_coverage_mean",
    "patient_coverage_mean", "loc_id_nunique", "female_percentage",
    "over_65_percentage", "edu_bachelors", "income_middle_percentage",
    "unemployment_rate", "race_white", "evangelical_pct", "week"
)

BLP_VARS <- c("republican_2016_std", "republicanStd2016_x_brand")

NUM_TREES <- 3000
MIN_NODE <- 15
SEED <- 42

DVS <- c("log_sales", "log_avg_price", "log_transactions")

# === Clean data ===
all_vars <- c(DVS, HETERO_FEATURES, CONTROLS, "healthrisk_impact", "fips")
df_clean <- df[complete.cases(df[, all_vars]), ]
cat(sprintf("Clean data: %d rows, %d counties\n\n", nrow(df_clean), length(unique(df_clean$fips))))

# === Run GRF + BLP for each DV ===
all_blp <- list()

for (dv in DVS) {
    cat(sprintf("--- %s ---\n", dv))

    Y <- df_clean[[dv]]
    W <- df_clean$healthrisk_impact
    X <- as.matrix(df_clean[, c(HETERO_FEATURES, CONTROLS)])
    W_hat <- rep(mean(W), length(W))

    cat("  Fitting causal forest... ")
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
    cat(sprintf("  ATE: %.4f (SE=%.4f, p=%.4f)\n",
                ate[1], ate[2], 2 * pnorm(-abs(ate[1] / ate[2]))))

    # BLP
    A <- as.matrix(df_clean[, BLP_VARS, drop = FALSE])
    blp <- best_linear_projection(cf, A)
    blp_mat <- as.matrix(blp)

    cat("  BLP:\n")
    for (i in seq_len(nrow(blp_mat))) {
        sig <- ifelse(blp_mat[i, 4] < 0.05, "*", ifelse(blp_mat[i, 4] < 0.10, ".", ""))
        cat(sprintf("    %-35s β=%.4f SE=%.4f p=%.4f %s\n",
                    rownames(blp_mat)[i],
                    blp_mat[i, 1], blp_mat[i, 2], blp_mat[i, 4], sig))
    }

    blp_df <- data.frame(
        dv = dv,
        variable = rownames(blp_mat),
        estimate = blp_mat[, 1],
        std_error = blp_mat[, 2],
        t_value = blp_mat[, 3],
        p_value = blp_mat[, 4],
        stringsAsFactors = FALSE
    )
    all_blp[[dv]] <- blp_df
    cat("\n")
}

# === Save ===
blp_all <- do.call(rbind, all_blp)
rownames(blp_all) <- NULL
write.csv(blp_all, file.path(OUTPUT_DIR, "blp_election2016.csv"), row.names = FALSE)

# === Comparison table ===
cat("\n=== Comparison: 2016 vs 2020 Election (interaction on log_sales) ===\n")
interaction_row <- blp_all[blp_all$dv == "log_sales" &
                           blp_all$variable == "republicanStd2016_x_brand", ]
cat(sprintf("  2016: β = %.4f, SE = %.4f, p = %.4f\n",
            interaction_row$estimate, interaction_row$std_error, interaction_row$p_value))
cat("  2020: β = -0.0341, SE = 0.0170, p = 0.044 (from production)\n")

cat(sprintf("\nResults saved to: %s\n", OUTPUT_DIR))
