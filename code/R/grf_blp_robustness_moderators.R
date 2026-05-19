#!/usr/bin/env Rscript
# Robustness: BLP with Income and Urban-Rural Moderators
# Tests whether the Republican × Brand effect is moderated by income or urban-rural.
# Uses BLP A-matrix expansion (proper inference) instead of OLS on ITEs.

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
                        paste0("grf_blp_robustness_moderators_", TIMESTAMP))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=== GRF BLP Robustness: Income & Urban-Rural Moderators ===\n")
cat(sprintf("Output: %s\n\n", OUTPUT_DIR))

# === Load & prepare data (same as production) ===
data_path <- file.path(BASE_DIR, "data", "final_data",
                       "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv")
hc_path <- file.path(BASE_DIR, "data", "processed",
                     "county_healthcare_indicators_2020.csv")
relig_path <- file.path(BASE_DIR, "data", "external",
                        "county_religiosity_2020.csv")

df <- read.csv(data_path)
hc <- read.csv(hc_path)
relig <- read.csv(relig_path)

# Create DVs
df$avg_price <- df$rx_prc_amt_sum / df$transaction_count
df$log_avg_price <- log1p(df$avg_price)
if (!"log_sales" %in% names(df)) df$log_sales <- log1p(df$rx_prc_amt_sum)
if (!"log_transactions" %in% names(df)) df$log_transactions <- log1p(df$transaction_count)

# Merge
df$fips <- as.integer(df$fips)
hc$fips <- as.integer(hc$fips)
relig$fips <- as.integer(relig$fips)
df <- merge(df, hc, by = "fips", all.x = TRUE)
df <- merge(df, relig[, c("fips", "evangelical_pct")], by = "fips", all.x = TRUE)

# Standardize
df$republican_percentage_std <- scale(df$republican_percentage)[, 1]
df$republicanStd_x_brand <- df$republican_percentage_std * df$is_brand

# Income moderator: income_middle_percentage (same as in 13-control spec)
df$income_mid_std <- scale(df$income_middle_percentage)[, 1]
df$repBrand_x_income <- df$republicanStd_x_brand * df$income_mid_std

# Urban-rural moderator
df$urban_rural_std <- scale(df$urban_rural_value_lessBetter)[, 1]
df$repBrand_x_urbanrural <- df$republicanStd_x_brand * df$urban_rural_std

cat(sprintf("Data: %d rows, %d counties\n", nrow(df), length(unique(df$fips))))

# === Specification (same as production) ===
HETERO_FEATURES <- c(
    "republican_percentage_std", "is_brand", "republicanStd_x_brand",
    "drug_ANDROGENS_ANABOLIC_sum", "drug_CONTRACEPTIVES_sum",
    "drug_ESTROGENS_sum", "drug_PROGESTINS_sum"
)

CONTROLS <- c(
    "dspns_qty_mean", "pharmacy_density", "insurance_coverage_mean",
    "patient_coverage_mean", "loc_id_nunique", "female_percentage",
    "over_65_percentage", "edu_bachelors", "income_middle_percentage",
    "unemployment_rate", "race_white", "evangelical_pct", "week"
)

NUM_TREES <- 3000
MIN_NODE <- 15
SEED <- 42

# === Clean data ===
all_vars <- c("log_sales", HETERO_FEATURES, CONTROLS, "healthrisk_impact", "fips",
              "income_mid_std", "repBrand_x_income",
              "urban_rural_std", "repBrand_x_urbanrural")
df_clean <- df[complete.cases(df[, all_vars]), ]
cat(sprintf("Clean data: %d rows, %d counties\n\n", nrow(df_clean), length(unique(df_clean$fips))))

# === Fit causal forest ONCE (same as production) ===
cat("Fitting causal forest (log_sales)... ")
flush.console()
t0 <- Sys.time()

Y <- df_clean$log_sales
W <- df_clean$healthrisk_impact
X <- as.matrix(df_clean[, c(HETERO_FEATURES, CONTROLS)])
W_hat <- rep(mean(W), length(W))

cf <- causal_forest(
    X = X, Y = Y, W = W, W.hat = W_hat,
    num.trees = NUM_TREES, min.node.size = MIN_NODE,
    seed = SEED, clusters = df_clean$fips
)
t1 <- Sys.time()
cat(sprintf("done (%.1f sec)\n\n", as.numeric(difftime(t1, t0, units = "secs"))))

# === BLP 1: Production (baseline) ===
cat("--- BLP: Production (baseline) ---\n")
A_prod <- as.matrix(df_clean[, c("republican_percentage_std", "republicanStd_x_brand")])
blp_prod <- best_linear_projection(cf, A_prod)
blp_prod_mat <- as.matrix(blp_prod)
for (i in seq_len(nrow(blp_prod_mat))) {
    sig <- ifelse(blp_prod_mat[i, 4] < 0.05, "*", ifelse(blp_prod_mat[i, 4] < 0.10, ".", ""))
    cat(sprintf("  %-35s β=%.4f SE=%.4f p=%.4f %s\n",
                rownames(blp_prod_mat)[i],
                blp_prod_mat[i, 1], blp_prod_mat[i, 2], blp_prod_mat[i, 4], sig))
}

# === BLP 2: Income moderator ===
cat("\n--- BLP: Income Moderator (income_middle_percentage) ---\n")
A_income <- as.matrix(df_clean[, c("republican_percentage_std", "republicanStd_x_brand",
                                    "income_mid_std", "repBrand_x_income")])
blp_income <- best_linear_projection(cf, A_income)
blp_income_mat <- as.matrix(blp_income)
for (i in seq_len(nrow(blp_income_mat))) {
    sig <- ifelse(blp_income_mat[i, 4] < 0.05, "*", ifelse(blp_income_mat[i, 4] < 0.10, ".", ""))
    cat(sprintf("  %-35s β=%.4f SE=%.4f p=%.4f %s\n",
                rownames(blp_income_mat)[i],
                blp_income_mat[i, 1], blp_income_mat[i, 2], blp_income_mat[i, 4], sig))
}

# === BLP 3: Urban-Rural moderator ===
cat("\n--- BLP: Urban-Rural Moderator ---\n")
A_urban <- as.matrix(df_clean[, c("republican_percentage_std", "republicanStd_x_brand",
                                   "urban_rural_std", "repBrand_x_urbanrural")])
blp_urban <- best_linear_projection(cf, A_urban)
blp_urban_mat <- as.matrix(blp_urban)
for (i in seq_len(nrow(blp_urban_mat))) {
    sig <- ifelse(blp_urban_mat[i, 4] < 0.05, "*", ifelse(blp_urban_mat[i, 4] < 0.10, ".", ""))
    cat(sprintf("  %-35s β=%.4f SE=%.4f p=%.4f %s\n",
                rownames(blp_urban_mat)[i],
                blp_urban_mat[i, 1], blp_urban_mat[i, 2], blp_urban_mat[i, 4], sig))
}

# === Save all results ===
save_blp <- function(mat, label) {
    data.frame(
        model = label,
        variable = rownames(mat),
        estimate = mat[, 1],
        std_error = mat[, 2],
        t_value = mat[, 3],
        p_value = mat[, 4],
        stringsAsFactors = FALSE
    )
}

all_results <- rbind(
    save_blp(blp_prod_mat, "production"),
    save_blp(blp_income_mat, "income_moderator"),
    save_blp(blp_urban_mat, "urbanrural_moderator")
)
rownames(all_results) <- NULL
write.csv(all_results, file.path(OUTPUT_DIR, "blp_moderator_results.csv"), row.names = FALSE)

# === Summary ===
cat("\n=== Summary ===\n")
cat("Income moderator (repBrand × income_mid_std):\n")
r <- blp_income_mat["repBrand_x_income", ]
cat(sprintf("  β = %.4f, SE = %.4f, p = %.4f\n", r[1], r[2], r[4]))

cat("Urban-Rural moderator (repBrand × urban_rural_std):\n")
r <- blp_urban_mat["repBrand_x_urbanrural", ]
cat(sprintf("  β = %.4f, SE = %.4f, p = %.4f\n", r[1], r[2], r[4]))

cat(sprintf("\nMain interaction (repStd_x_brand) remains:\n"))
cat(sprintf("  Production:       β = %.4f, p = %.4f\n",
            blp_prod_mat["republicanStd_x_brand", 1], blp_prod_mat["republicanStd_x_brand", 4]))
cat(sprintf("  With income mod:  β = %.4f, p = %.4f\n",
            blp_income_mat["republicanStd_x_brand", 1], blp_income_mat["republicanStd_x_brand", 4]))
cat(sprintf("  With urban mod:   β = %.4f, p = %.4f\n",
            blp_urban_mat["republicanStd_x_brand", 1], blp_urban_mat["republicanStd_x_brand", 4]))

cat(sprintf("\nResults saved to: %s\n", OUTPUT_DIR))
