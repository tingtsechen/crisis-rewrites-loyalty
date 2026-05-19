#!/usr/bin/env Rscript
# ===========================================================================
# VIF check: republican_percentage_std after adding evangelical_pct
# Also check correlations between key variables
# ===========================================================================

library(grf)

# Paths
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
    BASE_DIR <- dirname(dirname(normalizePath(script_path)))
} else {
    BASE_DIR <- getwd()
}
DATA_DIR <- file.path(BASE_DIR, "data")

# Load data
df <- read.csv(file.path(DATA_DIR, "final_data",
    "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv"))
hc <- read.csv(file.path(DATA_DIR, "processed",
    "county_healthcare_indicators_2020.csv"))
df <- merge(df, hc, by = "fips", all.x = TRUE)
relig <- read.csv(file.path(DATA_DIR, "external",
    "county_religiosity_2020.csv"))
relig$fips <- as.integer(relig$fips)
df <- merge(df, relig[, c("fips", "evangelical_pct")],
            by = "fips", all.x = TRUE)

df$republican_percentage_std <- scale(df$republican_percentage)[, 1]
df$republicanStd_x_brand <- df$republican_percentage_std * df$is_brand

cat("============================================================\n")
cat("1. CORRELATIONS\n")
cat("============================================================\n\n")

# Correlation between evangelical_pct and republican_percentage
r <- cor(df$evangelical_pct, df$republican_percentage, use = "complete.obs")
cat(sprintf("  r(evangelical_pct, republican_percentage) = %.4f\n", r))

r_std <- cor(df$evangelical_pct, df$republican_percentage_std, use = "complete.obs")
cat(sprintf("  r(evangelical_pct, republican_percentage_std) = %.4f\n", r_std))

# Correlations among all 13 controls + hetero features
controls <- c("dspns_qty_mean", "pharmacy_density", "insurance_coverage_mean",
              "patient_coverage_mean", "loc_id_nunique",
              "female_percentage", "over_65_percentage",
              "edu_bachelors", "income_middle_percentage",
              "unemployment_rate", "race_white", "evangelical_pct", "week")

key_vars <- c("republican_percentage_std", "evangelical_pct",
              "is_brand", "republicanStd_x_brand")

cat("\n  Correlations with republican_percentage_std:\n")
for (v in controls) {
    if (v %in% names(df)) {
        r_val <- cor(df$republican_percentage_std, df[[v]], use = "complete.obs")
        flag <- ifelse(abs(r_val) > 0.4, " ***HIGH***", "")
        cat(sprintf("    %-30s r = %7.4f%s\n", v, r_val, flag))
    }
}

cat("\n============================================================\n")
cat("2. VIF ANALYSIS\n")
cat("============================================================\n\n")

# VIF for republican_percentage_std
# VIF = 1 / (1 - R²) where R² is from regressing the variable on all other predictors

# Model WITHOUT evangelical_pct
controls_no_evang <- controls[controls != "evangelical_pct"]
x_no_evang <- c("republican_percentage_std", "is_brand",
                "republicanStd_x_brand", controls_no_evang)
df_clean <- df[complete.cases(df[, c(x_no_evang, "evangelical_pct")]), ]

# VIF for republican_percentage_std WITHOUT evangelical
fml_no <- as.formula(paste("republican_percentage_std ~",
    paste(setdiff(x_no_evang, "republican_percentage_std"), collapse = " + ")))
r2_no <- summary(lm(fml_no, data = df_clean))$r.squared
vif_no <- 1 / (1 - r2_no)
cat(sprintf("  VIF(republican_percentage_std) WITHOUT evangelical_pct:\n"))
cat(sprintf("    R² = %.4f, VIF = %.2f\n\n", r2_no, vif_no))

# VIF for republican_percentage_std WITH evangelical
x_with_evang <- c("republican_percentage_std", "is_brand",
                  "republicanStd_x_brand", controls)
fml_with <- as.formula(paste("republican_percentage_std ~",
    paste(setdiff(x_with_evang, "republican_percentage_std"), collapse = " + ")))
r2_with <- summary(lm(fml_with, data = df_clean))$r.squared
vif_with <- 1 / (1 - r2_with)
cat(sprintf("  VIF(republican_percentage_std) WITH evangelical_pct:\n"))
cat(sprintf("    R² = %.4f, VIF = %.2f\n\n", r2_with, vif_with))

# VIF for ALL variables in the full model
cat("  VIF for all variables in the full model (with evangelical_pct):\n")
all_x <- c("republican_percentage_std", "is_brand",
           "republicanStd_x_brand", controls)
for (v in all_x) {
    fml <- as.formula(paste(v, "~", paste(setdiff(all_x, v), collapse = " + ")))
    r2 <- summary(lm(fml, data = df_clean))$r.squared
    vif_val <- 1 / (1 - r2)
    flag <- ifelse(vif_val > 5, " ***HIGH***", ifelse(vif_val > 3, " *MODERATE*", ""))
    cat(sprintf("    %-35s VIF = %6.2f%s\n", v, vif_val, flag))
}

cat("\n============================================================\n")
cat("3. BLP COMPARISON: WITH vs WITHOUT evangelical_pct\n")
cat("============================================================\n\n")

# Hetero features
hetero <- c("republican_percentage_std", "is_brand", "republicanStd_x_brand",
            "drug_ANDROGENS_ANABOLIC_sum", "drug_CONTRACEPTIVES_sum",
            "drug_ESTROGENS_sum", "drug_PROGESTINS_sum")
blp_vars <- c("republican_percentage_std", "republicanStd_x_brand")

df_clean$log_sales <- log1p(df_clean$rx_prc_amt_sum)

# WITHOUT evangelical
x_feat_no <- c(hetero, controls_no_evang)
X_no <- as.matrix(df_clean[, x_feat_no])
W <- df_clean$healthrisk_impact
Y <- df_clean$log_sales
W_hat <- rep(mean(W), length(W))

cat("  Fitting CF WITHOUT evangelical_pct ... ")
flush.console()
cf_no <- causal_forest(X = X_no, Y = Y, W = W, W.hat = W_hat,
                       num.trees = 3000, min.node.size = 15,
                       seed = 42, clusters = df_clean$fips)
A_no <- as.matrix(df_clean[, blp_vars])
blp_no <- as.matrix(best_linear_projection(cf_no, A_no))
cat("done\n")
cat(sprintf("    interaction: beta=%.4f, SE=%.4f, p=%.4f\n",
            blp_no[3,1], blp_no[3,2], blp_no[3,4]))

# WITH evangelical
x_feat_with <- c(hetero, controls)
X_with <- as.matrix(df_clean[, x_feat_with])

cat("  Fitting CF WITH evangelical_pct ... ")
flush.console()
cf_with <- causal_forest(X = X_with, Y = Y, W = W, W.hat = W_hat,
                         num.trees = 3000, min.node.size = 15,
                         seed = 42, clusters = df_clean$fips)
A_with <- as.matrix(df_clean[, blp_vars])
blp_with <- as.matrix(best_linear_projection(cf_with, A_with))
cat("done\n")
cat(sprintf("    interaction: beta=%.4f, SE=%.4f, p=%.4f\n",
            blp_with[3,1], blp_with[3,2], blp_with[3,4]))

cat("\n  Comparison:\n")
cat(sprintf("    WITHOUT evangelical: beta=%.4f, p=%.4f\n", blp_no[3,1], blp_no[3,4]))
cat(sprintf("    WITH evangelical:    beta=%.4f, p=%.4f\n", blp_with[3,1], blp_with[3,4]))
cat(sprintf("    Beta change: %.4f (%.1f%%)\n",
            blp_with[3,1] - blp_no[3,1],
            100 * (blp_with[3,1] - blp_no[3,1]) / blp_no[3,1]))

cat("\nDONE\n")
