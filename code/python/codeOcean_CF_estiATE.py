"""
Individual Treatment Effects Analysis Script for Code Ocean

This script analyzes individual treatment effects estimated from causal forest,
examining how political affiliation and brand status affect treatment outcomes.

Author: Ting-Tse Chen
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from econml.dml import CausalForestDML
from typing import List, Tuple, Any
from scipy import stats
from datetime import datetime

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
FINAL_DATA_DIR = os.path.join(DATA_DIR, "final_data")

# Create timestamped output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
NOTE = "only_dspns_qty_sum"  # ✓ 推薦版本：兩個 outcome 都顯著
OUTPUT_DIR = os.path.join(BASE_DIR, "data_output", f"cf_ate_{TIMESTAMP}_{NOTE}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def select_features():
    """
    Select features for heterogeneity analysis - combining features from v3 and v5
    """
    # Heterogeneity features
    hetero_features = [
        # Original features
        "republican_percentage_std",
        "is_brand",
        "republicanStd_x_brand",
        # ====== 新加入的變量 (New Variables) ======
        # 藥品類別總量 (Drug Category Sums)
        "drug_ANDROGENS_ANABOLIC_sum",  # 雄性激素類藥物總量
        "drug_CONTRACEPTIVES_sum",  # 避孕藥總量
        "drug_ESTROGENS_sum",  # 雌激素類藥物總量
        "drug_PROGESTINS_sum",  # 孕激素類藥物總量
    ]

    # Control features
    control_features = [
        # Prescription Related
        "dspns_qty_sum",  # ✓ 推薦：兩個 outcome 都顯著
        # "dspns_qty_mean",  # 備選：無 data leakage 但 log_transactions 不顯著
        # "avg_prc",
        # =======================================
        # Demographic features
        "total_county_population",
        "female_percentage",
        "male_percentage",
        "over_65_percentage",
        # Education-related
        "edu_less_than_hs",
        "edu_hs_graduate",
        "edu_some_college",
        "edu_bachelors",
        "edu_graduate",
        # Income-related
        # ====== 新加入的變量 (New Variables) ======
        # "household_median",  # 家庭收入中位數
        # =======================================
        "income_low_percentage",
        "income_lower_middle_percentage",
        "income_middle_percentage",
        "income_upper_middle_percentage",
        "income_high_percentage",
        # Race distribution
        "race_white",
        "race_black",
        "race_native",
        "race_asian",
        "race_pacific",
        "race_other",
        "race_two_or_more",
        # Employment-related
        "employment_rate",
        "unemployment_rate",
        # Healthcare-related
        "pharmacy_density",
        "insurance_coverage_mean",
        "patient_coverage_mean",
        "loc_id_nunique",
        # Healthcare indicators
        "physicians_family_medicine_f1200120",
        "physicians_medical_specialists_f0461820",
        "physicians_cardiovascular_f0463520",
        "physicians_psychiatry_f1013420",
        "hospitals_rural_f1403320",
        "hospitals_accreditation_f0887720",
        "hospitals_veterans_f0892920",
        "medicare_enrollment_f1554920",
        "medicare_advantage_f1319220",
        "medicare_prescription_drug_f1420720",
        # Geography and urbanization
        "urban_rural_value_lessBetter",
        "metro_indicator",
        # COVID-19 related
        "cases",
        "deaths",
        # Time-related
        "week",
        "month",
        "weekday",
    ]

    return hetero_features, control_features


def create_additional_features(df):
    """
    Create additional features for CF analysis
    """
    df = df.copy()

    # Standardize republican_percentage
    df["republican_percentage_std"] = (
        df["republican_percentage"] - df["republican_percentage"].mean()
    ) / df["republican_percentage"].std()

    # Create interaction terms
    df["republicanStd_x_brand"] = df["republican_percentage_std"] * df["is_brand"]

    # Categorize counties by political leaning
    def categorize_political_leaning(row):
        if row["republican_percentage"] <= 40:
            return "Liberal"
        elif row["republican_percentage"] < 60:
            return "Moderate"
        else:
            return "Conservative"

    df["political_leaning"] = df.apply(categorize_political_leaning, axis=1)

    return df


def calculate_ate_pvalue(te_pred, n_bootstrap=1000, alpha=0.05):
    """
    Calculate the p-value for the Average Treatment Effect (ATE)
    """
    # Calculate ATE
    ate = np.mean(te_pred)

    # Bootstrap to estimate standard error and confidence interval
    bootstrap_ates = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(te_pred, size=len(te_pred), replace=True)
        bootstrap_ates.append(np.mean(bootstrap_sample))

    # Calculate standard error
    se = np.std(bootstrap_ates)

    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_ates, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_ates, (1 - alpha / 2) * 100)

    # Calculate p-value (two-sided test)
    z_stat = ate / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Determine significance
    significance = (
        "***"
        if p_value < 0.001
        else "**"
        if p_value < 0.01
        else "*"
        if p_value < 0.05
        else "ns"
    )

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significance": significance,
        "z_stat": z_stat,
    }


def run_causal_forest(
    df,
    hetero_features,
    control_features,
    Y_target="log_sales",
    treatment_var="healthrisk_impact",
):
    """
    Run causal forest analysis
    """
    # Handle missing values
    df = df.copy()
    required_cols = [treatment_var, Y_target] + hetero_features + control_features
    df_clean = df.dropna(subset=required_cols)

    X = df_clean[hetero_features + control_features].values
    W = df_clean[treatment_var].values
    Y = df_clean[Y_target].values

    # Initialize and fit the causal forest model
    cf = CausalForestDML(
        model_t=LinearRegression(),
        model_y=LinearRegression(),
        n_estimators=3000,
        min_samples_leaf=15,
        random_state=42,
    )

    cf.fit(Y, T=W, X=X)

    # Get treatment effects and feature importance
    te_pred = cf.effect(X)
    feature_importance = cf.feature_importances_

    # Calculate ATE and p-value
    ate_results = calculate_ate_pvalue(te_pred)

    return cf, te_pred, None, feature_importance, ate_results


def save_results(cf, te_pred, propensity_scores, ate_results, OUTPUT_DIR, feature_names=None, feature_importance=None):
    """
    Save results from causal forest analysis
    """
    # Save treatment effects
    pd.DataFrame(te_pred, columns=["treatment_effect"]).to_csv(
        os.path.join(OUTPUT_DIR, "treatment_effects.csv"), index=False
    )

    # Save feature importance
    if feature_importance is not None and feature_names is not None:
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    # Save ATE results
    if ate_results is not None:
        pd.DataFrame([ate_results]).to_csv(
            os.path.join(OUTPUT_DIR, "ate_results.csv"), index=False
        )


def print_results_summary(te_pred, Y_target, ate_results=None):
    """
    Print summary of treatment effects and feature importance
    """
    # Calculate statistics
    mean_effect = np.mean(te_pred)
    std_effect = np.std(te_pred)
    min_effect = np.min(te_pred)
    max_effect = np.max(te_pred)
    percentiles = np.percentile(te_pred, [10, 25, 50, 75, 90])

    print("\n" + "=" * 50)
    print(f"TREATMENT EFFECT STATISTICS FOR {Y_target.upper()}")
    print("=" * 50)
    print(f"N = {len(te_pred):,}")
    print(f"Mean = {mean_effect:.4f}")
    print(f"SD = {std_effect:.4f}")
    print(f"Range = [{min_effect:.4f}, {max_effect:.4f}]")
    print(f"10th percentile = {percentiles[0]:.4f}")
    print(f"25th percentile = {percentiles[1]:.4f}")
    print(f"Median = {percentiles[2]:.4f}")
    print(f"75th percentile = {percentiles[3]:.4f}")
    print(f"90th percentile = {percentiles[4]:.4f}")

    if ate_results is not None:
        print("\nAverage Treatment Effect (ATE):")
        print(f"  Estimate: {ate_results['ate']:.4f}")
        print(f"  Standard Error: {ate_results['se']:.4f}")
        print(
            f"  95% CI: [{ate_results['ci_lower']:.4f}, {ate_results['ci_upper']:.4f}]"
        )
        print(f"  p-value: {ate_results['p_value']:.4f}")
        print(f"  Significance: {ate_results['significance']}")

    # Feature importance reporting removed
    # In our two-stage framework:
    # - Causal Forest is used for PREDICTION of ITEs
    # - OLS analysis is used for CAUSAL INTERPRETATION
    # Feature importance from prediction stage should not be interpreted causally

    print("=" * 50 + "\n")


def load_and_prepare_data():
    """
    Load and prepare data for analysis
    """
    try:
        # Load main data
        df = pd.read_csv(
            os.path.join(
                FINAL_DATA_DIR,
                "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv",
            )
        )
        print(f"\nMain data loaded successfully: {len(df)} rows")

        # Load healthcare indicators
        healthcare_data = pd.read_csv(
            os.path.join(DATA_DIR, "processed", "county_healthcare_indicators_2020.csv")
        )
        print(
            f"\nHealthcare indicators loaded successfully: {len(healthcare_data)} rows"
        )

        # Merge healthcare data with main data
        df = pd.merge(df, healthcare_data, on="fips", how="left")
        print(f"\nData merged successfully: {len(df)} rows")

        # Create log-transformed outcome variables if they don't exist
        if "log_sales" not in df.columns:
            df["log_sales"] = np.log1p(df["rx_prc_amt_sum"])
        if "log_transactions" not in df.columns:
            df["log_transactions"] = np.log1p(df["transaction_count"])

        # Print summary of available variables
        print(f"\nTotal number of variables: {len(df.columns)}")
        print(f"Date range: {df['sld_dt'].min()} to {df['sld_dt'].max()}")
        print(f"Number of counties: {df['fips'].nunique()}")
        print(f"Number of states: {df['state_abbr'].nunique()}")

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """
    Main function to run the causal forest analysis
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        if df is None:
            return

        # Create additional features
        df = create_additional_features(df)

        # Select features
        hetero_features, control_features = select_features()

        # Run causal forest analysis for both outcome variables
        for Y_target in ["log_sales", "log_transactions"]:
            print(f"\nRunning analysis for {Y_target}")
            OUTPUT_DIR_Y = os.path.join(OUTPUT_DIR, Y_target)
            os.makedirs(OUTPUT_DIR_Y, exist_ok=True)

            cf, te_pred, propensity_scores, feature_importance, ate_results = (
                run_causal_forest(
                    df, hetero_features, control_features, Y_target=Y_target
                )
            )

            # Save results
            all_features = hetero_features + control_features
            save_results(
                cf,
                te_pred,
                propensity_scores,
                ate_results,
                OUTPUT_DIR_Y,
                feature_names=all_features,
                feature_importance=feature_importance,
            )

            # Print results summary
            print_results_summary(te_pred, Y_target, ate_results)

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
