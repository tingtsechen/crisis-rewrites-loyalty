"""
Individual Treatment Effects Analysis Script for Code Ocean

This script analyzes individual treatment effects estimated from causal forest,
examining how political affiliation and brand status affect treatment outcomes.

Author: Ting-Tse Chen
Date: 2024
"""

import os
import re
from typing import Any
import pandas as pd
import statsmodels.formula.api as smf
import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_latest_cf_folder(results_dir):
    """Find the latest cf_ate folder based on timestamp (format: cf_ate_YYYYMMDD_HHMMSS)"""
    pattern = re.compile(r'^cf_ate_\d{8}_\d{6}')
    cf_folders = [f for f in os.listdir(results_dir)
                  if pattern.match(f) and os.path.isdir(os.path.join(results_dir, f))]
    if not cf_folders:
        return None
    latest = sorted(cf_folders)[-1]
    return os.path.join(results_dir, latest)


class Config:
    """Configuration class for data paths and analysis parameters."""

    def __init__(self, note: str = ""):
        """Initialize configuration."""
        # Analysis parameters
        self.OUTCOMES = ["log_sales", "log_transactions"]

        # Initialize paths based on environment
        if os.path.exists("/code"):
            self.BASE_DIR = "/code"
            self.DATA_DIR = "/data"
            self.RESULTS_DIR = "/data_output"
        else:
            # Local environment
            self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
            self.RESULTS_DIR = os.path.join(self.BASE_DIR, "data_output")

        self.FINAL_DIR = os.path.join(self.DATA_DIR, "final_data")

        # Find latest CF results folder
        self.CF_RESULTS_DIR = find_latest_cf_folder(self.RESULTS_DIR)
        if self.CF_RESULTS_DIR:
            logger.info(f"Using CF results from: {self.CF_RESULTS_DIR}")
        else:
            logger.warning("No timestamped CF folder found, using legacy structure")
            self.CF_RESULTS_DIR = self.RESULTS_DIR

        # Create timestamped output directory for OLS results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        note_suffix = f"_{note}" if note else ""
        self.OLS_OUTPUT_DIR = os.path.join(self.RESULTS_DIR, f"ols_results_{timestamp}{note_suffix}")
        os.makedirs(self.OLS_OUTPUT_DIR, exist_ok=True)

        logger.info(f"Base directory: {self.BASE_DIR}")
        logger.info(f"Data directory: {self.DATA_DIR}")
        logger.info(f"OLS output directory: {self.OLS_OUTPUT_DIR}")


class RegressionAnalyzer:
    """Class for performing regression analysis on treatment effects."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run_regression(self, outcome: str, model_type: str = "with_fe") -> Any:
        """
        Run regression analysis with different model specifications

        Args:
            outcome: Outcome variable name
            model_type: Type of model ('core_only' for core variables only, 'with_fe' for full model)
        """
        # Prepare COVID-19 cases per capita
        self.data["cases_per_capita"] = (
            self.data["cases"] / self.data["total_county_population"]
        )

        # Standardize continuous variables
        vars_to_standardize = [
            "household_median",
            "unemployment_rate",
            "over_65_percentage",
            "edu_bachelor_above",
            "pharmacy_density",
            "cases_per_capita",
        ]

        for var in vars_to_standardize:
            self.data[f"{var}_std"] = (
                self.data[var] - self.data[var].mean()
            ) / self.data[var].std()

        # Create centered version of republican_percentage
        self.data["republican_percentage_centered"] = (
            self.data["republican_percentage"]
            - self.data["republican_percentage"].mean()
        )
        self.data["republican_percentage_centered_std"] = (
            self.data["republican_percentage_centered"]
            / self.data["republican_percentage"].std()
        )

        # Create interaction term
        self.data["republicanCentered_x_brand"] = (
            self.data["republican_percentage_centered_std"] * self.data["is_brand"]
        )

        # Choose model specification based on model_type
        if model_type == "core_only":
            # Core model: only main variables + fixed effects
            formula = (
                "ITE ~ republican_percentage_centered_std + is_brand + republicanCentered_x_brand + "
                "C(state_abbr) + C(weekday) + C(month)"
            )
        else:  # model_type == "with_fe" (full model with controls)
            formula = (
                "ITE ~ republican_percentage_centered_std + is_brand + republicanCentered_x_brand + "
                "household_median_std + unemployment_rate_std + over_65_percentage_std + "
                "edu_bachelor_above_std + pharmacy_density_std + cases_per_capita_std + "
                "race_white + urban_rural_value_lessBetter + "
                "C(state_abbr) + C(weekday) + C(month)"
            )

        if outcome == "log_transactions":
            # Save a copy of the full data
            full_data = self.data.copy()

            # Calculate transaction volume percentiles
            q75 = self.data["transaction_count"].quantile(0.75)  # 75th percentile
            q25 = self.data["transaction_count"].quantile(0.25)  # 25th percentile
            median = self.data["transaction_count"].median()  # Median

            print("\nTransaction Volume Distribution:")
            print(f"Minimum: {self.data['transaction_count'].min():.1f}")
            print(f"25th percentile: {q25:.1f}")
            print(f"Median: {median:.1f}")
            print(f"75th percentile: {q75:.1f}")
            print(f"Maximum: {self.data['transaction_count'].max():.1f}")
            print(f"Mean: {self.data['transaction_count'].mean():.1f}")

            print("\n===== Full Sample Analysis =====")

            # Run the regression with selected formula
            model = smf.ols(formula, data=full_data).fit(
                cov_type="cluster",
                cov_kwds={"groups": full_data["fips"], "use_correction": True},
            )

            return model

        else:
            # Run the regression with selected formula
            model = smf.ols(formula, data=self.data).fit(
                cov_type="cluster",
                cov_kwds={"groups": self.data["fips"], "use_correction": True},
            )

            return model


def save_regression_results_csv(model: Any, output_path: str) -> None:
    """Save regression results to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create results dataframe
    results_df = pd.DataFrame(
        {
            "coefficient": model.params,
            "std_error": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
            "ci_lower": model.conf_int()[0],
            "ci_upper": model.conf_int()[1],
        }
    )

    # Add significance indicators
    results_df["significance"] = ""
    results_df.loc[results_df["p_value"] < 0.001, "significance"] = "***"
    results_df.loc[
        (results_df["p_value"] >= 0.001) & (results_df["p_value"] < 0.01),
        "significance",
    ] = "**"
    results_df.loc[
        (results_df["p_value"] >= 0.01) & (results_df["p_value"] < 0.05), "significance"
    ] = "*"

    results_df.to_csv(output_path)


def main(note: str = "", trim_percentile: int = None):
    """Main execution function.

    Args:
        note: Note to add to output folder name
        trim_percentile: If set (e.g., 90), keeps only the middle X% of ITE values
                        (removes bottom and top (100-X)/2 percentiles)
    """
    config = Config(note=note)

    # Load data
    data_path = os.path.join(
        config.FINAL_DIR,
        "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv",
    )

    # Check if file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")

    logger.info(f"Loading data from: {data_path}")
    df_orig = pd.read_csv(data_path)

    # Remove missing values
    df_orig = df_orig.dropna(subset=["republican_percentage", "is_brand"])
    logger.info(f"Number of observations: {len(df_orig)}")

    # Process each outcome variable
    for outcome in config.OUTCOMES:
        logger.info(f"\nAnalyzing {outcome}...")

        # Load treatment effects from latest CF folder
        ite_path = os.path.join(config.CF_RESULTS_DIR, outcome, "treatment_effects.csv")

        if not os.path.exists(ite_path):
            logger.warning(f"Treatment effects file not found: {ite_path}")
            continue

        df_ite = pd.read_csv(ite_path)
        df_ite = df_ite.rename(columns={"treatment_effect": "ITE"})

        # Prepare regression data
        df = df_orig.copy().iloc[: len(df_ite)].copy()
        df["ITE"] = df_ite["ITE"].values

        # Apply trimming if specified
        if trim_percentile is not None:
            import numpy as np
            lower_pct = (100 - trim_percentile) / 2
            upper_pct = 100 - lower_pct
            lower_bound = np.percentile(df["ITE"], lower_pct)
            upper_bound = np.percentile(df["ITE"], upper_pct)

            n_before = len(df)
            df = df[(df["ITE"] >= lower_bound) & (df["ITE"] <= upper_bound)]
            n_after = len(df)

            print(f"\n*** TRIMMING APPLIED: Keeping {trim_percentile}% of ITE values ***")
            print(f"  Percentile range: {lower_pct:.1f}th to {upper_pct:.1f}th")
            print(f"  ITE bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
            print(f"  Observations: {n_before:,} -> {n_after:,} (removed {n_before - n_after:,}, {(n_before - n_after)/n_before*100:.1f}%)")

        # Initialize analyzer
        analyzer = RegressionAnalyzer(df)

        # Run both core model and full model
        logger.info("Running OLS heterogeneity analysis...")

        # Model 1: Core variables only
        print(f"\n{'=' * 80}")
        print(f"MODEL 1: CORE VARIABLES ONLY - {outcome.upper()}")
        print("=" * 80)
        model_core = analyzer.run_regression(outcome, "core_only")

        print("Model Statistics:")
        print(f"  R-squared: {model_core.rsquared:.4f}")
        print(f"  Adjusted R-squared: {model_core.rsquared_adj:.4f}")
        print(f"  Number of observations: {int(model_core.nobs):,}")

        # Print key coefficients for core model
        print("\nKey Coefficient Estimates:")
        main_vars = [
            "republican_percentage_centered_std",
            "is_brand",
            "republicanCentered_x_brand",
        ]

        for var in main_vars:
            if var in model_core.params.index:
                coef = model_core.params[var]
                se = model_core.bse[var]
                p_val = model_core.pvalues[var]
                ci_lower = model_core.conf_int().loc[var, 0]
                ci_upper = model_core.conf_int().loc[var, 1]

                sig = (
                    "***"
                    if p_val < 0.001
                    else "**"
                    if p_val < 0.01
                    else "*"
                    if p_val < 0.05
                    else ""
                )

                print(f"\n  {var}:")
                print(f"    Coefficient: {coef:.4f}{sig}")
                print(f"    Std. Error: {se:.4f}")
                print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"    p-value: {p_val:.4f}")

        # Print core model summary
        print(f"\n{'=' * 60}")
        print("CORE MODEL OLS REGRESSION SUMMARY")
        print("=" * 60)
        print(model_core.summary())

        # Model 2: Full model with controls
        print(f"\n{'=' * 80}")
        print(f"MODEL 2: FULL MODEL WITH CONTROLS - {outcome.upper()}")
        print("=" * 80)
        model_full = analyzer.run_regression(outcome, "with_fe")

        print("Model Statistics:")
        print(f"  R-squared: {model_full.rsquared:.4f}")
        print(f"  Adjusted R-squared: {model_full.rsquared_adj:.4f}")
        print(f"  Number of observations: {int(model_full.nobs):,}")

        # Print key coefficients for full model
        print("\nKey Coefficient Estimates:")
        for var in main_vars:
            if var in model_full.params.index:
                coef = model_full.params[var]
                se = model_full.bse[var]
                p_val = model_full.pvalues[var]
                ci_lower = model_full.conf_int().loc[var, 0]
                ci_upper = model_full.conf_int().loc[var, 1]

                sig = (
                    "***"
                    if p_val < 0.001
                    else "**"
                    if p_val < 0.01
                    else "*"
                    if p_val < 0.05
                    else ""
                )

                print(f"\n  {var}:")
                print(f"    Coefficient: {coef:.4f}{sig}")
                print(f"    Std. Error: {se:.4f}")
                print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"    p-value: {p_val:.4f}")

        # Print full model summary
        print(f"\n{'=' * 60}")
        print("FULL MODEL OLS REGRESSION SUMMARY")
        print("=" * 60)
        print(model_full.summary())

        # Create results directory if needed
        outcome_dir = os.path.join(config.OLS_OUTPUT_DIR, outcome)
        os.makedirs(outcome_dir, exist_ok=True)

        # Save core model results
        save_regression_results_csv(
            model_core,
            os.path.join(outcome_dir, "ols_core_model_results.csv"),
        )

        # Save full model results
        save_regression_results_csv(
            model_full,
            os.path.join(outcome_dir, "ols_full_model_results.csv"),
        )

        print(f"\nResults saved to: {outcome_dir}")
        print("=" * 80)

    print(f"\n{'=' * 80}")
    print(f"ALL RESULTS SAVED TO: {config.OLS_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    # 可以在這裡修改 note 參數來加入備註
    # trim_percentile: 設定要保留的百分位數範圍 (例如 90 表示保留 5th-95th percentile)
    main(note="only_dspns_qty_sum", trim_percentile=None)
    logger.info("\nAnalysis completed!")
