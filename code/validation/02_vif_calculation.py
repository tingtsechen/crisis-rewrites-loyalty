"""
Variance Inflation Factor (VIF) Calculation
Calculates VIF for regression model variables
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === CONFIGURATION ===
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_data", "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_output", "validation_tests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

# === CREATE STANDARDIZED VARIABLES ===
# This matches the original OLS analysis which standardized all continuous variables
import numpy as np

# List of variables to standardize (matching original analysis)
vars_to_standardize = [
    'republican_percentage',
    'total_county_population',
    'edu_bachelor_above',
    'female_percentage',
    'over_65_percentage',
    'household_median',
    'race_black',
    'unemployment_rate',
    'cases',
    'loc_id_nunique',
    'pharmacy_density',
    'insurance_coverage_mean',
    'urban_rural_value_lessBetter'
]

# Standardize all continuous variables
for var in vars_to_standardize:
    if var in df.columns:
        df[f'{var}_std'] = (df[var] - df[var].mean()) / df[var].std()

# Rename for consistency
df['republican_percentage_std'] = df['republican_percentage_std'] if 'republican_percentage_std' in df.columns else (
    df['republican_percentage'] - df['republican_percentage'].mean()
) / df['republican_percentage'].std()

# === SELECT REGRESSION VARIABLES ===
# These match the original OLS model specification from ols_result_all folder
# Based on: 0127_OLS_reg_sldDt_log_rx_prc_daily_insurance_fixedEffect_remove_healthRisk.txt
# Note: Original analysis used standardized versions of all continuous variables

regression_vars = [
    'republican_percentage_std',
    'healthrisk_impact',
    'is_brand',
    'total_county_population_std',
    'edu_bachelor_above_std',
    'female_percentage_std',
    'over_65_percentage_std',
    'household_median_std',
    'race_black_std',
    'unemployment_rate_std',
    'cases_std',
    'loc_id_nunique_std',
    'pharmacy_density_std',
    'insurance_coverage_mean_std',
    'urban_rural_value_lessBetter_std'
]

# Filter to existing variables
existing_vars = [v for v in regression_vars if v in df.columns]
print(f"Variables found: {len(existing_vars)}/{len(regression_vars)}")

# === PREPARE DATA ===
vif_data = df[existing_vars].dropna()
print(f"Observations: {len(vif_data):,}")

# === CALCULATE VIF ===
vif_results = []
for i, col in enumerate(existing_vars):
    vif_value = variance_inflation_factor(vif_data.values, i)
    vif_results.append({
        'Feature': col,
        'VIF': vif_value
    })

# Create dataframe (keep original order like the reference file)
vif_df = pd.DataFrame(vif_results)

# === PRINT RESULTS ===
print("\n" + "=" * 60)
print("完整模型的方差膨脹因子 (VIF):")
print("=" * 60)
print(vif_df.to_string(index=True))

print("\n" + "=" * 60)
print("VARIANCE INFLATION FACTORS (sorted)")
print("=" * 60)

vif_df_sorted = vif_df.sort_values('VIF', ascending=False)
for _, row in vif_df_sorted.iterrows():
    status = "✓" if row['VIF'] < 5 else "⚠" if row['VIF'] < 10 else "✗"
    print(f"{row['Feature']:35} | VIF: {row['VIF']:6.2f} {status}")

print("\n" + "=" * 60)
max_vif = vif_df['VIF'].max()
print(f"Maximum VIF: {max_vif:.2f}")

if max_vif < 5:
    print("✓ All VIF < 5: No multicollinearity concern")
elif max_vif < 10:
    print("⚠ Some VIF between 5-10: Moderate multicollinearity, but acceptable")
else:
    print("✗ VIF > 10: Consider removing highly correlated variables")

# === SAVE RESULTS ===
vif_df.to_csv(os.path.join(OUTPUT_DIR, 'vif_results.csv'), index=False)
print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'vif_results.csv')}")
