"""
Heterogeneity Analysis by Urban-Rural Classification
Tests whether the Republican × Brand effect varies with urbanicity
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# === CONFIGURATION ===
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_data", "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_output", "validation_tests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

# === CREATE VARIABLES ===
df['log_sales'] = np.log1p(df['rx_prc_amt_sum'])
df['rep_x_brand'] = df['republican_percentage_std'] * df['is_brand']

# Standardize urban-rural index
df['urban_rural_std'] = (df['urban_rural_value_lessBetter'] - df['urban_rural_value_lessBetter'].mean()) / df['urban_rural_value_lessBetter'].std()

# Split by urban/rural (median)
median_ur = df['urban_rural_value_lessBetter'].median()
df['is_rural'] = df['urban_rural_value_lessBetter'] > median_ur

print("=" * 80)
print("HETEROGENEITY ANALYSIS BY URBAN-RURAL CLASSIFICATION")
print("=" * 80)
print(f"Urban-Rural Index Range: {df['urban_rural_value_lessBetter'].min():.0f} - {df['urban_rural_value_lessBetter'].max():.0f}")
print(f"Median: {median_ur}")
print(f"Urban Counties (≤ median): N = {(~df['is_rural']).sum():,}")
print(f"Rural Counties (> median): N = {df['is_rural'].sum():,}")

# === MODEL FORMULA ===
formula = """log_sales ~ republican_percentage_std + is_brand + rep_x_brand +
             healthrisk_impact + edu_bachelor_above + female_percentage +
             over_65_percentage + household_median + race_black +
             unemployment_rate + cases + pharmacy_density +
             insurance_coverage_mean + urban_rural_value_lessBetter +
             C(state_abbr) + C(month) + C(weekday)"""

# === APPROACH 1: SUBSAMPLE ANALYSIS ===
print("\n" + "=" * 80)
print("APPROACH 1: SUBSAMPLE ANALYSIS")
print("=" * 80)

# Urban
urban_df = df[~df['is_rural']].dropna(subset=['log_sales', 'republican_percentage_std', 'is_brand'])
model_urban = smf.ols(formula, data=urban_df).fit(cov_type='cluster', cov_kwds={'groups': urban_df['state_abbr']})

print("\nUrban Counties:")
print(f"  Republican × Brand: β = {model_urban.params['rep_x_brand']:.4f}, p = {model_urban.pvalues['rep_x_brand']:.4f}")
print(f"  N = {int(model_urban.nobs):,}, R² = {model_urban.rsquared:.4f}")

# Rural
rural_df = df[df['is_rural']].dropna(subset=['log_sales', 'republican_percentage_std', 'is_brand'])
model_rural = smf.ols(formula, data=rural_df).fit(cov_type='cluster', cov_kwds={'groups': rural_df['state_abbr']})

print("\nRural Counties:")
print(f"  Republican × Brand: β = {model_rural.params['rep_x_brand']:.4f}, p = {model_rural.pvalues['rep_x_brand']:.4f}")
print(f"  N = {int(model_rural.nobs):,}, R² = {model_rural.rsquared:.4f}")

# Z-test for difference
urban_coef = model_urban.params['rep_x_brand']
rural_coef = model_rural.params['rep_x_brand']
urban_se = model_urban.bse['rep_x_brand']
rural_se = model_rural.bse['rep_x_brand']

z_diff = (urban_coef - rural_coef) / np.sqrt(urban_se**2 + rural_se**2)
p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

print(f"\nDifference Test: Z = {z_diff:.3f}, p = {p_diff:.4f}")

# === APPROACH 2: THREE-WAY INTERACTION ===
print("\n" + "=" * 80)
print("APPROACH 2: THREE-WAY INTERACTION MODEL")
print("=" * 80)

# Create interaction terms
df['rep_x_brand_x_rural'] = df['republican_percentage_std'] * df['is_brand'] * df['urban_rural_std']
df['rep_x_rural'] = df['republican_percentage_std'] * df['urban_rural_std']
df['brand_x_rural'] = df['is_brand'] * df['urban_rural_std']

formula_3way = """log_sales ~ republican_percentage_std + is_brand + urban_rural_std +
                 rep_x_brand + rep_x_rural + brand_x_rural + rep_x_brand_x_rural +
                 healthrisk_impact + edu_bachelor_above + female_percentage +
                 over_65_percentage + household_median + race_black + unemployment_rate +
                 cases + pharmacy_density + insurance_coverage_mean +
                 C(state_abbr) + C(month) + C(weekday)"""

model_3way = smf.ols(formula_3way, data=df.dropna()).fit(cov_type='cluster', cov_kwds={'groups': df.dropna()['state_abbr']})

print("\nThree-Way Interaction (Republican × Brand × Urban-Rural):")
print(f"  β = {model_3way.params['rep_x_brand_x_rural']:.4f}")
print(f"  SE = {model_3way.bse['rep_x_brand_x_rural']:.4f}")
print(f"  p = {model_3way.pvalues['rep_x_brand_x_rural']:.4f}")

# === INTERPRETATION ===
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if p_diff > 0.05 and model_3way.pvalues['rep_x_brand_x_rural'] > 0.05:
    print("Both tests suggest urbanicity does NOT moderate the Republican × Brand effect.")
    print("→ The effect is NOT driven by access/availability differences.")
    print("→ Supports demand-side (identity-based) explanation.")
else:
    print("Evidence suggests urbanicity moderates the effect.")
    print("→ Access/availability may play a role.")
    print("→ Further investigation needed.")

# === SAVE RESULTS ===
results = pd.DataFrame({
    'Analysis': ['Subsample (Urban)', 'Subsample (Rural)', 'Difference Test', 'Three-Way Interaction'],
    'Coefficient': [urban_coef, rural_coef, '', model_3way.params['rep_x_brand_x_rural']],
    'SE': [urban_se, rural_se, '', model_3way.bse['rep_x_brand_x_rural']],
    'p-value': [model_urban.pvalues['rep_x_brand'], model_rural.pvalues['rep_x_brand'], p_diff, model_3way.pvalues['rep_x_brand_x_rural']],
    'N': [int(model_urban.nobs), int(model_rural.nobs), '', int(model_3way.nobs)]
})
results.to_csv(os.path.join(OUTPUT_DIR, 'heterogeneity_urban_rural.csv'), index=False)
print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'heterogeneity_urban_rural.csv')}")
