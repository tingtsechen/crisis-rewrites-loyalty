"""
Heterogeneity Analysis by Income Level
Tests whether the Republican × Brand effect varies with income
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

# Standardize income
df['income_std'] = (df['household_median'] - df['household_median'].mean()) / df['household_median'].std()

# Split by income (median)
median_income = df['household_median'].median()
df['high_income'] = df['household_median'] > median_income

print("=" * 80)
print("HETEROGENEITY ANALYSIS BY INCOME LEVEL")
print("=" * 80)
print(f"Median Household Income: ${median_income:,.0f}")
print(f"High Income Counties: N = {df['high_income'].sum():,}")
print(f"Low Income Counties: N = {(~df['high_income']).sum():,}")

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

# High income
high_df = df[df['high_income']].dropna(subset=['log_sales', 'republican_percentage_std', 'is_brand'])
model_high = smf.ols(formula, data=high_df).fit(cov_type='cluster', cov_kwds={'groups': high_df['state_abbr']})

print("\nHigh Income Counties:")
print(f"  Republican × Brand: β = {model_high.params['rep_x_brand']:.4f}, p = {model_high.pvalues['rep_x_brand']:.4f}")
print(f"  N = {int(model_high.nobs):,}, R² = {model_high.rsquared:.4f}")

# Low income
low_df = df[~df['high_income']].dropna(subset=['log_sales', 'republican_percentage_std', 'is_brand'])
model_low = smf.ols(formula, data=low_df).fit(cov_type='cluster', cov_kwds={'groups': low_df['state_abbr']})

print("\nLow Income Counties:")
print(f"  Republican × Brand: β = {model_low.params['rep_x_brand']:.4f}, p = {model_low.pvalues['rep_x_brand']:.4f}")
print(f"  N = {int(model_low.nobs):,}, R² = {model_low.rsquared:.4f}")

# Z-test for difference
high_coef = model_high.params['rep_x_brand']
low_coef = model_low.params['rep_x_brand']
high_se = model_high.bse['rep_x_brand']
low_se = model_low.bse['rep_x_brand']

z_diff = (high_coef - low_coef) / np.sqrt(high_se**2 + low_se**2)
p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

print(f"\nDifference Test: Z = {z_diff:.3f}, p = {p_diff:.4f}")

# === APPROACH 2: THREE-WAY INTERACTION ===
print("\n" + "=" * 80)
print("APPROACH 2: THREE-WAY INTERACTION MODEL")
print("=" * 80)

# Create interaction terms
df['rep_x_brand_x_income'] = df['republican_percentage_std'] * df['is_brand'] * df['income_std']
df['rep_x_income'] = df['republican_percentage_std'] * df['income_std']
df['brand_x_income'] = df['is_brand'] * df['income_std']

formula_3way = """log_sales ~ republican_percentage_std + is_brand + income_std +
                 rep_x_brand + rep_x_income + brand_x_income + rep_x_brand_x_income +
                 healthrisk_impact + edu_bachelor_above + female_percentage +
                 over_65_percentage + race_black + unemployment_rate + cases +
                 pharmacy_density + insurance_coverage_mean + urban_rural_value_lessBetter +
                 C(state_abbr) + C(month) + C(weekday)"""

model_3way = smf.ols(formula_3way, data=df.dropna()).fit(cov_type='cluster', cov_kwds={'groups': df.dropna()['state_abbr']})

print("\nThree-Way Interaction (Republican × Brand × Income):")
print(f"  β = {model_3way.params['rep_x_brand_x_income']:.4f}")
print(f"  SE = {model_3way.bse['rep_x_brand_x_income']:.4f}")
print(f"  p = {model_3way.pvalues['rep_x_brand_x_income']:.4f}")

# === INTERPRETATION ===
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if p_diff > 0.05 and model_3way.pvalues['rep_x_brand_x_income'] > 0.05:
    print("Both tests suggest income does NOT moderate the Republican × Brand effect.")
    print("→ The effect is NOT purely driven by price sensitivity.")
    print("→ Supports political identity explanation.")
else:
    print("Evidence suggests income moderates the effect.")
    print("→ Price sensitivity may play a role.")
    print("→ Further investigation needed.")

# === SAVE RESULTS ===
results = pd.DataFrame({
    'Analysis': ['Subsample (High)', 'Subsample (Low)', 'Difference Test', 'Three-Way Interaction'],
    'Coefficient': [high_coef, low_coef, '', model_3way.params['rep_x_brand_x_income']],
    'SE': [high_se, low_se, '', model_3way.bse['rep_x_brand_x_income']],
    'p-value': [model_high.pvalues['rep_x_brand'], model_low.pvalues['rep_x_brand'], p_diff, model_3way.pvalues['rep_x_brand_x_income']],
    'N': [int(model_high.nobs), int(model_low.nobs), '', int(model_3way.nobs)]
})
results.to_csv(os.path.join(OUTPUT_DIR, 'heterogeneity_income.csv'), index=False)
print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'heterogeneity_income.csv')}")
