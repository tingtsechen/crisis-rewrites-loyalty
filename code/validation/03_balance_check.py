"""
Balance Check: High vs Low Republican Counties
Compares covariate distributions to justify control variable inclusion
"""

import pandas as pd
import numpy as np
from scipy import stats

# === CONFIGURATION ===
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_data", "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_output", "validation_tests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

# === SPLIT BY REPUBLICAN VOTE SHARE ===
# Using median split (can also use terciles or other cutoffs)
median_rep = df['republican_percentage_std'].median()
df['high_republican'] = df['republican_percentage_std'] > median_rep

print(f"Median Republican Vote Share (std): {median_rep:.3f}")
print(f"N (High Republican): {df['high_republican'].sum():,}")
print(f"N (Low Republican): {(~df['high_republican']).sum():,}")

# === VARIABLES TO COMPARE ===
balance_vars = [
    ('household_median', 'Median Household Income ($)'),
    ('edu_bachelor_above', 'Education (Bachelor+) %'),
    ('over_65_percentage', 'Population Over 65 %'),
    ('female_percentage', 'Female Population %'),
    ('race_white', 'White Population %'),
    ('race_black', 'Black Population %'),
    ('unemployment_rate', 'Unemployment Rate %'),
    ('pharmacy_density', 'Pharmacy Density'),
    ('urban_rural_value_lessBetter', 'Urban-Rural Index'),
    ('insurance_coverage_mean', 'Insurance Coverage'),
    ('cases', 'COVID-19 Cases'),
]

# === CALCULATE BALANCE STATISTICS ===
results = []
for var, label in balance_vars:
    if var not in df.columns:
        print(f"Warning: {var} not found in data")
        continue

    high_rep = df[df['high_republican']][var].dropna()
    low_rep = df[~df['high_republican']][var].dropna()

    # T-test for difference
    t_stat, p_value = stats.ttest_ind(high_rep, low_rep)

    # Standardized difference (Cohen's d)
    # This is the preferred measure for balance checks
    pooled_std = np.sqrt((high_rep.std()**2 + low_rep.std()**2) / 2)
    std_diff = (high_rep.mean() - low_rep.mean()) / pooled_std if pooled_std > 0 else 0

    results.append({
        'Variable': label,
        'High_Republican_Mean': round(high_rep.mean(), 2),
        'Low_Republican_Mean': round(low_rep.mean(), 2),
        'Difference': round(high_rep.mean() - low_rep.mean(), 2),
        'Std_Diff': round(std_diff, 3),
        'p_value': round(p_value, 4)
    })

# Create dataframe
balance_df = pd.DataFrame(results)

# === PRINT RESULTS ===
print("\n" + "=" * 90)
print("BALANCE CHECK: High vs Low Republican Counties")
print("=" * 90)
print()
print(f"{'Variable':<30} | {'High Rep':>12} | {'Low Rep':>12} | {'Std.Diff':>10} | {'p-value':>10}")
print("-" * 90)

for _, row in balance_df.iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    imbalance = "⚠" if abs(row['Std_Diff']) > 0.25 else ""
    print(f"{row['Variable']:<30} | {row['High_Republican_Mean']:>12} | {row['Low_Republican_Mean']:>12} | {row['Std_Diff']:>10.3f} {imbalance} | {row['p_value']:>9.4f} {sig}")

# === SUMMARY ===
print("\n" + "=" * 90)
print("INTERPRETATION:")
print("-" * 90)
high_imbalance = (balance_df['Std_Diff'].abs() > 0.25).sum()
print(f"Variables with |Std. Diff.| > 0.25: {high_imbalance}/{len(balance_df)}")
print()
print("What this means:")
print("- Standardized Difference (Cohen's d) measures effect size of group differences")
print("- |Std. Diff.| > 0.25 is typically considered meaningful imbalance")
print("- Imbalance is EXPECTED in observational studies")
print("- This JUSTIFIES inclusion of control variables in regression")
print("- Without controls, treatment effect estimates could be confounded")

# === SAVE RESULTS ===
balance_df.to_csv(os.path.join(OUTPUT_DIR, 'balance_check.csv'), index=False)
print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'balance_check.csv')}")
