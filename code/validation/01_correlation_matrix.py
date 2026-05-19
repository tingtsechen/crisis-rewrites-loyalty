"""
Correlation Matrix Analysis
Generates correlation matrix table and heatmap visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_data", "merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_output", "validation_tests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

# === SELECT VARIABLES ===
variables = {
    'republican_percentage_std': 'Republican Vote Share',
    'healthrisk_impact': 'Pandemic Period',
    'is_brand': 'Brand Status',
    'rx_prc_amt_sum': 'Sales (raw)',
    'transaction_count': 'Transactions (raw)',
    'edu_bachelor_above': 'Education',
    'household_median': 'Median Income',
    'race_black': 'Black Population %',
    'unemployment_rate': 'Unemployment Rate',
    'insurance_coverage_mean': 'Insurance Coverage',
    'pharmacy_density': 'Pharmacy Density',
    'urban_rural_value_lessBetter': 'Urban-Rural',
    'cases': 'COVID Cases'
}

# Filter existing variables
existing_vars = {k: v for k, v in variables.items() if k in df.columns}
var_list = list(existing_vars.keys())
var_labels = list(existing_vars.values())

# === CREATE LOG-TRANSFORMED DVs ===
if 'rx_prc_amt_sum' in df.columns:
    df['log_sales'] = np.log1p(df['rx_prc_amt_sum'])
if 'transaction_count' in df.columns:
    df['log_transactions'] = np.log1p(df['transaction_count'])

# Update variable list with log-transformed DVs
final_vars = []
final_labels = []
for k, v in existing_vars.items():
    if k == 'rx_prc_amt_sum':
        final_vars.append('log_sales')
        final_labels.append('Log(Sales)')
    elif k == 'transaction_count':
        final_vars.append('log_transactions')
        final_labels.append('Log(Transactions)')
    else:
        final_vars.append(k)
        final_labels.append(v)

# === CALCULATE CORRELATION MATRIX ===
corr_data = df[final_vars].dropna()
corr_matrix = corr_data.corr()

# === PRINT CORRELATION TABLE ===
print("=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)
print(f"N = {len(corr_data):,}")
print()
print(corr_matrix.round(2).to_string())

# === CREATE HEATMAP ===
fig, ax = plt.subplots(figsize=(12, 10))

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Create heatmap
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
    xticklabels=final_labels,
    yticklabels=final_labels,
    ax=ax
)

plt.title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_S1_correlation_matrix.png'), dpi=300, bbox_inches='tight')
print(f"\nFigure saved to {os.path.join(OUTPUT_DIR, 'Figure_S1_correlation_matrix.png')}")

# === SAVE CORRELATION TABLE ===
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix.csv'))
print(f"Table saved to {os.path.join(OUTPUT_DIR, 'correlation_matrix.csv')}")
