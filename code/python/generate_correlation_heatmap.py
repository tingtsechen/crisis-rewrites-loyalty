"""Generate correlation matrix heatmap for the 13-control + 7 heterogeneity feature specification."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(SCRIPT_DIR, '..', '..', '..')

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Load data
DATA_PATH = os.path.join(
    PROJ_ROOT, 'data', 'final_data',
    'merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv'
)
RELIG_PATH = os.path.join(PROJ_ROOT, 'data', 'external', 'county_religiosity_2020.csv')
print('Loading data...')
df = pd.read_csv(DATA_PATH)
relig = pd.read_csv(RELIG_PATH)

# Merge evangelical
df['fips'] = df['fips'].astype(str)
relig['fips'] = relig['fips'].astype(str)
df = df.merge(relig[['fips', 'evangelical_pct']], on='fips', how='left')

# Select our 13 controls + key heterogeneity features
vars_to_plot = [
    'republican_percentage',
    'is_brand',
    'log_sales',
    'log_avg_price',
    'log_transactions',
    'dspns_qty_mean',
    'pharmacy_density',
    'insurance_coverage_mean',
    'patient_coverage_mean',
    'loc_id_nunique',
    'female_percentage',
    'over_65_percentage',
    'edu_bachelors',
    'income_middle_percentage',
    'unemployment_rate',
    'race_white',
    'evangelical_pct',
]

# Nice labels
labels = {
    'republican_percentage': 'Republican Vote Share',
    'is_brand': 'Brand Status',
    'log_sales': 'log(Sales)',
    'log_avg_price': 'log(Avg Price)',
    'log_transactions': 'log(Transactions)',
    'dspns_qty_mean': 'Dispensed Qty (mean)',
    'pharmacy_density': 'Pharmacy Density',
    'insurance_coverage_mean': 'Insurance Coverage',
    'patient_coverage_mean': 'Patient Coverage',
    'loc_id_nunique': 'Pharmacy Count',
    'female_percentage': 'Female %',
    'over_65_percentage': 'Over 65 %',
    'edu_bachelors': 'Education (BA+)',
    'income_middle_percentage': 'Middle Income %',
    'unemployment_rate': 'Unemployment Rate',
    'race_white': 'White %',
    'evangelical_pct': 'Evangelical %',
}

# Filter to available columns
available = [v for v in vars_to_plot if v in df.columns]
df_sub = df[available].dropna()

print(f'Computing correlation matrix ({len(available)} vars, N={len(df_sub):,})...')
corr = df_sub.corr()

# Rename for display
corr = corr.rename(index=labels, columns=labels)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
    vmin=-1, vmax=1, center=0,
    square=True, linewidths=0.5, linecolor='white',
    cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
    annot_kws={'size': 7},
    ax=ax,
)

ax.set_title('Correlation Matrix of Key Variables (13-Control Specification)',
             fontsize=12, fontweight='bold', pad=15)
ax.tick_params(axis='both', labelsize=8)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

for ext in ['pdf', 'png']:
    path = os.path.join(SCRIPT_DIR, f'figure_correlation_matrix.{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
print('Saved: figure_correlation_matrix.pdf/.png')
plt.close()
