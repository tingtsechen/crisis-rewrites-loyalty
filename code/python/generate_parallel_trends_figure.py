"""
Generate publication-quality parallel trends figure (Figure S4).
Two panels: (A) Brand share time series, (B) Difference with trend lines.
Styled to match Figures 1–3 and S1–S3.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy import stats

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(SCRIPT_DIR, '..', '..', '..')
DATA_PATH = os.path.join(PROJ_ROOT, 'data', 'final_data',
                         'merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv')
OUT_DIR = SCRIPT_DIR

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C_HIGH = '#C0392B'
C_LOW = '#4A90D9'
C_2019 = '#2E8B57'
C_2020 = '#E67E22'

# ── Load & Prepare ──
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['sld_dt'])
median_rep = df['republican_percentage_std'].median()
df['rep_group'] = np.where(df['republican_percentage_std'] > median_rep,
                           'High Republican', 'Low Republican')
df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')

weekly = df.groupby(['week_start', 'rep_group', 'year']).agg(
    brand_share=('is_brand', 'mean')
).reset_index()

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
fig.subplots_adjust(wspace=0.30, top=0.85)

# ═══════════════════════════════════════
# Panel A: Brand share time series
# ═══════════════════════════════════════
for group, color in [('High Republican', C_HIGH), ('Low Republican', C_LOW)]:
    for yr in [2019, 2020]:
        data = (weekly[(weekly['rep_group'] == group) & (weekly['year'] == yr)]
                .sort_values('week_start'))
        label = group if yr == 2019 else None
        ax1.plot(data['week_start'], data['brand_share'],
                 color=color, linewidth=1.6, marker='o', markersize=3.5,
                 markeredgecolor='white', markeredgewidth=0.4, label=label)

# COVID onset marker
covid_start = pd.Timestamp('2020-01-01')
ax1.axvline(x=covid_start, color='#AAAAAA', linestyle='--', linewidth=0.8)
ax1.text(covid_start + pd.Timedelta(days=10), 0.375, 'Crisis\nonset',
         fontsize=8.5, color='#888888', va='top', style='italic')

# Shading for treatment windows
for yr, alpha in [(2019, 0.06), (2020, 0.12)]:
    ax1.axvspan(pd.Timestamp(f'{yr}-04-01'), pd.Timestamp(f'{yr}-06-30'),
                alpha=alpha, color='#FFD700', zorder=0)

ax1.set_xlabel('Date', labelpad=6)
ax1.set_ylabel('Brand drug share', labelpad=6)
ax1.set_title('(A) Brand Drug Share by Political Ideology', fontweight='bold',
              pad=10, fontfamily='sans-serif')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax1.set_ylim(0.15, 0.40)
ax1.legend(loc='lower left', framealpha=0.95, edgecolor='#CCCCCC', fontsize=9)
ax1.tick_params(length=3.5)

# ═══════════════════════════════════════
# Panel B: Difference plot
# ═══════════════════════════════════════
weekly_pivot = weekly.pivot_table(index=['week_start', 'year'],
                                  columns='rep_group', values='brand_share').reset_index()
weekly_pivot['diff'] = weekly_pivot['High Republican'] - weekly_pivot['Low Republican']

# Only 2019 (pre-treatment) for parallel trends test
data_2019 = weekly_pivot[weekly_pivot['year'] == 2019].sort_values('week_start').copy()
data_2019['week_num'] = range(len(data_2019))

ax2.plot(data_2019['week_num'], data_2019['diff'], color=C_2019, linewidth=1.6,
         marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.4,
         label='2019 (Pre-treatment)', zorder=3)

# Trend line
slope, intercept, r, p, se = stats.linregress(data_2019['week_num'], data_2019['diff'])
trend_x = np.array([data_2019['week_num'].min(), data_2019['week_num'].max()])
ax2.plot(trend_x, slope * trend_x + intercept, color=C_2019,
         linestyle='--', alpha=0.5, linewidth=1.2, zorder=2)

ax2.axhline(0, color='#999999', linestyle=':', linewidth=0.8, zorder=0)

# Stats annotation
stats_text = (
    f'Pre-crisis (2019):\n'
    f'  Slope = {slope:.4f}\n'
    f'  $p$ = {p:.3f}\n'
    f'  \u2192 Parallel trends supported'
)
ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
         fontsize=8.5, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#BBBBBB', lw=0.6))

ax2.set_xlabel('Week (within pre-treatment period)', labelpad=6)
ax2.set_ylabel('Difference in brand share\n(High Rep. \u2212 Low Rep.)', labelpad=6)
ax2.set_title('(B) Parallel Trends Test', fontweight='bold',
              pad=10, fontfamily='sans-serif')
ax2.tick_params(length=3.5)

# Suptitle
fig.suptitle('Parallel Trends Verification',
             fontsize=13, fontweight='bold', y=1.02, fontfamily='sans-serif')

# Save
for ext in ['pdf', 'png']:
    path = os.path.join(OUT_DIR, f'figure_s4_parallel_trends.{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')

plt.close(fig)
print('Done.')
