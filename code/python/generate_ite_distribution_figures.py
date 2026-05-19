"""
Generate all publication-quality figures for the paper.

Main text:
  Figure 1 — ITE distributions by brand status (brand vs. generic)
  Figure 2 — Coefficient plot: between-category placebo (G03 vs C10/C01/C02)
  Figure 3 — Coefficient plot: within-G03 subcategory gradient

Supplement:
  Figure S1 — Overall ITE distributions (3 panels)
  Figure S2 — ITE distributions by Republican vote share tercile
  Figure S3 — Specification curve (70 specs, β and p for log_sales)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy import stats

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(SCRIPT_DIR, '..', '..', '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'output', 'grf_blp_main_G03')
ROBUST_DIR = os.path.join(SCRIPT_DIR, '..', 'output', 'grf_blp_robustness')
SUBGRP_DIR = os.path.join(SCRIPT_DIR, '..', 'output', 'grf_blp_subgroup_G03')
SPEC_FILE = os.path.join(PROJ_ROOT, 'data_output',
                         'grf_blp_dual_dv_20260311_004952', 'dual_dv_results.csv')
OUT_DIR = SCRIPT_DIR

# ── Global Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
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

DVS = [
    ('log_sales',        '(A) log(Sales)',          r'Individual treatment effect, $\hat{\tau}_i$'),
    ('log_avg_price',    '(B) log(Average Price)',  r'Individual treatment effect, $\hat{\tau}_i$'),
    ('log_transactions', '(C) log(Transactions)',   r'Individual treatment effect, $\hat{\tau}_i$'),
]

C_GENERIC = '#4A90D9'
C_BRAND   = '#D94A4A'
C_SIG     = '#C0392B'
C_NONSIG  = '#7F8C8D'
TERCILE_COLORS = ['#4A90D9', '#E6A817', '#C0392B']


def load_ite(dv_name: str):
    ite_df = pd.read_csv(os.path.join(DATA_DIR, dv_name, 'treatment_effects.csv'))
    ate_df = pd.read_csv(os.path.join(DATA_DIR, dv_name, 'ate_results.csv'))
    return ite_df, ate_df.iloc[0]


def trim(arr, pct=1.0):
    lo, hi = np.percentile(arr, pct), np.percentile(arr, 100 - pct)
    return arr[(arr >= lo) & (arr <= hi)], lo, hi


def save_fig(fig, name):
    for ext in ['pdf', 'png']:
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {name}.pdf/.png')


# ═══════════════════════════════════════════════════════
# FIGURE 1 (main text): ITE by Brand Status
# ═══════════════════════════════════════════════════════
print('Figure 1 (ITE by Brand)...')
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))
fig1.subplots_adjust(wspace=0.28, top=0.84)

for idx, (dv_name, panel_title, xlabel) in enumerate(DVS):
    ax = axes1[idx]
    ite_df, _ = load_ite(dv_name)
    ites_brand = ite_df.loc[ite_df['is_brand'] == 1, 'ite'].values
    ites_generic = ite_df.loc[ite_df['is_brand'] == 0, 'ite'].values

    _, lo, hi = trim(ite_df['ite'].values, pct=5.0)
    x_grid = np.linspace(lo, hi, 500)

    brand_t = ites_brand[(ites_brand >= lo) & (ites_brand <= hi)]
    generic_t = ites_generic[(ites_generic >= lo) & (ites_generic <= hi)]

    kde_g = stats.gaussian_kde(generic_t, bw_method='scott')
    kde_b = stats.gaussian_kde(brand_t, bw_method='scott')

    ax.fill_between(x_grid, kde_g(x_grid), alpha=0.18, color=C_GENERIC, zorder=1)
    ax.plot(x_grid, kde_g(x_grid), color=C_GENERIC, linewidth=1.5, zorder=3)
    ax.fill_between(x_grid, kde_b(x_grid), alpha=0.18, color=C_BRAND, zorder=2)
    ax.plot(x_grid, kde_b(x_grid), color=C_BRAND, linewidth=1.5, zorder=4)

    # Group mean lines
    mean_g = np.mean(ites_generic)
    mean_b = np.mean(ites_brand)
    ax.axvline(mean_g, color=C_GENERIC, linestyle='--', linewidth=0.9, alpha=0.7, zorder=5)
    ax.axvline(mean_b, color=C_BRAND, linestyle='--', linewidth=0.9, alpha=0.7, zorder=5)
    ax.axvline(0, color='#999999', linestyle=':', linewidth=0.8, zorder=0)

    # Stats box (left side, below title)
    diff = mean_b - mean_g
    lines = [
        f'Brand mean = {mean_b:.4f}',
        f'Generic mean = {mean_g:.4f}',
        f'\u0394 = {diff:+.4f}',
    ]
    ax.text(0.04, 0.96, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#BBBBBB', lw=0.6))

    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel('Density' if idx == 0 else '', labelpad=6)
    ax.set_title(panel_title, fontweight='bold', pad=10, fontfamily='sans-serif')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    ax.tick_params(length=3.5)
# Single shared legend (top center)
n_g = int((ite_df['is_brand'] == 0).sum())
n_b = int((ite_df['is_brand'] == 1).sum())
legend_el = [
    Line2D([0], [0], color=C_GENERIC, linewidth=1.5,
           label=f'Generic ($n$ = {n_g:,})'),
    Line2D([0], [0], color=C_BRAND, linewidth=1.5,
           label=f'Brand ($n$ = {n_b:,})'),
    Line2D([0], [0], color='grey', linestyle='--', linewidth=0.9, label='Group mean'),
]
fig1.legend(handles=legend_el, loc='upper center', ncol=3,
            bbox_to_anchor=(0.5, 0.97), frameon=True, edgecolor='#CCCCCC',
            fontsize=9, handlelength=2, columnspacing=3)
fig1.suptitle('Figure 2. Individual Treatment Effect Distributions by Brand Status',
              fontsize=13, fontweight='bold', y=1.05, fontfamily='sans-serif')
save_fig(fig1, 'figure_2_ite_by_brand')
plt.close(fig1)


# ═══════════════════════════════════════════════════════
# FIGURE 2 (main text): Between-Category Coefficient Plot
# ═══════════════════════════════════════════════════════
print('Figure 2 (Between-category placebo)...')

# Extract interaction coefficients
main_blp = pd.read_csv(os.path.join(DATA_DIR, 'blp_summary.csv'))
robust_blp = pd.read_csv(os.path.join(ROBUST_DIR, 'blp_all_categories.csv'))

# Build data for the plot
cat_data = []

# G03 main
row = main_blp[(main_blp['dv'] == 'log_sales') &
               (main_blp['variable'] == 'republicanStd_x_brand')].iloc[0]
cat_data.append(('G03\n(Sex hormones)\n$n$ = 22,737',
                 row['estimate'], row['std_error'], row['p_value'], True))

for cat, label, n in [('C10', 'C10\n(Lipid-modifying)\n$n$ = 21,841', 21841),
                       ('C01', 'C01\n(Cardiac therapy)\n$n$ = 12,377', 12377),
                       ('C02', 'C02\n(Antihypertensives)\n$n$ = 20,113', 20113)]:
    r = robust_blp[(robust_blp['category'] == cat) &
                   (robust_blp['dv'] == 'log_sales') &
                   (robust_blp['variable'] == 'republicanStd_x_brand')].iloc[0]
    cat_data.append((label, r['estimate'], r['std_error'], r['p_value'], False))

fig2, ax2 = plt.subplots(figsize=(7, 5))

y_positions = list(range(len(cat_data) - 1, -1, -1))
for i, (label, beta, se, pval, is_sig) in enumerate(cat_data):
    y = y_positions[i]
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    color = C_SIG if is_sig else C_NONSIG
    marker = 'D' if is_sig else 'o'

    ax2.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=2.0, zorder=2, solid_capstyle='round')
    ax2.plot(beta, y, marker=marker, color=color, markersize=8,
             markeredgecolor='white', markeredgewidth=0.8, zorder=3)

    # p-value annotation
    p_str = f'$p$ = {pval:.3f}' if pval >= 0.001 else f'$p$ < 0.001'
    ax2.text(ci_hi + 0.005, y, p_str, va='center', fontsize=9, color=color)

ax2.axvline(0, color='#999999', linestyle='--', linewidth=0.8, zorder=1)
ax2.set_yticks(y_positions)
ax2.set_yticklabels([d[0] for d in cat_data], fontsize=10)
ax2.set_xlabel(r'BLP interaction coefficient ($\beta$, republicanStd $\times$ brand)', labelpad=8)
ax2.set_title('Between-Category Placebo Test\n(BLP Interaction on log Sales)',
              fontweight='bold', pad=12, fontfamily='sans-serif', fontsize=12)
ax2.tick_params(axis='y', length=0)
ax2.set_xlim(-0.15, 0.15)

# Legend
legend_el2 = [
    Line2D([0], [0], marker='D', color=C_SIG, linestyle='-', linewidth=2,
           markersize=7, markeredgecolor='white', label='Significant ($p$ < 0.05)'),
    Line2D([0], [0], marker='o', color=C_NONSIG, linestyle='-', linewidth=2,
           markersize=7, markeredgecolor='white', label='Not significant'),
]
ax2.legend(handles=legend_el2, loc='lower right', framealpha=0.95,
           edgecolor='#CCCCCC', fontsize=9)

fig2.tight_layout()
save_fig(fig2, 'figure_s5_between_category_placebo')
plt.close(fig2)


# ═══════════════════════════════════════════════════════
# FIGURE 3 (main text): Within-G03 Subcategory Gradient
# ═══════════════════════════════════════════════════════
print('Figure 3 (Within-G03 gradient)...')

sub_blp = pd.read_csv(os.path.join(SUBGRP_DIR, 'blp_all_subcategories.csv'))

sub_data = []
for subcat, label, threat in [
    ('ESTROGENS', 'Estrogens\n$n$ = 13,697', 'Highest identity threat'),
    ('CONTRACEPTIVES', 'Contraceptives\n$n$ = 16,293', 'Moderate identity threat'),
    ('ANDROGENS_ANABOLIC', 'Androgens\n$n$ = 7,214', 'Lowest (physician-controlled)'),
]:
    r = sub_blp[(sub_blp['subcategory'] == subcat) &
                (sub_blp['dv'] == 'log_sales') &
                (sub_blp['variable'] == 'republicanStd_x_brand')]
    if len(r) > 0:
        r = r.iloc[0]
        sub_data.append((label, r['estimate'], r['std_error'], r['p_value'],
                         r['p_value'] < 0.10, threat))
    else:
        # Progestins — no interaction
        pass

# Add main G03 as reference
row = main_blp[(main_blp['dv'] == 'log_sales') &
               (main_blp['variable'] == 'republicanStd_x_brand')].iloc[0]
sub_data.insert(0, ('G03 Overall\n$n$ = 22,737', row['estimate'], row['std_error'],
                     row['p_value'], True, 'Reference'))

fig3, ax3 = plt.subplots(figsize=(8, 5))

# Color coding by identity threat
threat_colors = {
    'Reference': '#2C3E50',
    'Highest identity threat': '#C0392B',
    'Moderate identity threat': '#E67E22',
    'Lowest (physician-controlled)': C_NONSIG,
}

y_positions3 = list(range(len(sub_data) - 1, -1, -1))
for i, (label, beta, se, pval, is_sig, threat) in enumerate(sub_data):
    y = y_positions3[i]
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    color = threat_colors[threat]
    marker = 'D' if is_sig else 'o'

    ax3.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=2.0, zorder=2, solid_capstyle='round')
    ax3.plot(beta, y, marker=marker, color=color, markersize=8,
             markeredgecolor='white', markeredgewidth=0.8, zorder=3)

    # p-value annotation only (no threat labels)
    p_str = f'$p$ = {pval:.3f}'
    ax3.text(max(ci_hi, 0.05) + 0.015, y, p_str, va='center', fontsize=9, color=color)

ax3.axvline(0, color='#999999', linestyle='--', linewidth=0.8, zorder=1)

# Horizontal separator between G03 overall and subcategories
ax3.axhline(y=y_positions3[0] - 0.5, color='#DDDDDD', linewidth=0.8, linestyle='-')

ax3.set_yticks(y_positions3)
ax3.set_yticklabels([d[0] for d in sub_data], fontsize=10)
ax3.set_xlabel(r'BLP interaction coefficient ($\beta$, republicanStd $\times$ brand)', labelpad=8)
ax3.set_title('Figure 3. Within-G03 Subcategory Gradient\n(BLP Interaction on log Sales)',
              fontweight='bold', pad=12, fontfamily='sans-serif', fontsize=12)
ax3.tick_params(axis='y', length=0)

legend_el3 = [
    Line2D([0], [0], marker='D', color='grey', linestyle='None',
           markersize=7, markeredgecolor='white', label='$p$ < 0.10'),
    Line2D([0], [0], marker='o', color='grey', linestyle='None',
           markersize=7, markeredgecolor='white', label='$p$ \u2265 0.10'),
]
ax3.legend(handles=legend_el3, loc='lower right', framealpha=0.95,
           edgecolor='#CCCCCC', fontsize=9)

fig3.tight_layout()
save_fig(fig3, 'figure_3_within_g03_gradient')
plt.close(fig3)


# ═══════════════════════════════════════════════════════
# FIGURE S1 (supplement): Overall ITE Distribution
# ═══════════════════════════════════════════════════════
print('Figure S1 (Overall ITE)...')
figS1, axesS1 = plt.subplots(1, 3, figsize=(15, 4.5))
figS1.subplots_adjust(wspace=0.28, top=0.84)

for idx, (dv_name, panel_title, xlabel) in enumerate(DVS):
    ax = axesS1[idx]
    ite_df, ate_row = load_ite(dv_name)
    ites_full = ite_df['ite'].values
    ate_val, ate_se = ate_row['ate'], ate_row['se']
    ites_t, lo, hi = trim(ites_full, pct=1.0)

    kde = stats.gaussian_kde(ites_t, bw_method='scott')
    x_grid = np.linspace(lo, hi, 500)

    ax.hist(ites_t, bins=60, density=True, color='#D5D5D5', edgecolor='white',
            linewidth=0.3, zorder=1)
    ax.plot(x_grid, kde(x_grid), color='#333333', linewidth=1.6, zorder=3)
    ax.axvline(0, color='#999999', linestyle=':', linewidth=0.8, zorder=2)
    ax.axvline(ate_val, color=C_SIG, linestyle='-', linewidth=1.2, zorder=4)
    ax.axvspan(ate_val - 1.96 * ate_se, ate_val + 1.96 * ate_se,
               alpha=0.10, color=C_SIG, zorder=0)

    pct_neg = np.mean(ites_full < 0) * 100
    lines = [f'$N$ = {len(ites_full):,}', f'ATE = {ate_val:.4f}',
             f'SD = {np.std(ites_full):.4f}', f'% neg. = {pct_neg:.1f}%']
    ax.text(0.96, 0.96, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#BBBBBB', lw=0.6))

    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel('Density' if idx == 0 else '', labelpad=6)
    ax.set_title(panel_title, fontweight='bold', pad=10, fontfamily='sans-serif')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    ax.tick_params(length=3.5)

legend_s1 = [
    Line2D([0], [0], color='#333333', linewidth=1.6, label='KDE'),
    Line2D([0], [0], color=C_SIG, linewidth=1.2, label='ATE'),
    mpatches.Patch(facecolor=C_SIG, alpha=0.15, label='95% CI'),
    Line2D([0], [0], color='#999999', linestyle=':', linewidth=0.8, label='Zero'),
]
figS1.legend(handles=legend_s1, loc='upper center', ncol=4,
             bbox_to_anchor=(0.5, 0.97), frameon=True, edgecolor='#CCCCCC',
             fontsize=9, handlelength=2, columnspacing=2.5)
figS1.suptitle('Distribution of Individual Treatment Effects from Causal Forest',
               fontsize=13, fontweight='bold', y=1.05, fontfamily='sans-serif')
save_fig(figS1, 'figure_s1_ite_distributions')
plt.close(figS1)


# ═══════════════════════════════════════════════════════
# FIGURE S2 (supplement): ITE by Republican Tercile
# ═══════════════════════════════════════════════════════
print('Figure S2 (Republican tercile)...')
figS2, axesS2 = plt.subplots(1, 3, figsize=(15, 4.5))
figS2.subplots_adjust(wspace=0.28, top=0.84)

TERCILE_LABELS = ['Low', 'Medium', 'High']

for idx, (dv_name, panel_title, xlabel) in enumerate(DVS):
    ax = axesS2[idx]
    ite_df, _ = load_ite(dv_name)
    ite_df['rep_tercile'] = pd.qcut(
        ite_df['republican_percentage'], q=3, labels=TERCILE_LABELS)

    _, lo, hi = trim(ite_df['ite'].values, pct=1.0)
    x_grid = np.linspace(lo, hi, 500)

    means = {}
    for t_idx, tercile in enumerate(TERCILE_LABELS):
        ites_grp = ite_df.loc[ite_df['rep_tercile'] == tercile, 'ite'].values
        ites_t = ites_grp[(ites_grp >= lo) & (ites_grp <= hi)]
        kde = stats.gaussian_kde(ites_t, bw_method='scott')
        c = TERCILE_COLORS[t_idx]
        ax.fill_between(x_grid, kde(x_grid), alpha=0.12, color=c, zorder=t_idx)
        ax.plot(x_grid, kde(x_grid), color=c, linewidth=1.5, zorder=3 + t_idx)
        means[tercile] = np.mean(ites_grp)

    ax.axvline(0, color='#999999', linestyle=':', linewidth=0.8, zorder=0)
    lines = [f'{t}: {means[t]:.4f}' for t in TERCILE_LABELS]
    ax.text(0.96, 0.96, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#BBBBBB', lw=0.6))

    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel('Density' if idx == 0 else '', labelpad=6)
    ax.set_title(panel_title, fontweight='bold', pad=10, fontfamily='sans-serif')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    ax.tick_params(length=3.5)

tercile_ns = ite_df['rep_tercile'].value_counts()
legend_s2 = [
    Line2D([0], [0], color=TERCILE_COLORS[i], linewidth=1.5,
           label=f'{TERCILE_LABELS[i]} Republican ($n$ = {tercile_ns[TERCILE_LABELS[i]]:,})')
    for i in range(3)
]
figS2.legend(handles=legend_s2, loc='upper center', ncol=3,
             bbox_to_anchor=(0.5, 0.97), frameon=True, edgecolor='#CCCCCC',
             fontsize=9, handlelength=2, columnspacing=2)
figS2.suptitle('ITE Distributions by Republican Vote Share Tercile',
               fontsize=13, fontweight='bold', y=1.05, fontfamily='sans-serif')
save_fig(figS2, 'figure_s2_ite_by_republican')
plt.close(figS2)


# ═══════════════════════════════════════════════════════
# FIGURE S3 (supplement): Specification Curve
# ═══════════════════════════════════════════════════════
print('Figure S3 (Specification curve)...')

spec_df = pd.read_csv(SPEC_FILE)
spec_df = spec_df.sort_values('beta_sales').reset_index(drop=True)
spec_df['rank'] = range(1, len(spec_df) + 1)

figS3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6),
                                         gridspec_kw={'height_ratios': [3, 1]},
                                         sharex=True)
figS3.subplots_adjust(hspace=0.08)

# Top panel: coefficient + 95% CI
for _, row in spec_df.iterrows():
    r = row['rank']
    beta = row['beta_sales']
    se = row['se_sales']
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    is_sig = row['p_sales'] < 0.05
    color = C_SIG if is_sig else C_NONSIG

    ax_top.plot([r, r], [ci_lo, ci_hi], color=color, linewidth=0.8, alpha=0.5, zorder=1)
    ax_top.plot(r, beta, 'o', color=color, markersize=3, zorder=2)

ax_top.axhline(0, color='#999999', linestyle='--', linewidth=0.8, zorder=0)

# Highlight the production spec
prod_row = spec_df[spec_df['spec'].str.contains('evangelical', na=False)]
if len(prod_row) > 0:
    pr = prod_row.iloc[0]
    ax_top.plot(pr['rank'], pr['beta_sales'], 'D', color='#2C3E50',
                markersize=7, markeredgecolor='white', markeredgewidth=0.8, zorder=4)
    ax_top.annotate('Production\nspec (13 ctrl)',
                    xy=(pr['rank'], pr['beta_sales']),
                    xytext=(pr['rank'] + 8, pr['beta_sales'] + 0.015),
                    fontsize=8, color='#2C3E50',
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=0.8))

# Median beta line
median_beta = spec_df['beta_sales'].median()
ax_top.axhline(median_beta, color='#E67E22', linestyle=':', linewidth=1.0, alpha=0.7)
ax_top.text(len(spec_df) + 1, median_beta, f'median = {median_beta:.4f}',
            fontsize=8, va='bottom', color='#E67E22')

ax_top.set_ylabel(r'$\beta$ (republicanStd $\times$ brand)', labelpad=8)
ax_top.set_title('Specification Curve: BLP Interaction Coefficient Across 70 Specifications',
                 fontweight='bold', pad=10, fontfamily='sans-serif', fontsize=12)

n_sig = (spec_df['p_sales'] < 0.05).sum()
n_total = len(spec_df)
legend_s3 = [
    Line2D([0], [0], marker='o', color=C_SIG, linestyle='None', markersize=5,
           label=f'$p$ < 0.05 ({n_sig}/{n_total} specs)'),
    Line2D([0], [0], marker='o', color=C_NONSIG, linestyle='None', markersize=5,
           label=f'$p$ \u2265 0.05 ({n_total - n_sig}/{n_total} specs)'),
]
ax_top.legend(handles=legend_s3, loc='upper left', framealpha=0.95,
              edgecolor='#CCCCCC', fontsize=9)

# Bottom panel: p-value
for _, row in spec_df.iterrows():
    r = row['rank']
    is_sig = row['p_sales'] < 0.05
    color = C_SIG if is_sig else C_NONSIG
    ax_bot.plot(r, row['p_sales'], 'o', color=color, markersize=3, zorder=2)

ax_bot.axhline(0.05, color='#333333', linestyle='--', linewidth=0.8, zorder=0)
ax_bot.text(len(spec_df) + 1, 0.05, '$\\alpha$ = 0.05', fontsize=8, va='bottom')
ax_bot.set_ylabel('$p$-value', labelpad=8)
ax_bot.set_xlabel('Specification (ranked by coefficient)', labelpad=8)
ax_bot.set_ylim(-0.01, max(spec_df['p_sales']) * 1.1)
ax_bot.tick_params(length=3.5)
ax_top.tick_params(length=3.5)

# Add n_controls as color intensity or text
ax_bot.set_xlim(0, len(spec_df) + 3)

figS3.tight_layout()
save_fig(figS3, 'figure_s3_specification_curve')
plt.close(figS3)

print('\nAll figures done.')
