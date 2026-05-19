"""
Generate Figure 1: Model-Free Evidence of Politically Polarized Brand Switching.

Panel (A): Change in brand-name drug market share (percentage points)
           by political group (Democrat-leaning vs Republican-leaning).
Panel (B): Change in transaction volume (%) by political group and drug type.

Counties classified into terciles of 2020 Republican vote share:
Democrat-leaning (bottom third), Republican-leaning (top third).
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
DATA_PATH = os.path.join(
    PROJ_ROOT, 'data', 'processed',
    'merged_data_Gclass_SldDt_04-06_2019-2020_8_states_drugCategory_county_election2020_ODY-loc.csv'
)
OUT_DIR = SCRIPT_DIR

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

BAR_COLOR = '#7B8FA1'
C_BRAND = '#1B2A4A'
C_GENERIC = '#8B1A1A'


def load_and_classify(path: str) -> pd.DataFrame:
    """Load transaction data and classify counties by political leaning."""
    print(f'Loading data from: {path}')
    df = pd.read_csv(path, usecols=[
        'brand_generic', 'gpi_2_desc', 'republican_percentage',
        'year', 'fips'
    ])

    # Filter to G03 subcategories
    g03_cats = ['ANDROGENS-ANABOLIC', 'CONTRACEPTIVES', 'ESTROGENS', 'PROGESTINS']
    df = df[df['gpi_2_desc'].isin(g03_cats)].copy()

    # Classify counties by terciles of republican_percentage
    terciles = df['republican_percentage'].quantile([0, 0.33, 0.67, 1])
    df['political_group'] = pd.cut(
        df['republican_percentage'],
        bins=terciles,
        labels=['Democrat-leaning', 'Competitive', 'Republican-leaning'],
        include_lowest=True,
    )

    print(f'  Total G03 transactions: {len(df):,}')
    for grp in ['Democrat-leaning', 'Republican-leaning']:
        n = (df['political_group'] == grp).sum()
        print(f'  {grp}: {n:,} transactions')

    return df


def compute_brand_share_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute change in brand market share (percentage points) by political group."""
    results = []
    for grp in ['Democrat-leaning', 'Republican-leaning']:
        sub = df[df['political_group'] == grp]

        for yr in [2019, 2020]:
            yr_data = sub[sub['year'] == yr]
            n_total = len(yr_data)
            n_brand = (yr_data['brand_generic'] == 'B').sum()
            share = n_brand / n_total * 100 if n_total > 0 else 0
            se = np.sqrt(share / 100 * (1 - share / 100) / n_total) * 100 if n_total > 0 else 0

            results.append({
                'group': grp, 'year': yr,
                'brand_share': share, 'se': se, 'n': n_total,
            })

    res = pd.DataFrame(results)

    # Compute change (2020 - 2019) and SE of difference
    changes = []
    for grp in ['Democrat-leaning', 'Republican-leaning']:
        pre = res[(res['group'] == grp) & (res['year'] == 2019)].iloc[0]
        post = res[(res['group'] == grp) & (res['year'] == 2020)].iloc[0]
        change = post['brand_share'] - pre['brand_share']
        se_diff = np.sqrt(pre['se'] ** 2 + post['se'] ** 2)
        changes.append({'group': grp, 'change': change, 'se': se_diff})
        print(f'  {grp}: {pre["brand_share"]:.2f}% -> {post["brand_share"]:.2f}% '
              f'(Δ = {change:+.2f} pp, SE = {se_diff:.2f})')

    return pd.DataFrame(changes)


def compute_transaction_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute % change in transaction volume by political group and drug type."""
    results = []
    for grp in ['Democrat-leaning', 'Republican-leaning']:
        for dtype, label in [('B', 'Brand'), ('G', 'Generic')]:
            sub = df[(df['political_group'] == grp) & (df['brand_generic'] == dtype)]
            n_pre = (sub['year'] == 2019).sum()
            n_post = (sub['year'] == 2020).sum()
            pct_change = (n_post - n_pre) / n_pre * 100 if n_pre > 0 else 0
            results.append({
                'group': grp, 'type': label,
                'pct_change': pct_change, 'n_pre': n_pre, 'n_post': n_post,
            })
            print(f'  {grp} {label}: {n_pre:,} -> {n_post:,} ({pct_change:+.1f}%)')

    return pd.DataFrame(results)


def plot_figure(brand_share: pd.DataFrame, txn_change: pd.DataFrame) -> None:
    """Generate the two-panel figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.35)

    # ── Panel A: Brand Share Change ──
    groups = brand_share['group'].tolist()
    x = np.arange(len(groups))
    bars = ax1.bar(x, brand_share['change'], width=0.5,
                   color=BAR_COLOR, edgecolor='white', linewidth=0.5)
    ax1.errorbar(x, brand_share['change'], yerr=1.96 * brand_share['se'],
                 fmt='none', color='#333333', capsize=4, capthick=1.2, linewidth=1.2)

    # Value labels
    for i, (val, se) in enumerate(zip(brand_share['change'], brand_share['se'])):
        offset = -0.08 if val < 0 else 0.05
        ax1.text(i, val + offset, f'{val:.2f}', ha='center', va='top' if val < 0 else 'bottom',
                 fontsize=10, color='#333333')

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel('Change in Brand Share (Percentage Points)')
    ax1.set_xlabel('Political Group')
    ax1.set_title('a) Brand Market Share Change', fontweight='bold', loc='left', pad=10)
    ax1.axhline(0, color='#999999', linewidth=0.5, zorder=0)

    # Symmetric y-axis
    y_max = max(abs(brand_share['change'].min()), abs(brand_share['change'].max())) * 1.3
    ax1.set_ylim(-y_max, y_max)

    # ── Panel B: Transaction Volume Change ──
    bar_width = 0.35
    groups_b = ['Democrat-leaning', 'Republican-leaning']
    x2 = np.arange(len(groups_b))

    for i, grp in enumerate(groups_b):
        brand_val = txn_change[(txn_change['group'] == grp) & (txn_change['type'] == 'Brand')]['pct_change'].values[0]
        generic_val = txn_change[(txn_change['group'] == grp) & (txn_change['type'] == 'Generic')]['pct_change'].values[0]

        ax2.bar(x2[i] - bar_width / 2, brand_val, bar_width,
                color=C_BRAND, edgecolor='white', linewidth=0.5, label='Brand' if i == 0 else '')
        ax2.bar(x2[i] + bar_width / 2, generic_val, bar_width,
                color=C_GENERIC, edgecolor='white', linewidth=0.5, label='Generic' if i == 0 else '')

        # Value labels
        ax2.text(x2[i] - bar_width / 2, brand_val - 1.5, f'{brand_val:.1f}%',
                 ha='center', va='top', fontsize=9, color='white', fontweight='bold')
        ax2.text(x2[i] + bar_width / 2, generic_val - 1.5, f'{generic_val:.1f}%',
                 ha='center', va='top', fontsize=9, color='white', fontweight='bold')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(groups_b)
    ax2.set_ylabel('Total Transaction Volume Change (%)')
    ax2.set_xlabel('Political Group')
    ax2.set_title('b) Transaction Volume Change', fontweight='bold', loc='left', pad=10)
    ax2.axhline(0, color='#999999', linewidth=0.5, zorder=0)
    ax2.legend(loc='lower right', frameon=True, edgecolor='#CCCCCC', fontsize=9)

    # Save
    for ext in ['pdf', 'png']:
        path = os.path.join(OUT_DIR, f'figure_1_model_free_evidence.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'\n  Saved: figure_1_model_free_evidence.pdf/.png')
    plt.close(fig)


def main() -> None:
    df = load_and_classify(DATA_PATH)

    print('\n── Brand Share Change ──')
    brand_share = compute_brand_share_change(df)

    print('\n── Transaction Volume Change ──')
    txn_change = compute_transaction_change(df)

    print('\n── Generating Figure ──')
    plot_figure(brand_share, txn_change)


if __name__ == '__main__':
    main()
