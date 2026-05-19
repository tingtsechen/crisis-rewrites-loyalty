# Supplementary Analysis Code

This folder contains Python scripts for replicating the supplementary analyses.

## Scripts

| Script | Description | Output |
|--------|-------------|--------|
| `01_correlation_matrix.py` | Correlation matrix table and heatmap | Figure S1, Table S2 |
| `02_vif_calculation.py` | Variance Inflation Factors | Table S2b |
| `03_balance_check.py` | Covariate balance across High/Low Republican counties | Table S5 |
| `04_heterogeneity_income.py` | Income heterogeneity analysis | Tables S6a, S6b |
| `05_heterogeneity_urban_rural.py` | Urban-Rural heterogeneity analysis | Tables S7a, S7b |
| `06_parallel_trends.py` | Parallel trends visualization and test | Figure S2, Table S8 |

## Requirements

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels
```

## Usage

1. Update `DATA_PATH` in each script to point to your data file
2. Update `OUTPUT_DIR` to your desired output location
3. Run scripts:

```bash
python 01_correlation_matrix.py
python 02_vif_calculation.py
python 03_balance_check.py
python 04_heterogeneity_income.py
python 05_heterogeneity_urban_rural.py
python 06_parallel_trends.py
```

## Key Findings Summary

| Analysis | Key Result | Interpretation |
|----------|------------|----------------|
| VIF | All < 5.35 | No multicollinearity concern |
| Balance Check | Significant differences | Justifies control variables |
| Income Heterogeneity | p = 0.11 (subsample), p = 0.85 (3-way) | Effect not driven by price sensitivity |
| Urban-Rural Heterogeneity | p = 0.63 (subsample), p = 0.93 (3-way) | Effect not driven by access/availability |
| Parallel Trends | Slope = −0.001, p = 0.33 | Pre-COVID trends are parallel ✓ |

## Notes

- All models include full set of control variables
- State, month, and weekday fixed effects included
- Standard errors clustered at state level
- Subsample analyses use median splits
- Three-way interactions use standardized moderators
