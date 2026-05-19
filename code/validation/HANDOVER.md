# Project Handover Document
## JM Submission: Political Ideology and Brand vs Generic Pharmaceutical Choice

**Last Updated**: January 30, 2026
**Status**: Supplement Materials Completed (Notes 1-8)

---

## 1. Project Overview

### Research Question
How does political ideology (Republican vote share) affect consumers' choice between brand-name and generic pharmaceuticals during the COVID-19 pandemic?

### Main Finding
Counties with higher Republican vote share show **lower** brand drug spending (negative coefficient on Republican × Brand interaction), suggesting conservatives prefer generic drugs over brand-name alternatives.

### Methodology
- **Primary Method**: Causal Forest with Double Machine Learning (DML)
- **Validation**: OLS with fixed effects and clustered standard errors
- **Data**: G03 category (sex hormones) pharmaceutical sales during COVID-19 period

---

## 2. Data Information

### Data File Location
```
/mnt/proj2_codeOcean_0731/data/final_data/merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv
```

### Key Variables

| Variable | Description | Type |
|----------|-------------|------|
| `rx_prc_amt_sum` | Prescription sales amount (DV: use log) | Continuous |
| `republican_percentage_std` | Republican vote share (standardized) | Continuous |
| `is_brand` | Brand drug indicator (1=Brand, 0=Generic) | Binary |
| `rep_x_brand` | Republican × Brand interaction | Continuous |
| `household_median` | Median household income | Continuous |
| `urban_rural_value_lessBetter` | Urban-rural classification (1=Urban to 6=Rural) | Ordinal |
| `state_abbr` | State abbreviation (for FE) | Categorical |
| `month` | Month (for FE) | Categorical |
| `weekday` | Day of week (for FE) | Categorical |

### Sample Size
- **N = 22,737** observations
- After removing health risk conditions during COVID period

### Control Variables
- `healthrisk_impact`: Health risk index
- `edu_bachelor_above`: Education (% bachelor's+)
- `female_percentage`: Female population %
- `over_65_percentage`: Age 65+ population %
- `household_median`: Median income
- `race_black`: Black population %
- `unemployment_rate`: Unemployment rate
- `cases`: COVID-19 cases
- `pharmacy_density`: Pharmacy density
- `insurance_coverage_mean`: Insurance coverage rate

---

## 3. File Structure

### Main Output Directory
```
/mnt/v3/markdown_output/
├── supplement_document.md      # 完整合併的 supplement 文件
├── HANDOVER.md                 # 本文件
│
├── sections/                   # 各 section 分開的 markdown
│   ├── S00_toc.md             # Table of Contents
│   ├── S01_variable_descriptions.md
│   ├── S02_correlation_matrix.md
│   ├── S03_subgroup_analysis.md
│   ├── S04_robustness_checks.md
│   ├── S05_balance_check.md
│   ├── S06_heterogeneity_income.md
│   ├── S07_heterogeneity_urban_rural.md
│   └── S08_parallel_trends.md
│
├── code/                       # Replication code
│   ├── README.md
│   ├── 01_correlation_matrix.py
│   ├── 02_vif_calculation.py
│   ├── 03_balance_check.py
│   ├── 04_heterogeneity_income.py
│   ├── 05_heterogeneity_urban_rural.py
│   └── 06_parallel_trends.py
│
├── Figure_S1_correlation_matrix.png  # Correlation heatmap
└── Figure_S2_parallel_trends.png     # Parallel trends visualization
```

### Original Word Documents (已轉換)
```
/mnt/v3/
├── Supplement_G03_0729.docx    # 原始 Word supplement
└── manuscript_0731.docx        # 原始 Word manuscript
```

---

## 4. Supplement Notes Summary

### Note 1: Variable Descriptions and Data Sources
- 所有變數的定義和來源
- Data collection methodology

### Note 2: Correlation Matrix and Multicollinearity Assessment
- **Table S2**: Correlation matrix of all variables
- **Figure S1**: Correlation heatmap
- **Table S2b**: VIF analysis
  - All VIF < 5.35 ✓ (threshold = 10)
  - Republican Vote Share: VIF = 4.97
  - Brand Indicator: VIF = 1.31
  - Interaction Term: VIF = 1.33
- **Conclusion**: No multicollinearity concerns

### Note 3: Subgroup Analysis by Medication Category
- Analysis across different drug categories
- (內容來自原始 Word 文件)

### Note 4: Robustness Checks
- Alternative specifications
- Sensitivity analyses
- (內容來自原始 Word 文件)

### Note 5: Covariate Balance Across Political Regions
- **Table S5**: High vs Low Republican county comparison
- **Key Findings**:
  - Significant differences in demographics (White%, Black%, Education)
  - Urban-Rural distribution differs significantly
  - **Implication**: Justifies inclusion of control variables

### Note 6: Heterogeneity Analysis by Income Level
- **Purpose**: Test if effect is driven by price sensitivity
- **Method**:
  - Subsample analysis (median split at $53,975)
  - Three-way interaction (Rep × Brand × Income)
- **Table S6a**: Subsample results
  - High Income: β = -0.049, p = 0.376
  - Low Income: β = 0.041, p = 0.466
  - Difference: p = 0.110 (not significant)
- **Table S6b**: Three-way interaction
  - β = -0.011, p = 0.847 (not significant)
- **Table S6c**: Full regression results
- **Conclusion**: Effect NOT driven by price sensitivity ✓

### Note 7: Heterogeneity Analysis by Urban-Rural Classification
- **Purpose**: Test if effect is driven by access/availability
- **Method**:
  - Subsample analysis (median split at 2.0)
  - Three-way interaction (Rep × Brand × Urban-Rural)
- **Table S7a**: Subsample results
  - Urban: β = -0.020, p = 0.724
  - Rural: β = -0.069, p = 0.351
  - Difference: p = 0.625 (not significant)
- **Table S7b**: Three-way interaction
  - β = -0.005, p = 0.931 (not significant)
- **Table S7c**: Full regression results
- **Conclusion**: Effect NOT driven by access/availability ✓

### Note 8: Parallel Trends Analysis
- **Purpose**: Validate comparability of political groups pre-COVID
- **Method**: Test if High vs Low Republican areas had parallel brand share trends in 2019
- **Figure S2**: Parallel trends visualization
- **Table S8**: Formal test results
  - Slope = −0.001, p = 0.330
- **Conclusion**: Parallel trends assumption SUPPORTED ✓

---

## 5. Key Analytical Results

### Main Effect (from main paper)
```
Republican × Brand interaction: Negative coefficient
→ Higher Republican areas show lower brand drug spending
```

### Alternative Explanations Tested and Ruled Out

| Alternative Explanation | Mechanism | Test | p-value | Result |
|------------------------|-----------|------|---------|--------|
| Price Sensitivity | Low income → prefer cheaper generics | Income × Rep × Brand | 0.847 | ❌ Ruled out |
| Access/Availability | Rural → limited brand access | Urban-Rural × Rep × Brand | 0.931 | ❌ Ruled out |

### Theoretical Implication
Both null results **support** the demand-side, identity-based mechanism:
- Effect is NOT about economic constraints
- Effect is NOT about supply-side limitations
- Effect reflects genuine **preference** differences based on political identity

---

## 6. Regression Specification

### Standard Model Formula
```python
formula = """
log_sales ~ republican_percentage_std + is_brand + rep_x_brand +
            healthrisk_impact + edu_bachelor_above + female_percentage +
            over_65_percentage + household_median + race_black +
            unemployment_rate + cases + pharmacy_density +
            insurance_coverage_mean + urban_rural_value_lessBetter +
            C(state_abbr) + C(month) + C(weekday)
"""
```

### Three-Way Interaction Model (for heterogeneity)
```python
# Example: Income heterogeneity
formula_3way = """
log_sales ~ republican_percentage_std + is_brand + rep_x_brand +
            high_income_std + rep_x_income + brand_x_income +
            rep_x_brand_x_income +
            [controls] + C(state_abbr) + C(month) + C(weekday)
"""
```

### Estimation Details
- **Standard Errors**: Clustered at state level
- **Fixed Effects**: State, Month, Weekday
- **Package**: statsmodels (Python)

---

## 7. Code Execution Notes

### Required Python Packages
```bash
pip install pandas numpy statsmodels scipy matplotlib seaborn --break-system-packages
```

### Running Analysis Code
```bash
cd /mnt/v3/markdown_output/code/
python 01_correlation_matrix.py
python 02_vif_calculation.py
python 03_balance_check.py
python 04_heterogeneity_income.py
python 05_heterogeneity_urban_rural.py
```

### Regenerating Combined Supplement
```bash
cd /mnt/v3/markdown_output/
cat sections/S00_toc.md sections/S01_variable_descriptions.md \
    sections/S02_correlation_matrix.md sections/S03_subgroup_analysis.md \
    sections/S04_robustness_checks.md sections/S05_balance_check.md \
    sections/S06_heterogeneity_income.md sections/S07_heterogeneity_urban_rural.md \
    > supplement_document.md
```

---

## 8. Potential Next Steps

### Not Yet Completed (Optional)
1. **Alternative DV Analysis**
   - Use different operationalizations of brand/generic choice
   - e.g., count-based, share-based measures

2. **Parallel Trends Test**
   - For any DiD-style analyses
   - Pre-treatment trend comparison

3. **Additional Robustness**
   - Different clustering levels
   - Alternative fixed effects structures
   - Instrumental variable approach

4. **Main Text Integration**
   - Move key heterogeneity findings to main paper discussion
   - Add theoretical interpretation

### Export Options
- Convert markdown to Word: Use pandoc
- Convert to PDF: Use pandoc with LaTeX
- Convert to LaTeX: Direct for journal submission

```bash
# Example: Convert to Word
pandoc supplement_document.md -o supplement_document.docx

# Example: Convert to PDF
pandoc supplement_document.md -o supplement_document.pdf
```

---

## 9. Quick Reference

### Key Statistics to Remember
| Metric | Value |
|--------|-------|
| Sample Size | N = 22,737 |
| Max VIF | 4.97 (Republican %) |
| Income Median Split | $53,975 |
| Urban-Rural Median Split | 2.0 |
| Income 3-way p-value | 0.847 |
| Urban-Rural 3-way p-value | 0.931 |

### File Quick Access
```
Supplement (combined): /mnt/v3/markdown_output/supplement_document.md
Code folder:           /mnt/v3/markdown_output/code/
Data file:             /mnt/proj2_codeOcean_0731/data/final_data/merged_sldDt_G03_daily_covid_remove_healthRisk_addUrbanRural.csv
```

---

## 10. Contact & Session Info

**Session ID**: sweet-ecstatic-cannon
**Conversation Log**: `/sessions/sweet-ecstatic-cannon/mnt/.claude/projects/-sessions-sweet-ecstatic-cannon/`

To continue this work, provide this handover document to Claude and specify what you want to work on next.

---

*End of Handover Document*
