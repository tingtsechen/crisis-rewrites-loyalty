# Crisis Rewrites Loyalty: How Political Ideology Reshapes Brand-to-Generic Substitution

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Replication code for the paper:

> Chen, T-T., Lu, S., & Ni, J. (2026). *Crisis Rewrites Loyalty: How Political Ideology Reshapes Brand-to-Generic Substitution.* Submitted to *Proceedings of the National Academy of Sciences*.

## Summary

Political identity is known to shape visible consumer choices, but whether it penetrates private decisions, where no audience observes the purchase, remains unclear. Using county-level pharmaceutical claims data from the acute phase of the COVID-19 health crisis (April–June 2019 vs 2020), we investigate whether political ideology shapes private consumption decisions. Using Generalized Random Forests with Best Linear Projection (GRF+BLP) inference, we find that county-level Republican vote share significantly moderated the crisis's effect on brand-name drug sales (β = −0.034, p = 0.044): in more conservative counties, brand-name sex hormone sales declined disproportionately relative to generics — a pattern consistent with politically patterned brand-to-generic substitution. The effect is absent in three politically neutral drug categories (all p > 0.40) and is precisely null within sex hormones where physician-directed prescribing limits patient choice. The paper develops and tests mechanism interpretations in full.

## Repository Structure

```
crisis-rewrites-loyalty/
├── code/
│   ├── R/                        # Primary pipeline (GRF + BLP, R)
│   ├── python/                   # Legacy CF (EconML) + figure generators
│   │   └── generate_synthetic_data.py    # Creates sample data with real schema
│   ├── data/                     # Scripts to download public control variables
│   └── validation/               # Supplementary validation tests
├── data/
│   ├── README.md                 # Data access information
│   └── synthetic/                # Synthetic sample data (committed, 100 rows each)
├── docs/
│   ├── methods.md                # Methods section
│   └── figure_captions.md        # All figure captions
├── environment/
│   ├── requirements.txt          # Python dependencies
│   └── R_packages.txt            # R dependencies
└── results/
    ├── tables/                   # Production model outputs (coefficients, SE, p-values)
    └── figures/                  # All main-text and supplementary figures (PDF/PNG)
```

## Data Availability

The data underlying this paper come from three categories, with different sharing arrangements.

### 1. Proprietary prescription panel — *not redistributable*

The county-day-brand prescription panel used in the main analyses is licensed from a commercial pharmaceutical data provider under a Data Use Agreement (DUA) that **prohibits public redistribution** of the raw transactions and any derived county-day-product panels. A restricted Code Ocean capsule containing the proprietary panel will be submitted alongside the manuscript for editorial review under restricted access. Researchers who wish to reproduce the full analysis can contact the corresponding author for guidance on requesting access from the data provider.

> Note: PNAS accepts proprietary-data papers; this Data Availability statement follows the convention for "proprietary or sensitive data that cannot be made public" per the PNAS [Editorial and Journal Policies](https://www.pnas.org/author-center/editorial-and-journal-policies).

### 2. Public-source control variables — *fetchable from primary providers*

All county-level control variables (political vote share, demographics, religion, COVID counts, urban-rural codes) are public-domain. Rather than redistributing them in this repository, we provide download scripts that fetch each dataset from its primary provider:

| Variable | Provider | Script |
|---|---|---|
| 2020 county presidential vote share | MIT Election Data + Science Lab / Tony McGovern repo | `code/data/download_election_2020.py` |
| 2016 county presidential vote share (robustness) | Same source | `code/data/download_election_2016.py` |
| Religion adherence (evangelical share) | U.S. Religion Census 2020 / ARDA | `code/data/download_religion_census.py` |
| Rural-Urban Continuum Codes | USDA Economic Research Service | `code/data/download_rucc.py` |
| Demographics, income, education, race | U.S. Census Bureau ACS 5-year API | `code/data/download_acs_demographics.py` |
| County COVID-19 cases/deaths | New York Times COVID-19 GitHub | `code/data/download_covid_cases.py` |

To fetch all public controls at once:

```bash
pip install requests pandas openpyxl census
for f in code/data/download_*.py; do python "$f"; done
```

See `code/data/README.md` for details (including the free Census API key required for ACS).

### 3. Synthetic sample data — *included in this repository, for code verification only*

To allow reviewers to confirm that all analysis scripts execute end-to-end without errors, we include a synthetic placeholder dataset at `data/synthetic/`. The 100-row samples share the same schema (column names, dtypes, structure) as the real proprietary panel, but values are drawn from independent random distributions.

> ⚠️ **Statistical results computed from these synthetic data are NOT meaningful and do NOT reproduce the paper's findings.** The synthetic data exist solely so that `Rscript code/R/grf_blp_production.R` and the Python pipeline scripts can be exercised in CI or by reviewers. See `data/synthetic/README.md`.

To regenerate, or to generate full-size synthetic panels matching the real N:

```bash
python code/python/generate_synthetic_data.py             # 100 rows each (default)
python code/python/generate_synthetic_data.py --full      # G03=22,737; C01=12,377; C02=20,113; C10=21,841
```

## Reproducibility

### Quick smoke test (synthetic data, ~1 min)

```bash
# 1. Make sure synthetic data is present (it is committed; this just regenerates)
python code/python/generate_synthetic_data.py --output data/synthetic

# 2. Run the figure generators against synthetic data to confirm pipeline works
python code/python/generate_model_free_figure.py
python code/python/generate_ite_distribution_figures.py
```

### Full reproduction (requires proprietary data access)

If you have access to the real prescription panel, place it under `data/final_data/` then run:

#### Primary pipeline (R, recommended)

```bash
# Install R dependencies
Rscript -e 'install.packages(readLines("environment/R_packages.txt"))'

# Table 1 — Main G03 BLP (~6 min)
Rscript code/R/grf_blp_production.R

# Table 2 — Between-category placebo, C01/C02/C10 (~18 min)
Rscript code/R/grf_blp_robustness_drug_categories.R

# Table 3 — Within-G03 subcategory analysis (~24 min)
Rscript code/R/grf_blp_subgroup_G03.R

# Supplementary robustness
Rscript code/R/grf_blp_robustness_election2016.R
Rscript code/R/grf_blp_robustness_moderators.R
Rscript code/R/grf_blp_healthcare_robustness.R
Rscript code/R/check_vif_religiosity.R
```

#### Legacy Python pipeline (robustness comparison)

```bash
# Install Python dependencies (pip format)
pip install -r environment/requirements.txt
# (or equivalently, with bash installer:)
# bash environment/install_python.sh

# Stage 1: Causal Forest (3,000 trees)
python code/python/codeOcean_CF_estiATE.py

# Stage 2: OLS on individual treatment effects
python code/python/codeOcean_analyze_cf_ite_ols.py

# Model-free descriptive statistics (input to Figure 1)
python code/python/codeOcean_model_free_stats.py
```

#### Figures

```bash
python code/python/generate_model_free_figure.py          # Figure 1
python code/python/generate_ite_distribution_figures.py   # Figures 2, S3, S4
python code/python/generate_parallel_trends_figure.py     # Figure S1
python code/python/generate_correlation_heatmap.py        # Supplementary correlation
```

#### Validation tests (Supplementary Information)

```bash
python code/validation/01_correlation_matrix.py
python code/validation/02_vif_calculation.py
python code/validation/03_balance_check.py
python code/validation/04_heterogeneity_income.py
python code/validation/05_heterogeneity_urban_rural.py
python code/validation/06_parallel_trends.py
```

## Model Specification

- **Method:** Generalized Random Forest (`grf::causal_forest`) + Best Linear Projection (`best_linear_projection`) from Semenova & Chernozhukov (2021).
- **Forest:** 3,000 trees, min.node.size = 15, cluster = county FIPS, seed = 42.
- **Treatment:** binary indicator (April–June 2020 = 1, April–June 2019 = 0).
- **Sample:** 22,737 county-day-product observations across 122 counties in 8 southeastern U.S. states.
- **Heterogeneity features (7):** standardized Republican vote share, brand indicator, their interaction, and four G03 subcategory volume indicators (androgens, estrogens, progestins, contraceptives).
- **Controls (13):** see `docs/methods.md` for the full list and selection rationale.
- **BLP A-matrix:** standardized Republican vote share + Republican × brand interaction.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chen2026crisis,
  title   = {Crisis Rewrites Loyalty: How Political Ideology Reshapes Brand-to-Generic Substitution},
  author  = {Chen, Ting-Tse and Lu, Steven and Ni, Jian},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2026},
  note    = {Under review}
}
```

A permanent code archive (with DOI) will be created on Zenodo at acceptance via GitHub Release integration.

## License

Code in this repository is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material for any purpose, provided you give appropriate credit. Public-domain data downloaded via `code/data/` scripts remain under their original providers' terms.

## Contact

For questions about the code or for instructions on requesting access to the proprietary prescription data, please open a GitHub issue or contact the corresponding author.
