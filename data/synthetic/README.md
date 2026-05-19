# Synthetic Sample Data

> ⚠️ **For code execution verification only. NOT used in any analyses reported in the paper.**

## What is this?

This folder contains synthetic county-day panel datasets with the **same schema** (column names, data types, row structure) as the real proprietary prescription data used in our analyses, but with **values drawn from independent random distributions**.

```
data/synthetic/
├── synthetic_merged_sldDt_G03_daily_covid.csv      # 100 rows, 62 cols (matches real G03 schema)
├── synthetic_merged_sldDt_C01_daily_covid.csv      # 100 rows, 61 cols
├── synthetic_merged_sldDt_C02_daily_covid.csv      # 100 rows, 61 cols
└── synthetic_merged_sldDt_C10_daily_covid.csv      # 100 rows, 61 cols
```

## Why?

The real prescription panel is governed by a Data Use Agreement (DUA) and **cannot be publicly redistributed**. To allow reviewers and readers to verify that all R and Python analysis scripts execute end-to-end without errors, we provide this synthetic placeholder.

## What it allows

✅ Confirm that R `grf` + BLP pipeline scripts run without syntax/dependency errors
✅ Confirm Python `econml` legacy pipeline scripts execute end-to-end
✅ Inspect input expected by each analysis script (column names, dtypes, value ranges)
✅ Test the figure-generation pipeline (`code/python/generate_*_figure.py`)

## What it does NOT do

❌ Reproduce the coefficients, standard errors, p-values, or figures from the paper
❌ Provide any inference about the real-world phenomena studied
❌ Serve as a benchmark for the proprietary data

Any statistical output produced from this synthetic dataset is **mathematically valid given the random data, but substantively meaningless**. Do not interpret causal-forest estimates, BLP coefficients, or ITE distributions derived from these data as related to the paper's findings.

## How to generate

The included CSVs are sized at **100 rows each** (small enough to commit to git). To regenerate or produce full-size synthetic panels (matching the real N), run:

```bash
# Small mode (100 rows each, default)
python code/python/generate_synthetic_data.py --output data/synthetic

# Full mode (matches real N: G03=22,737; C01=12,377; C02=20,113; C10=21,841)
python code/python/generate_synthetic_data.py --output data/synthetic --full

# Reproducible with explicit seed
python code/python/generate_synthetic_data.py --seed 42
```

See `code/python/generate_synthetic_data.py` for the simulation details.

## Accessing the real data

The proprietary prescription panel access process is described in the top-level `README.md` Data Availability section. Public-source control variables can be obtained via the scripts under `code/data/`.
