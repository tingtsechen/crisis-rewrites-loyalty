# Data

The data underlying this paper come from three categories with different sharing arrangements. See the top-level [README.md](../README.md#data-availability) for the full Data Availability statement.

## What is (and isn't) in this folder

```
data/
├── README.md                       # ← you are here
├── synthetic/                      # ✅ committed: synthetic sample data (100 rows each)
│   ├── README.md
│   ├── synthetic_merged_sldDt_G03_daily_covid.csv
│   ├── synthetic_merged_sldDt_C01_daily_covid.csv
│   ├── synthetic_merged_sldDt_C02_daily_covid.csv
│   └── synthetic_merged_sldDt_C10_daily_covid.csv
│
├── final_data/                     # ❌ not committed: real proprietary panel (DUA-restricted)
├── processed/                      # ❌ not committed: derived intermediates from the panel
└── external/                       # ⬇️ created by code/data/download_*.py (public controls)
```

`.gitignore` excludes all CSV/Parquet/Excel files except those explicitly inside `data/synthetic/`.

## 1. Proprietary prescription panel — *not redistributable*

The main analyses use county-day-brand pharmaceutical claims data from a commercial provider. The Data Use Agreement prohibits public redistribution of the raw transactions and any derived panels.

**To request access**, contact the corresponding author (see the GitHub issue tracker). Access is granted at the data provider's discretion.

**For PNAS journal editorial review**, the full Code Ocean capsule containing the proprietary panel was submitted separately; the data editor has restricted access through that pipeline.

## 2. Public-domain controls — *fetchable via scripts in `code/data/`*

Run these to populate `data/external/`:

```bash
pip install requests pandas openpyxl census
python code/data/download_election_2020.py      # 2020 county presidential
python code/data/download_election_2016.py      # 2016 (robustness)
python code/data/download_religion_census.py    # USRC 2020 (evangelical share)
python code/data/download_rucc.py               # USDA Rural-Urban Continuum
python code/data/download_acs_demographics.py   # Census ACS 5-year (requires API key)
python code/data/download_covid_cases.py        # NYT COVID 2020
```

See `code/data/README.md` for source attribution and prerequisites.

## 3. Synthetic sample data — *for code execution verification*

`data/synthetic/` contains 100-row CSVs that match the schema of the real proprietary panel. See `data/synthetic/README.md` for full details.

> ⚠️ Statistical output from synthetic data is **NOT meaningful** and does NOT reproduce the paper's findings. The synthetic data exist only so that R / Python pipeline scripts can be exercised without DUA-restricted access.

## Variable definitions

A full data dictionary (variables, units, definitions, sources) is provided in [`docs/methods.md`](../docs/methods.md) and in the paper's SI Appendix Table A.1.
