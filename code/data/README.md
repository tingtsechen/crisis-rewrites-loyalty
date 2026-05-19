# Public Data Download Scripts

These scripts fetch the public-source control variables used in the analyses, directly from primary providers. They are provided as a transparent substitute for redistributing the data ourselves, in line with standard data-availability practice for studies that rely on a mix of proprietary and public-domain sources.

## Scripts

| Script | Source | Use |
|---|---|---|
| `download_election_2020.py` | MIT Election Data + Science Lab (Harvard Dataverse) | Main political variable (Republican vote share) |
| `download_election_2016.py` | MIT Election Data + Science Lab | Robustness check: 2016 vs 2020 |
| `download_religion_census.py` | U.S. Religion Census 2020 (via ARDA) | `evangelical_pct` control |
| `download_rucc.py` | USDA Economic Research Service | `urban_rural_value_lessBetter`, `metro_indicator` |
| `download_acs_demographics.py` | U.S. Census Bureau ACS 5-year | Demographics, income, education, race |
| `download_covid_cases.py` | New York Times COVID-19 GitHub | `cases`, `deaths` |

## Usage

```bash
# Default: download all to data/external/
for script in code/data/download_*.py; do
    python "$script"
done

# Or one at a time with a custom output directory
python code/data/download_election_2020.py --output data/external
```

Each script will:
1. Fetch the raw file from the primary source
2. Apply minimal cleaning (filter to the 8 study states; harmonize FIPS)
3. Write a CSV to `data/external/` (or wherever specified)

## Reproducibility note

All public sources are versioned at their providers and have stable URLs as of the paper's submission date. If a URL changes, please open an issue at https://github.com/tingtsechen/crisis-rewrites-loyalty/issues and we will update the script.

## Dependencies

```bash
pip install requests pandas openpyxl
# For ACS:
pip install census us
```

The Census API requires a free API key from https://api.census.gov/data/key_signup.html. Set it as the environment variable `CENSUS_API_KEY` before running `download_acs_demographics.py`.
