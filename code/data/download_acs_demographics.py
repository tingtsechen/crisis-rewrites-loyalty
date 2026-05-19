"""
Download U.S. Census Bureau ACS 5-year (2016-2020) county-level demographics
for the eight study states.

Source: U.S. Census Bureau ACS API (https://api.census.gov/data)

Requires a free Census API key: https://api.census.gov/data/key_signup.html
Set as environment variable:  export CENSUS_API_KEY="your-key-here"

Output: data/external/acs_demographics_2020.csv

Variables fetched:
- B01003_001E: Total population
- B01001_*    : Age structure → over_65_percentage, male/female_percentage
- B15003_*    : Education attainment → edu_*
- B19013_001E: Median household income
- B19001_*    : Income distribution → income_*_percentage
- B02001_*    : Race counts → race_*
- B23025_*    : Employment → unemployment_rate, employment_rate
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ACS_BASE = "https://api.census.gov/data/2020/acs/acs5"

STUDY_STATE_FIPS = {
    "01": "Alabama", "12": "Florida", "13": "Georgia", "21": "Kentucky",
    "28": "Mississippi", "37": "North Carolina", "45": "South Carolina", "47": "Tennessee",
}

# Subset of variables; expand as needed.
VARIABLES = [
    "B01003_001E",   # Total population
    "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",  # Male 65+
    "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",  # Female 65+
    "B19013_001E",   # Median household income
    "B23025_005E",   # Unemployed (in labor force)
    "B23025_002E",   # In labor force
    "B02001_002E",   # White alone
    "B02001_003E",   # Black alone
    "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",  # Bachelor's+
    "B15003_001E",   # Education total denominator
]


def fetch_state(state_fips: str, api_key: str) -> pd.DataFrame:
    params = {
        "get": ",".join(VARIABLES) + ",NAME",
        "for": "county:*",
        "in": f"state:{state_fips}",
        "key": api_key,
    }
    resp = requests.get(ACS_BASE, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    return pd.DataFrame(rows[1:], columns=rows[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external",
                        help="Output directory")
    args = parser.parse_args()

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("ERROR: CENSUS_API_KEY environment variable not set.", file=sys.stderr)
        print("  Get a free key at https://api.census.gov/data/key_signup.html",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for sf, sname in STUDY_STATE_FIPS.items():
        print(f"Fetching ACS for {sname} ({sf}) ...")
        df = fetch_state(sf, api_key)
        df["state_abbr"] = list(STUDY_STATE_FIPS.values()).count  # placeholder
        df["state_fips"] = sf
        df["state_name"] = sname
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["fips"] = raw["state"].astype(str).str.zfill(2) + raw["county"].astype(str).str.zfill(3)
    raw["fips"] = raw["fips"].astype(int)

    # Convert numeric columns
    for v in VARIABLES:
        raw[v] = pd.to_numeric(raw[v], errors="coerce")

    # Build derived variables
    pop = raw["B01003_001E"]
    over_65 = raw[[c for c in VARIABLES if c.startswith("B01001_0") and
                   c in ["B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E",
                         "B01001_024E", "B01001_025E", "B01001_044E", "B01001_045E",
                         "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E"]]].sum(axis=1)
    bachelor_above = raw[["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]].sum(axis=1)
    edu_total = raw["B15003_001E"]

    out = pd.DataFrame({
        "fips": raw["fips"],
        "state_name": raw["state_name"],
        "total_county_population": pop,
        "over_65_percentage": (over_65 / pop * 100).round(3),
        "household_median": raw["B19013_001E"],
        "unemployment_rate": (raw["B23025_005E"] / raw["B23025_002E"] * 100).round(3),
        "race_white": (raw["B02001_002E"] / pop * 100).round(3),
        "race_black": (raw["B02001_003E"] / pop * 100).round(3),
        "edu_bachelor_above": (bachelor_above / edu_total * 100).round(3),
    })

    out_csv = out_dir / "acs_demographics_2020.csv"
    out.to_csv(out_csv, index=False)
    print(f"  saved to {out_csv} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
