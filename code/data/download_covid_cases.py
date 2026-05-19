"""
Download NYT COVID-19 county-level cases/deaths for the study period
(April–June 2020).

Source: New York Times COVID-19 GitHub repository
    https://github.com/nytimes/covid-19-data

Output: data/external/county_covid_2020.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

URL = ("https://raw.githubusercontent.com/nytimes/covid-19-data/"
       "master/us-counties-2020.csv")

STUDY_STATES = ["Alabama", "Florida", "Georgia", "Kentucky",
                "Mississippi", "North Carolina", "South Carolina", "Tennessee"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external",
                        help="Output directory")
    parser.add_argument("--start", default="2020-04-01",
                        help="Start date (default: 2020-04-01)")
    parser.add_argument("--end", default="2020-06-30",
                        help="End date (default: 2020-06-30)")
    parser.add_argument("--all-states", action="store_true",
                        help="Keep all states (default: filter to study states)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {URL} ...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    raw_csv = out_dir / "us-counties-2020.csv"
    raw_csv.write_bytes(resp.content)
    print(f"  raw saved to {raw_csv}")

    df = pd.read_csv(raw_csv, parse_dates=["date"])
    df = df[(df["date"] >= args.start) & (df["date"] <= args.end)]
    if not args.all_states:
        df = df[df["state"].isin(STUDY_STATES)]
    df["fips"] = pd.to_numeric(df["fips"], errors="coerce")
    df = df.dropna(subset=["fips"])
    df["fips"] = df["fips"].astype(int)

    df = df.rename(columns={"date": "sld_dt", "state": "state_name"})
    out_csv = out_dir / "county_covid_2020.csv"
    df[["fips", "state_name", "county", "sld_dt", "cases", "deaths"]].to_csv(out_csv, index=False)
    print(f"  saved to {out_csv} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
