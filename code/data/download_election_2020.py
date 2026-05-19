"""
Download 2020 U.S. county-level presidential election results.

Source: Tony McGovern's `US_County_Level_Election_Results_08-24` repository,
which aggregates official secretary-of-state county-level returns and is
widely cited in academic work (see also MIT Election Data + Science Lab on
Harvard Dataverse, DOI 10.7910/DVN/VOQCHQ for an alternative source).

Output: data/external/county_election_2020.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

URL = ("https://raw.githubusercontent.com/tonmcg/"
       "US_County_Level_Election_Results_08-24/master/2020_US_County_Level_Presidential_Results.csv")

STUDY_STATES = ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external",
                        help="Output directory (default: data/external)")
    parser.add_argument("--all-states", action="store_true",
                        help="Keep all 50 states (default: filter to study states)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {URL} ...")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    out_raw = out_dir / "2020_US_County_Level_Presidential_Results.csv"
    out_raw.write_bytes(resp.content)
    print(f"  raw saved to {out_raw}")

    df = pd.read_csv(out_raw)
    df.rename(columns={"county_fips": "fips"}, inplace=True)

    df["republican_percentage"] = df["votes_gop"] / (df["votes_gop"] + df["votes_dem"]) * 100

    if not args.all_states:
        df = df[df["state_name"].isin([
            "Alabama", "Florida", "Georgia", "Kentucky",
            "Mississippi", "North Carolina", "South Carolina", "Tennessee",
        ])]

    keep = ["fips", "state_name", "county_name", "votes_gop", "votes_dem",
            "total_votes", "republican_percentage"]
    keep = [c for c in keep if c in df.columns]
    out_clean = out_dir / "county_election_2020.csv"
    df[keep].to_csv(out_clean, index=False)
    print(f"  cleaned ({len(df):,} rows) saved to {out_clean}")


if __name__ == "__main__":
    main()
