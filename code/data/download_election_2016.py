"""
Download 2016 U.S. county-level presidential election results (used for
robustness check `grf_blp_robustness_election2016.R`).

Source: Tony McGovern's `US_County_Level_Election_Results_08-24` repo.

Output: data/external/county_election_2016.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

URL = ("https://raw.githubusercontent.com/tonmcg/"
       "US_County_Level_Election_Results_08-24/master/2016_US_County_Level_Presidential_Results.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {URL} ...")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    out_path = out_dir / "2016_US_County_Level_Presidential_Results.csv"
    out_path.write_bytes(resp.content)
    print(f"  saved to {out_path} ({len(resp.content):,} bytes)")


if __name__ == "__main__":
    main()
