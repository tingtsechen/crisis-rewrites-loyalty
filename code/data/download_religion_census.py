"""
Download 2020 U.S. Religion Census data for the `evangelical_pct` control.

Source: U.S. Religion Census 2020 (https://www.usreligioncensus.org).
The raw files are also archived at the Association of Religion Data Archives
(ARDA): https://www.thearda.com/Archive/Files/Descriptions/USRC2020.asp

This script downloads the published Excel files and extracts evangelical
shares at the county level.

Output: data/external/county_religiosity_2020.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

# Primary download URLs (verified at time of submission).
# If these change, please open an issue at the repo.
USRC_SUMMARY_URL = ("https://www.usreligioncensus.org/sites/default/files/"
                    "2023-10/2020_USRC_Summaries.xlsx")
USRC_DETAIL_URL = ("https://www.usreligioncensus.org/sites/default/files/"
                   "2023-10/2020_USRC_Group_Detail.xlsx")


def fetch(url: str, dest: Path) -> None:
    print(f"Fetching {url} ...")
    resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  saved to {dest} ({len(resp.content):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_xlsx = out_dir / "2020_USRC_Summaries.xlsx"
    detail_xlsx = out_dir / "2020_USRC_Group_Detail.xlsx"

    try:
        fetch(USRC_SUMMARY_URL, summary_xlsx)
        fetch(USRC_DETAIL_URL, detail_xlsx)
    except requests.HTTPError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        print("  USRC URLs may have moved. Please check https://www.usreligioncensus.org "
              "or download manually and re-run this script with the files already in --output.",
              file=sys.stderr)
        sys.exit(1)

    # Construct evangelical_pct from the Summaries sheet.
    # The USRC schema separates 'Evangelical Protestant', 'Mainline Protestant',
    # 'Catholic', etc.; we use the Evangelical Protestant adherents share.
    print("Building county_religiosity_2020.csv ...")
    df = pd.read_excel(summary_xlsx, sheet_name=0)

    # Standardize column names (USRC files use spaces and capitalization)
    df.columns = [c.strip() for c in df.columns]

    # Map likely-named columns (check exact names in your downloaded file)
    fips_col = next((c for c in df.columns if "fips" in c.lower()), None)
    pop_col = next((c for c in df.columns if "total" in c.lower() and "pop" in c.lower()), None)
    evang_col = next((c for c in df.columns if "evangelical" in c.lower()
                      and "adherent" in c.lower()), None)

    if not all([fips_col, pop_col, evang_col]):
        print("  WARNING: expected columns not auto-detected. Inspect the xlsx headers "
              "and adapt this script.", file=sys.stderr)
        print(f"  Available columns: {list(df.columns)[:20]}", file=sys.stderr)
        sys.exit(2)

    df["evangelical_pct"] = df[evang_col] / df[pop_col] * 100

    out_csv = out_dir / "county_religiosity_2020.csv"
    df[[fips_col, "evangelical_pct"]].rename(columns={fips_col: "fips"}).to_csv(out_csv, index=False)
    print(f"  saved to {out_csv} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
