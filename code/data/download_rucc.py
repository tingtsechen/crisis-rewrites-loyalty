"""
Download USDA Rural-Urban Continuum Codes (RUCC) — used for the
`urban_rural_value_lessBetter` and `metro_indicator` controls.

Source: USDA Economic Research Service
    https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/

Output: data/external/USDA_RUCC_2023.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

RUCC_URL = ("https://ers.usda.gov/sites/default/files/_laserfiche/"
            "DataFiles/53251/Ruralurbancontinuumcodes2023.xlsx")

STUDY_STATES = ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external",
                        help="Output directory")
    parser.add_argument("--all-states", action="store_true",
                        help="Keep all 50 states (default: filter to study states)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {RUCC_URL} ...")
    try:
        resp = requests.get(RUCC_URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        print("  USDA URL may have changed. Check the current location at "
              "https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/",
              file=sys.stderr)
        sys.exit(1)

    raw_xlsx = out_dir / "Ruralurbancontinuumcodes2023.xlsx"
    raw_xlsx.write_bytes(resp.content)
    print(f"  raw saved to {raw_xlsx}")

    df = pd.read_excel(raw_xlsx, sheet_name="Rural-urban Continuum Code 2023")

    # FIPS column ('FIPS' or 'FIPS_code' depending on year)
    fips_col = next((c for c in df.columns if c.lower().startswith("fips")), None)
    rucc_col = next((c for c in df.columns if "RUCC" in c or "rural_urban" in c.lower()), None)
    state_col = next((c for c in df.columns if c.lower() in ("state", "state_abbr")), None)

    if not all([fips_col, rucc_col]):
        print(f"  WARNING: expected columns not found. Available: {list(df.columns)}",
              file=sys.stderr)
        sys.exit(2)

    df = df.rename(columns={fips_col: "fips", rucc_col: "urban_rural_value_lessBetter"})
    df["metro_indicator"] = (df["urban_rural_value_lessBetter"].astype(int) <= 3).astype(int)

    if not args.all_states and state_col:
        df = df[df[state_col].isin(STUDY_STATES)]

    keep = ["fips", "urban_rural_value_lessBetter", "metro_indicator"]
    if state_col:
        keep.insert(1, state_col)

    out_csv = out_dir / "USDA_RUCC_2023.csv"
    df[keep].to_csv(out_csv, index=False)
    print(f"  saved to {out_csv} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
